#!/usr/bin/python3.9
# coding: utf-8
"""One-shot GPU YOLO-Pose action recognizer for owner search.

Keeps the same result contract as the upstream owner-search task:
waving, sitting, lying, lying_ground, falling, unknown.
"""
import json
import math
import threading
import time

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class YoloPoseActionRecognition:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.result_topic = rospy.get_param("~result_topic", "/owner_voice_reid_test/action_result")
        self.model_path = rospy.get_param(
            "~model_path", "/home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/models/pose/yolo11n-pose.pt"
        )
        self.device = rospy.get_param("~device", "cuda:0")
        self.image_size = int(rospy.get_param("~image_size", 416))
        self.confidence = float(rospy.get_param("~confidence", 0.30))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.capture_seconds = max(1.0, float(rospy.get_param("~capture_seconds", 7.0)))
        self.sample_count = max(6, int(rospy.get_param("~sample_count", 18)))
        self.min_samples = max(4, int(rospy.get_param("~min_samples", 6)))
        self.owner_roi = self.parse_roi(rospy.get_param("~owner_roi", ""))
        self.roi_padding = max(0.0, float(rospy.get_param("~roi_padding", 0.35)))
        self.bridge, self.lock, self.latest_frame = CvBridge(), threading.Lock(), None
        self.pub = rospy.Publisher(self.result_topic, String, queue_size=1, latch=True)
        try:
            from ultralytics import YOLO
            import torch
            if str(self.device).startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            self.model = YOLO(self.model_path)
            rospy.loginfo("YOLO-Pose action model ready: model=%s device=%s", self.model_path, self.device)
        except Exception as exc:
            self.model = None
            self.init_error = str(exc)
            rospy.logerr("YOLO-Pose action model unavailable: %s", exc)
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 24)
        threading.Thread(target=self.run_once, daemon=True).start()

    @staticmethod
    def parse_roi(value):
        try:
            values = [int(float(item.strip())) for item in str(value).split(",")]
            return values if len(values) == 4 else None
        except (TypeError, ValueError):
            return None

    def image_callback(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            with self.lock:
                self.latest_frame = frame.copy()
        except Exception as exc:
            rospy.logwarn_throttle(3.0, "YOLO-Pose image conversion failed: %s", exc)

    def frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def crop_owner(self, frame):
        if not self.owner_roi:
            return frame, (0, 0)
        x1, y1, x2, y2 = self.owner_roi
        height, width = frame.shape[:2]
        pad_x, pad_y = int((x2-x1) * self.roi_padding), int((y2-y1) * self.roi_padding)
        x1, x2 = max(0, x1-pad_x), min(width, x2+pad_x)
        y1, y2 = max(0, y1-pad_y), min(height, y2+pad_y)
        if x2 <= x1 or y2 <= y1:
            return frame, (0, 0)
        return frame[y1:y2, x1:x2], (x1, y1)

    @staticmethod
    def angle(a, b, c):
        if a is None or b is None or c is None:
            return None
        ab, cb = (a[0]-b[0], a[1]-b[1]), (c[0]-b[0], c[1]-b[1])
        denom = math.hypot(*ab) * math.hypot(*cb)
        if denom < 1e-6:
            return None
        cosine = max(-1.0, min(1.0, (ab[0]*cb[0] + ab[1]*cb[1]) / denom))
        return math.degrees(math.acos(cosine))

    def feature(self, frame):
        crop, _offset = self.crop_owner(frame)
        results = self.model.predict(crop, imgsz=self.image_size, conf=self.confidence,
                                     iou=self.iou, device=self.device, half=True,
                                     max_det=2, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0 or results[0].keypoints is None:
            return None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        keypoints = results[0].keypoints.data.cpu().numpy()
        # The owner ROI normally contains one person; choose the largest high-score pose.
        best = max(range(len(boxes)), key=lambda i: float(confidences[i]) * max(1.0, (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])))
        x1, y1, x2, y2 = [float(v) for v in boxes[best]]
        width, height = max(1.0, x2-x1), max(1.0, y2-y1)
        points = keypoints[best]
        def point(index):
            if index >= len(points) or len(points[index]) < 3 or float(points[index][2]) < 0.25:
                return None
            return (float(points[index][0]), float(points[index][1]))
        left_shoulder, right_shoulder = point(5), point(6)
        left_elbow, right_elbow = point(7), point(8)
        left_wrist, right_wrist = point(9), point(10)
        left_hip, right_hip = point(11), point(12)
        left_knee, right_knee = point(13), point(14)
        left_ankle, right_ankle = point(15), point(16)
        shoulders = [p for p in (left_shoulder, right_shoulder) if p]
        hips = [p for p in (left_hip, right_hip) if p]
        shoulder_y = sum(p[1] for p in shoulders) / len(shoulders) if shoulders else None
        hip_y = sum(p[1] for p in hips) / len(hips) if hips else None
        wrist_x = [p[0] / max(1.0, crop.shape[1]) for p in (left_wrist, right_wrist) if p and shoulder_y is not None and p[1] < shoulder_y - height * 0.05]
        knee_angles = [self.angle(left_hip, left_knee, left_ankle), self.angle(right_hip, right_knee, right_ankle)]
        knee_angles = [value for value in knee_angles if value is not None]
        torso_verticality = None
        if shoulders and hips:
            sx, sy = sum(p[0] for p in shoulders)/len(shoulders), sum(p[1] for p in shoulders)/len(shoulders)
            hx, hy = sum(p[0] for p in hips)/len(hips), sum(p[1] for p in hips)/len(hips)
            torso_verticality = abs(hy-sy) / max(1e-6, math.hypot(hx-sx, hy-sy))
        aspect = width / height
        lying = aspect > 1.15 or (torso_verticality is not None and torso_verticality < 0.45)
        sitting = (not lying and torso_verticality is not None and torso_verticality > 0.45 and
                   ((knee_angles and sum(knee_angles)/len(knee_angles) < 150.0) or aspect > 0.55))
        return {"wrist_x": wrist_x, "aspect": aspect, "lying": lying, "sitting": sitting,
                "center_y": (y1+y2)/(2.0*max(1.0, crop.shape[0])), "height": height/max(1.0, crop.shape[0])}

    def classify(self, features):
        usable = [item for item in features if item]
        if len(usable) < self.min_samples:
            return "unknown", 0.0, "too few valid pose samples"
        first, last = usable[:max(2, len(usable)//3)], usable[-max(2, len(usable)//3):]
        first_lie = sum(item["lying"] for item in first) / float(len(first))
        last_lie = sum(item["lying"] for item in last) / float(len(last))
        center_drop = sum(item["center_y"] for item in last)/len(last) - sum(item["center_y"] for item in first)/len(first)
        aspect_gain = sum(item["aspect"] for item in last)/len(last) - sum(item["aspect"] for item in first)/len(first)
        if first_lie < .25 and last_lie >= .60 and (center_drop > .06 or aspect_gain > .25):
            return "falling", .88, "upright to horizontal transition"
        wave_x = [value for item in usable for value in item["wrist_x"]]
        if len(wave_x) >= 4 and max(wave_x)-min(wave_x) > .11:
            return "waving", .85, "raised wrist motion"
        lie_ratio = sum(item["lying"] for item in usable) / float(len(usable))
        if lie_ratio >= .65:
            # Same conservative visual fallback as upstream before point-cloud surface confirmation.
            return "lying_ground", .78, "static horizontal posture"
        sit_ratio = sum(item["sitting"] for item in usable) / float(len(usable))
        if sit_ratio >= .65:
            return "sitting", .78, "seated pose"
        return "unknown", .25, "no reliable target action"

    def publish(self, result):
        result.setdefault("recognizer", "yolo_pose")
        result["total_sec"] = round(time.time() - self.started, 3)
        self.pub.publish(String(data=json.dumps(result, ensure_ascii=False)))
        rospy.loginfo("YOLO-Pose action result: %s", result)

    def run_once(self):
        self.started = time.time()
        if self.model is None:
            self.publish({"action": "unknown", "place": "unknown", "confidence": 0.0, "error": self.init_error})
            return
        deadline = self.started + 20.0
        while not rospy.is_shutdown() and self.frame() is None and time.time() < deadline:
            rospy.sleep(.05)
        if self.frame() is None:
            self.publish({"action": "unknown", "place": "unknown", "confidence": 0.0, "error": "camera timeout"})
            return
        features = []
        interval = self.capture_seconds / float(self.sample_count)
        for _ in range(self.sample_count):
            frame = self.frame()
            if frame is not None:
                try:
                    features.append(self.feature(frame))
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "YOLO-Pose inference failed: %s", exc)
            rospy.sleep(interval)
        action, confidence, reason = self.classify(features)
        place = "floor" if action in {"lying_ground", "falling"} else "unknown"
        self.publish({"action": action, "place": place, "confidence": confidence, "pose_reason": reason,
                      "speech": {"waving": "主人正在挥手示意。", "sitting": "主人当前坐着。",
                                 "lying_ground": "识别到主人躺在地面。", "falling": "识别到主人摔倒。"}.get(action, "动作不确定。")})


if __name__ == "__main__":
    rospy.init_node("owner_action_yolo_pose")
    YoloPoseActionRecognition()
    rospy.spin()
