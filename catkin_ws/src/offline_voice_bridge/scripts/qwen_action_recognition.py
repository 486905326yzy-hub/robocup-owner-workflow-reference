#!/usr/bin/env python3
"""One-shot, local Qwen vision action recognition for the owner-search task."""
import base64
import json
import math
import threading
import time
import urllib.error
import urllib.request

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class QwenActionRecognition:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.result_topic = rospy.get_param("~result_topic", "/owner_voice_reid_test/action_result")
        self.llm_url = rospy.get_param("~llm_url", "http://127.0.0.1:11434/api/chat")
        self.llm_model = rospy.get_param("~llm_model", "qwen3.5:0.8b")
        self.llm_timeout = max(10.0, float(rospy.get_param("~llm_timeout", 90.0)))
        self.llm_keep_alive = rospy.get_param("~llm_keep_alive", "30m")
        self.llm_max_tokens = max(24, int(rospy.get_param("~llm_max_tokens", 48)))
        self.capture_delay = max(0.0, float(rospy.get_param("~capture_delay", 0.3)))
        self.capture_duration = max(0.5, float(rospy.get_param("~capture_duration", 5.0)))
        self.capture_frame_count = max(3, int(rospy.get_param("~capture_frame_count", 9)))
        self.llm_frame_count = max(1, int(rospy.get_param("~llm_frame_count", 5)))
        self.jpeg_quality = min(95, max(30, int(rospy.get_param("~jpeg_quality", 65))))
        self.max_width = max(160, int(rospy.get_param("~image_max_width", 384)))
        self.owner_roi = self._parse_roi(rospy.get_param("~owner_roi", ""))
        self.roi_padding = max(0.0, float(rospy.get_param("~roi_padding", 0.55)))
        self.min_confidence = min(0.95, max(0.0, float(rospy.get_param("~min_confidence", 0.75))))
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.pub = rospy.Publisher(self.result_topic, String, queue_size=1, latch=True)
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 24)
        threading.Thread(target=self.run_once, daemon=True).start()

    @staticmethod
    def _parse_roi(value):
        try:
            values = [int(float(v.strip())) for v in str(value).split(",")]
            return values if len(values) == 4 else None
        except (TypeError, ValueError):
            return None

    def image_callback(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            with self.lock:
                self.latest_frame = frame.copy()
        except Exception as exc:
            rospy.logwarn_throttle(3.0, "Action image conversion failed: %s", exc)

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def crop_owner(self, frame):
        if not self.owner_roi:
            return frame
        x1, y1, x2, y2 = self.owner_roi
        h, w = frame.shape[:2]
        pad_x, pad_y = int((x2-x1)*self.roi_padding), int((y2-y1)*self.roi_padding)
        x1, x2 = max(0, x1-pad_x), min(w, x2+pad_x)
        y1, y2 = max(0, y1-pad_y), min(h, y2+pad_y)
        return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame

    def encode(self, frame):
        frame = self.crop_owner(frame)
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, max(1, int(h*self.max_width/w))))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def call_qwen(self, images):
        prompt = ("/no_think\nClassify exactly one visible action of the registered person across consecutive images. "
                  "Return ONLY JSON: {\"action\":\"waving|sitting|lying|lying_ground|falling|unknown\","
                  "\"place\":\"floor|chair|sofa|bed|unknown\",\"confidence\":0.0,"
                  "\"speech\":\"short Chinese description\"}. "
                  "waving: raised hand moving side-to-side. sitting: seated. lying: lying on furniture. "
                  "lying_ground: already lying still on floor. falling: clear upright-to-floor transition inside this clip. "
                  "For standing, walking, ambiguous, or any other action, output unknown. Do not identify the person.")
        payload = {"model": self.llm_model, "stream": False, "think": False,
                   "keep_alive": self.llm_keep_alive,
                   "messages": [{"role": "user", "content": prompt, "images": images}],
                   "format": "json", "options": {"temperature": 0, "num_predict": self.llm_max_tokens, "num_ctx": 8192, "num_gpu": 999}}
        request = urllib.request.Request(self.llm_url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.llm_timeout) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("Qwen request failed: %s" % exc)
        content = json.loads(raw).get("message", {}).get("content", "").strip()
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < start:
            raise RuntimeError("Qwen returned non-JSON: %s" % content[:300])
        return json.loads(content[start:end+1])

    def publish(self, result):
        result.setdefault("recognizer", "qwen_vision")
        result.setdefault("action", "unknown")
        result.setdefault("place", "unknown")
        result.setdefault("confidence", 0.0)
        result["total_sec"] = round(time.time() - self.started, 3)
        self.pub.publish(String(data=json.dumps(result, ensure_ascii=False)))
        rospy.loginfo("Qwen action result: %s", result)

    def run_once(self):
        self.started = time.time()
        deadline = self.started + 20.0
        while not rospy.is_shutdown() and self.get_frame() is None and time.time() < deadline:
            rospy.sleep(0.05)
        if self.get_frame() is None:
            self.publish({"action": "unknown", "place": "unknown", "error": "camera timeout"})
            return
        rospy.sleep(self.capture_delay)
        frames, interval = [], self.capture_duration / float(self.capture_frame_count)
        for _ in range(self.capture_frame_count):
            frame = self.get_frame()
            if frame is not None:
                frames.append(frame)
            rospy.sleep(interval)
        if not frames:
            self.publish({"action": "unknown", "place": "unknown", "error": "no frames"})
            return
        indices = [int(round(i*(len(frames)-1)/float(max(1, self.llm_frame_count-1)))) for i in range(min(self.llm_frame_count, len(frames)))]
        try:
            result = self.call_qwen([self.encode(frames[i]) for i in indices])
            action = str(result.get("action", "unknown")).strip().lower()
            confidence = float(result.get("confidence", 0.0) or 0.0)
            if action not in {"waving", "sitting", "standing", "lying", "fallen"} or confidence < self.min_confidence:
                result["action"] = "unknown"
            result["confidence"] = confidence
            self.publish(result)
        except Exception as exc:
            rospy.logwarn("Qwen action recognition failed: %s", exc)
            self.publish({"action": "unknown", "place": "unknown", "error": str(exc)})


if __name__ == "__main__":
    rospy.init_node("owner_action_qwen")
    QwenActionRecognition()
    rospy.spin()
