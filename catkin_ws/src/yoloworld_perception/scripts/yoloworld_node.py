#!/usr/bin/env python3.9
import os
import sys
import time
import warnings

# catkin_install_python generates a ROS Noetic wrapper using the system
# Python 3.8 interpreter, even when this source file has a Python 3.9
# shebang. Re-exec the source under the dedicated CUDA 12.8 runtime before
# importing any Python packages.
if sys.version_info < (3, 9):
    os.execvp("python3.9", ["python3.9", os.path.abspath(__file__)] + sys.argv[1:])

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated.*",
    category=FutureWarning,
    module="huggingface_hub.file_download",
)

import cv2
import rospy
import torch
from perception_msgs.msg import Detection2D, Detection2DArray
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


def parse_classes(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return ["person"]


def resolve_model_path(model_path):
    expanded = os.path.expanduser(os.path.expandvars(model_path))
    if os.path.isdir(expanded):
        raise ValueError(f"model_path is a directory, not a .pt file: {expanded}")
    if os.path.exists(expanded):
        return expanded
    if os.path.sep in expanded:
        raise ValueError(f"model_path does not exist: {expanded}")
    return model_path


class YoloWorldNode:
    def __init__(self):
        self.frame_count = 0
        self.last_log_time = 0.0
        self.paused = False

        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.model_path = resolve_model_path(rospy.get_param("~model_path", "yolov8s-world.pt"))
        self.classes = parse_classes(rospy.get_param("~classes", ["person"]))
        self.device = rospy.get_param("~device", "cuda:0")
        self.require_gpu = rospy.get_param("~require_gpu", True)
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.conf = float(rospy.get_param("~conf", 0.25))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.max_det = int(rospy.get_param("~max_det", 50))
        self.process_every_n = max(1, int(rospy.get_param("~process_every_n", 1)))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))
        self.log_interval = float(rospy.get_param("~log_interval", 5.0))

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            message = "CUDA requested but torch.cuda.is_available() is false"
            if self.require_gpu:
                raise RuntimeError(message)
            rospy.logwarn("%s; falling back to CPU", message)
            self.device = "cpu"

        from ultralytics import YOLOWorld

        rospy.loginfo("Loading YOLO-World model: %s", self.model_path)
        self.model = YOLOWorld(self.model_path)
        self.model.set_classes(self.classes)
        rospy.loginfo("YOLO-World classes: %s", ", ".join(self.classes))
        rospy.loginfo("YOLO-World device: %s", self.device)

        self.det_pub = rospy.Publisher(
            "~detections", Detection2DArray, queue_size=5
        )
        self.debug_pub = rospy.Publisher(
            "~debug_image", Image, queue_size=2
        ) if self.publish_debug else None

        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.pause_sub = rospy.Subscriber("~pause", Bool, self.pause_callback, queue_size=1)
        rospy.loginfo("Subscribed image topic: %s", self.image_topic)

    def pause_callback(self, msg):
        self.paused = bool(msg.data)
        if self.paused:
            rospy.loginfo("YOLO-World inference paused")
        else:
            rospy.loginfo("YOLO-World inference resumed")

    def image_callback(self, msg):
        if self.paused:
            return

        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        try:
            image = self.image_to_bgr(msg)
        except Exception as exc:
            rospy.logerr_throttle(2.0, "ROS image conversion failed: %s", exc)
            return

        start = time.time()
        try:
            with torch.inference_mode():
                results = self.model.predict(
                    source=image,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    max_det=self.max_det,
                    verbose=False,
                )
        except Exception as exc:
            rospy.logerr_throttle(2.0, "YOLO-World inference failed: %s", exc)
            return

        result = results[0]
        det_array = Detection2DArray()
        det_array.header = msg.header

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

            for box, score, class_id in zip(xyxy, confs, class_ids):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1 = max(0, min(x1, image.shape[1] - 1))
                x2 = max(0, min(x2, image.shape[1] - 1))
                y1 = max(0, min(y1, image.shape[0] - 1))
                y2 = max(0, min(y2, image.shape[0] - 1))

                det = Detection2D()
                det.header = msg.header
                det.class_id = int(class_id)
                det.class_name = str(names.get(int(class_id), class_id)) if isinstance(names, dict) else str(class_id)
                det.score = float(score)
                det.xmin = min(x1, x2)
                det.ymin = min(y1, y2)
                det.xmax = max(x1, x2)
                det.ymax = max(y1, y2)
                det.center_x = int((det.xmin + det.xmax) / 2)
                det.center_y = int((det.ymin + det.ymax) / 2)
                det_array.detections.append(det)

        self.det_pub.publish(det_array)

        if self.debug_pub is not None:
            debug = self.draw_debug(image.copy(), det_array.detections)
            self.debug_pub.publish(self.bgr_to_image(debug, msg))

        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            elapsed_ms = (now - start) * 1000.0
            rospy.loginfo(
                "YOLO-World detections=%d inference=%.1fms device=%s",
                len(det_array.detections),
                elapsed_ms,
                self.device,
            )
            self.last_log_time = now

    @staticmethod
    def draw_debug(image, detections):
        for det in detections:
            color = (0, 255, 0)
            cv2.rectangle(image, (det.xmin, det.ymin), (det.xmax, det.ymax), color, 2)
            label = f"{det.class_name} {det.score:.2f}"
            y = max(15, det.ymin - 5)
            cv2.putText(image, label, (det.xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return image

    @staticmethod
    def image_to_bgr(msg):
        """Convert common ROS Image encodings without cv_bridge.

        This keeps the GPU YOLO process independent of ROS Noetic's Python
        3.8-only cv_bridge binary, allowing it to use a modern PyTorch wheel.
        """
        encoding = msg.encoding.lower()
        channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(encoding)
        if channels is None:
            raise ValueError("unsupported image encoding: %s" % msg.encoding)
        row_bytes = msg.width * channels
        if msg.step < row_bytes:
            raise ValueError("invalid step %d for width %d" % (msg.step, msg.width))
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        needed = msg.height * msg.step
        if raw.size < needed:
            raise ValueError("image data is truncated")
        image = raw[:needed].reshape(msg.height, msg.step)[:, :row_bytes]
        if channels == 1:
            return cv2.cvtColor(image.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
        image = image.reshape(msg.height, msg.width, channels)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else image.copy()

    @staticmethod
    def bgr_to_image(image, source_msg):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("debug image must be uint8 BGR")
        output = Image()
        output.header = source_msg.header
        output.height, output.width = image.shape[:2]
        output.encoding = "bgr8"
        output.is_bigendian = 0
        output.step = output.width * 3
        output.data = image.tobytes()
        return output


def main():
    rospy.init_node("yoloworld_node")
    try:
        YoloWorldNode()
    except Exception as exc:
        rospy.logfatal("Failed to start yoloworld_node: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
