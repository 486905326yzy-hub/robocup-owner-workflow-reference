#!/usr/bin/env python3
import base64
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import actionlib
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from perception_msgs.msg import Detection2DArray, Person3DArray
from sensor_msgs.msg import Image, JointState, LaserScan
from std_msgs.msg import Bool, String

try:
    from gazebo_msgs.msg import ModelStates
except Exception:
    ModelStates = None

try:
    import tf2_geometry_msgs  # noqa: F401
    import tf2_ros
except Exception:
    tf2_ros = None


def clamp(value, low, high):
    return max(low, min(high, value))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def signed_angle_diff(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


def set_pose_yaw(pose, yaw):
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = math.sin(yaw * 0.5)
    pose.orientation.w = math.cos(yaw * 0.5)


class Task1FindOwner:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_detections = []
        self.latest_scan = None
        self.latest_yaw = None
        self.latest_model_states = None
        self.latest_persons_3d = None
        self.owner_track_center = None
        self.tf_buffer = None
        self.tf_listener = None

        self.waypoint_name = rospy.get_param("~waypoint_name", "living_room")
        self.waypoint_file = os.path.expanduser(rospy.get_param("~waypoint_file", "~/waypoints.xml"))
        self.owner_image_path = os.path.expanduser(rospy.get_param("~owner_image_path", ""))

        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.detections_topic = rospy.get_param("~detections_topic", "/perception/person_detections_2d")
        self.persons_3d_topic = rospy.get_param("~persons_3d_topic", "/perception/person_detections_3d")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.say_topic = rospy.get_param("~say_topic", "/voice/say")
        self.pause_yolo_topic = rospy.get_param("~pause_yolo_topic", "/yoloworld/pause")

        self.llm_url = rospy.get_param("~llm_url", "http://127.0.0.1:11434/api/chat")
        self.llm_unload_url = rospy.get_param("~llm_unload_url", self.llm_url.replace("/api/chat", "/api/generate"))
        self.llm_model = rospy.get_param("~llm_model", "qwen3.5:9b")
        self.llm_timeout = float(rospy.get_param("~llm_timeout", 90.0))
        self.llm_keep_alive = rospy.get_param("~llm_keep_alive", "2m")
        self.llm_max_tokens = int(rospy.get_param("~llm_max_tokens", 80))
        self.llm_num_gpu = int(rospy.get_param("~llm_num_gpu", -1))
        self.llm_num_ctx = int(rospy.get_param("~llm_num_ctx", 1024))
        self.llm_top_p = float(rospy.get_param("~llm_top_p", 0.7))
        self.llm_pause_yolo = bool(rospy.get_param("~llm_pause_yolo", self.llm_num_gpu != 0))
        self.llm_prewarm = bool(rospy.get_param("~llm_prewarm", False))
        self.llm_unload_after_request = bool(rospy.get_param("~llm_unload_after_request", False))
        self.llm_unload_wait = float(rospy.get_param("~llm_unload_wait", 0.0))
        self.llm_image_max_side = int(rospy.get_param("~llm_image_max_side", 384))
        self.llm_jpeg_quality = int(rospy.get_param("~llm_jpeg_quality", 72))
        self.owner_accept_threshold = float(rospy.get_param("~owner_accept_threshold", 0.68))

        self.face_verify_enabled = bool(rospy.get_param("~face_verify_enabled", True))
        self.face_auto_download = bool(rospy.get_param("~face_auto_download", False))
        self.face_model_name = rospy.get_param("~face_model_name", "buffalo_sc")
        self.face_model_root = os.path.expanduser(rospy.get_param("~face_model_root", "~/.insightface"))
        self.face_ctx_id = int(rospy.get_param("~face_ctx_id", -1))
        self.face_det_size = int(rospy.get_param("~face_det_size", 320))
        self.face_det_thresh = float(rospy.get_param("~face_det_thresh", 0.45))
        self.face_accept_threshold = float(rospy.get_param("~face_accept_threshold", 0.45))
        self.face_reject_threshold = float(rospy.get_param("~face_reject_threshold", 0.25))
        self.face_fast_reject = bool(rospy.get_param("~face_fast_reject", True))
        self.face_qwen_fallback = bool(rospy.get_param("~face_qwen_fallback", False))
        self.face_crop_top_ratio = float(rospy.get_param("~face_crop_top_ratio", 0.68))

        self.detection_min_score = float(rospy.get_param("~detection_min_score", 0.25))
        self.detection_min_area_ratio = float(rospy.get_param("~detection_min_area_ratio", 0.015))
        self.candidate_cooldown = float(rospy.get_param("~candidate_cooldown", 5.0))
        self.verify_top_k = int(rospy.get_param("~verify_top_k", 2))
        self.verify_during_scan = bool(rospy.get_param("~verify_during_scan", False))
        self.verify_after_scan_top_k = int(rospy.get_param("~verify_after_scan_top_k", 2))
        self.candidate_collection_interval = float(rospy.get_param("~candidate_collection_interval", 0.8))
        self.scan_candidate_pool_size = int(rospy.get_param("~scan_candidate_pool_size", 6))

        self.action_recognition_enabled = bool(rospy.get_param("~action_recognition_enabled", True))
        self.action_model_path = os.path.expanduser(rospy.get_param("~action_model_path", ""))
        self.action_device = rospy.get_param("~action_device", "cuda:0")
        self.action_require_gpu = bool(rospy.get_param("~action_require_gpu", True))
        self.action_imgsz = int(rospy.get_param("~action_imgsz", 384))
        self.action_conf = float(rospy.get_param("~action_conf", 0.25))
        self.action_iou = float(rospy.get_param("~action_iou", 0.45))
        self.action_half = bool(rospy.get_param("~action_half", True))
        self.action_sample_seconds = float(rospy.get_param("~action_sample_seconds", 2.5))
        self.action_sample_rate = float(rospy.get_param("~action_sample_rate", 5.0))
        self.action_min_keypoint_conf = float(rospy.get_param("~action_min_keypoint_conf", 0.25))
        self.action_max_det = int(rospy.get_param("~action_max_det", 4))
        self.action_pause_yolo = bool(rospy.get_param("~action_pause_yolo", False))
        self.action_use_owner_roi = bool(rospy.get_param("~action_use_owner_roi", True))
        self.action_roi_padding = float(rospy.get_param("~action_roi_padding", 0.35))
        self.action_min_pose_samples = int(rospy.get_param("~action_min_pose_samples", 5))
        self.action_static_required_ratio = float(rospy.get_param("~action_static_required_ratio", 0.65))
        self.action_report_standing = bool(rospy.get_param("~action_report_standing", False))
        self.action_sim_hint_enabled = bool(rospy.get_param("~action_sim_hint_enabled", False))
        self.action_sim_hint_priority = bool(rospy.get_param("~action_sim_hint_priority", True))
        self.action_sim_hint_timeout = float(rospy.get_param("~action_sim_hint_timeout", 0.8))
        self.action_sim_hint_topic = rospy.get_param("~action_sim_hint_topic", "/gazebo/model_states")
        self.action_sim_robot_names = [str(name).lower() for name in rospy.get_param("~action_sim_robot_names", ["wpb_home_mani", "wpb_home", "wpr1"])]
        self.action_sim_human_keywords = [str(name).lower() for name in rospy.get_param("~action_sim_human_keywords", ["person", "kai", "man", "owner", "guest"])]
        self.action_sim_front_bearing_limit = float(rospy.get_param("~action_sim_front_bearing_limit", 0.75))
        self.action_sim_max_distance = float(rospy.get_param("~action_sim_max_distance", 8.0))
        self.center_owner_before_action = bool(rospy.get_param("~center_owner_before_action", True))
        self.center_owner_timeout = float(rospy.get_param("~center_owner_timeout", 5.0))
        self.center_owner_tolerance = float(rospy.get_param("~center_owner_tolerance", 0.06))
        self.center_owner_angular_gain = float(rospy.get_param("~center_owner_angular_gain", 0.75))
        self.center_owner_max_angular_speed = abs(float(rospy.get_param("~center_owner_max_angular_speed", 0.45)))
        self.center_owner_lost_turn_speed = abs(float(rospy.get_param("~center_owner_lost_turn_speed", 0.12)))

        self.navigate_timeout = float(rospy.get_param("~navigate_timeout", 120.0))
        self.scan_angular_speed = float(rospy.get_param("~scan_angular_speed", 0.35))
        self.scan_duration = float(rospy.get_param("~scan_duration", 19.0))
        self.scan_total_angle = float(rospy.get_param("~scan_total_angle", 6.45))
        self.scan_timeout = float(rospy.get_param("~scan_timeout", 120.0))
        self.scan_after_arrival_delay = float(rospy.get_param("~scan_after_arrival_delay", 1.0))
        self.scan_return_to_owner = bool(rospy.get_param("~scan_return_to_owner", True))
        self.return_angular_speed = abs(float(rospy.get_param("~return_angular_speed", 0.35)))

        self.approach_enabled = bool(rospy.get_param("~approach_enabled", True))
        self.approach_timeout = float(rospy.get_param("~approach_timeout", 18.0))
        self.approach_linear_speed = float(rospy.get_param("~approach_linear_speed", 0.16))
        self.approach_angular_gain = float(rospy.get_param("~approach_angular_gain", 0.7))
        self.approach_stop_distance = float(rospy.get_param("~approach_stop_distance", 0.9))
        self.approach_center_tolerance = float(rospy.get_param("~approach_center_tolerance", 0.10))
        self.approach_target_height_ratio = float(rospy.get_param("~approach_target_height_ratio", 0.52))
        self.approach_use_3d = bool(rospy.get_param("~approach_use_3d", True))
        self.approach_3d_wait_timeout = float(rospy.get_param("~approach_3d_wait_timeout", 3.0))
        self.approach_3d_max_age = float(rospy.get_param("~approach_3d_max_age", 1.0))
        self.approach_goal_frame = rospy.get_param("~approach_goal_frame", "map")
        self.approach_base_frame = rospy.get_param("~approach_base_frame", "base_footprint")
        self.approach_3d_coordinate_mode = str(rospy.get_param("~approach_3d_coordinate_mode", "auto")).lower()
        self.approach_optical_lateral_sign = float(rospy.get_param("~approach_optical_lateral_sign", -1.0))
        self.approach_standoff_distance = float(rospy.get_param("~approach_standoff_distance", 0.55))
        self.approach_side_offset = float(rospy.get_param("~approach_side_offset", 0.25))
        self.approach_side_preference = str(rospy.get_param("~approach_side_preference", "nearest"))
        self.approach_min_person_distance = float(rospy.get_param("~approach_min_person_distance", 0.35))
        self.approach_max_person_distance = float(rospy.get_param("~approach_max_person_distance", 6.0))
        self.approach_min_goal_distance = float(rospy.get_param("~approach_min_goal_distance", 0.20))
        self.approach_goal_timeout = float(rospy.get_param("~approach_goal_timeout", self.approach_timeout))
        self.approach_fallback_2d = bool(rospy.get_param("~approach_fallback_2d", False))
        self.approach_pause_yolo = bool(rospy.get_param("~approach_pause_yolo", True))
        self.approach_plan_check = bool(rospy.get_param("~approach_plan_check", True))
        self.approach_plan_service = rospy.get_param("~approach_plan_service", "/move_base/make_plan")
        self.approach_plan_tolerance = float(rospy.get_param("~approach_plan_tolerance", 0.25))
        self.approach_max_goal_distance = float(rospy.get_param("~approach_max_goal_distance", 1.25))
        self.approach_3d_max_stages = max(1, int(rospy.get_param("~approach_3d_max_stages", 3)))
        self.approach_stage_timeout = float(rospy.get_param("~approach_stage_timeout", 12.0))
        self.approach_nav_start_distance = float(rospy.get_param("~approach_nav_start_distance", 1.8))
        self.approach_servo_fallback = bool(rospy.get_param("~approach_servo_fallback", True))
        self.approach_servo_stage_distance = float(rospy.get_param("~approach_servo_stage_distance", 1.6))
        self.approach_servo_stage_timeout = float(rospy.get_param("~approach_servo_stage_timeout", 16.0))
        self.approach_servo_timeout = float(rospy.get_param("~approach_servo_timeout", 10.0))
        self.approach_servo_stop_distance = float(rospy.get_param("~approach_servo_stop_distance", self.approach_standoff_distance + 0.12))
        self.approach_servo_linear_speed = float(rospy.get_param("~approach_servo_linear_speed", 0.16))
        self.approach_servo_angular_gain = float(rospy.get_param("~approach_servo_angular_gain", 0.75))
        self.approach_final_align_enabled = bool(rospy.get_param("~approach_final_align_enabled", True))
        self.approach_final_align_timeout = float(rospy.get_param("~approach_final_align_timeout", 5.0))
        self.approach_final_align_tolerance = float(rospy.get_param("~approach_final_align_tolerance", 0.06))
        self.approach_final_align_angular_gain = float(rospy.get_param("~approach_final_align_angular_gain", 0.9))
        self.approach_final_align_max_speed = abs(float(rospy.get_param("~approach_final_align_max_speed", 0.28)))
        self.approach_fine_refine_enabled = bool(rospy.get_param("~approach_fine_refine_enabled", True))
        self.approach_fine_refine_timeout = float(rospy.get_param("~approach_fine_refine_timeout", 8.0))
        self.approach_fine_distance_tolerance = float(rospy.get_param("~approach_fine_distance_tolerance", 0.06))
        self.approach_fine_bearing_tolerance = float(rospy.get_param("~approach_fine_bearing_tolerance", 0.05))
        self.approach_fine_stable_cycles = max(1, int(rospy.get_param("~approach_fine_stable_cycles", 4)))
        self.approach_fine_linear_gain = float(rospy.get_param("~approach_fine_linear_gain", 0.25))
        self.approach_fine_angular_gain = float(rospy.get_param("~approach_fine_angular_gain", 0.9))
        self.approach_fine_max_linear_speed = abs(float(rospy.get_param("~approach_fine_max_linear_speed", 0.08)))
        self.approach_fine_max_reverse_speed = abs(float(rospy.get_param("~approach_fine_max_reverse_speed", 0.05)))
        self.approach_fine_max_angular_speed = abs(float(rospy.get_param("~approach_fine_max_angular_speed", 0.22)))
        self.approach_fine_lidar_stop_distance = float(rospy.get_param("~approach_fine_lidar_stop_distance", 0.55))
        self.approach_fine_lidar_margin = float(rospy.get_param("~approach_fine_lidar_margin", 0.08))

        self.arm_assist_enabled = bool(rospy.get_param("~arm_assist_enabled", False))
        self.arm_assist_topic = rospy.get_param("~arm_assist_topic", "/wpb_home/mani_ctrl")
        self.arm_assist_lift_height = float(rospy.get_param("~arm_assist_lift_height", 1.0))
        self.arm_assist_gripper = float(rospy.get_param("~arm_assist_gripper", 0.16))
        self.arm_assist_publish_seconds = float(rospy.get_param("~arm_assist_publish_seconds", 1.5))
        self.arm_assist_hold_seconds = float(rospy.get_param("~arm_assist_hold_seconds", 3.0))

        self.speak_on_start = bool(rospy.get_param("~speak_on_start", True))
        self.speak_on_arrival = bool(rospy.get_param("~speak_on_arrival", True))
        self.speak_on_owner_found = bool(rospy.get_param("~speak_on_owner_found", True))
        self.say_wait_for_subscribers = bool(rospy.get_param("~say_wait_for_subscribers", True))
        self.say_wait_timeout = float(rospy.get_param("~say_wait_timeout", 30.0))
        self.say_repeat_count = max(1, int(rospy.get_param("~say_repeat_count", 1)))
        self.say_repeat_interval = float(rospy.get_param("~say_repeat_interval", 0.25))
        self.say_after_publish_delay = float(rospy.get_param("~say_after_publish_delay", 1.0))
        self.action_speech_hold = float(rospy.get_param("~action_speech_hold", 3.2))

        self.owner_image = self.load_owner_image()
        self.owner_image_b64 = self.encode_jpeg_b64(self.owner_image)
        self.face_app = None
        self.owner_face_embedding = None
        self.face_ready = False
        self.action_pose_model = None
        self.action_ready = False

        if tf2_ros is not None:
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        else:
            rospy.logwarn("tf2_ros is unavailable; 3D owner approach cannot transform goals to %s", self.approach_goal_frame)

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.say_pub = rospy.Publisher(self.say_topic, String, queue_size=5)
        self.arm_pub = rospy.Publisher(self.arm_assist_topic, JointState, queue_size=10)
        self.pause_yolo_pub = rospy.Publisher(self.pause_yolo_topic, Bool, queue_size=1, latch=True)

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.det_sub = rospy.Subscriber(self.detections_topic, Detection2DArray, self.detections_callback, queue_size=1)
        self.person3d_sub = rospy.Subscriber(self.persons_3d_topic, Person3DArray, self.persons_3d_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        if self.action_sim_hint_enabled and ModelStates is not None:
            self.model_states_sub = rospy.Subscriber(self.action_sim_hint_topic, ModelStates, self.model_states_callback, queue_size=1)
        else:
            self.model_states_sub = None

        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.make_plan = rospy.ServiceProxy(self.approach_plan_service, GetPlan)
        self.init_face_recognizer()
        self.init_action_recognizer()
        if self.llm_prewarm and self.face_qwen_fallback:
            threading.Thread(target=self.prewarm_qwen_model, daemon=True).start()

    def load_owner_image(self):
        if not self.owner_image_path:
            raise RuntimeError("owner_image_path is empty")
        if not os.path.exists(self.owner_image_path):
            raise RuntimeError("owner photo not found: %s" % self.owner_image_path)
        image = cv2.imread(self.owner_image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to read owner photo as image: %s" % self.owner_image_path)
        return image

    @staticmethod
    def normalize_embedding(embedding):
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return None
        return vector / norm

    @staticmethod
    def select_largest_face(faces):
        if not faces:
            return None

        def face_priority(face):
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                return 0.0
            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            det_score = float(getattr(face, "det_score", 1.0) or 1.0)
            return width * height * det_score

        return max(faces, key=face_priority)

    def init_face_recognizer(self):
        if not self.face_verify_enabled:
            rospy.loginfo("Lightweight face verification disabled")
            return
        model_dir = os.path.join(self.face_model_root, "models", self.face_model_name)
        cached_models = []
        if os.path.isdir(model_dir):
            cached_models = [name for name in os.listdir(model_dir) if name.endswith(".onnx")]
        if not cached_models and not self.face_auto_download:
            fallback_text = "falling back to Qwen" if self.face_qwen_fallback else "continuing without Qwen fallback"
            rospy.logwarn(
                "InsightFace model %s is not cached under %s; %s. "
                "Run tools/install_face_recognition_deps.sh or set face_auto_download=true to download it.",
                self.face_model_name,
                model_dir,
                fallback_text,
            )
            return
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            fallback_text = "falling back to Qwen-only owner verification" if self.face_qwen_fallback else "Qwen fallback disabled"
            rospy.logwarn(
                "InsightFace is not available; %s: %s",
                fallback_text,
                exc,
            )
            return

        providers = ["CPUExecutionProvider"] if self.face_ctx_id < 0 else None
        try:
            self.face_app = FaceAnalysis(
                name=self.face_model_name,
                root=self.face_model_root,
                providers=providers,
            )
            self.face_app.prepare(
                ctx_id=self.face_ctx_id,
                det_thresh=self.face_det_thresh,
                det_size=(self.face_det_size, self.face_det_size),
            )
            faces = self.face_app.get(self.owner_image)
            owner_face = self.select_largest_face(faces)
            if owner_face is None:
                fallback_text = "falling back to Qwen verification" if self.face_qwen_fallback else "Qwen fallback disabled"
                rospy.logwarn("No face found in owner reference photo; %s", fallback_text)
                return
            self.owner_face_embedding = self.normalize_embedding(owner_face.embedding)
            if self.owner_face_embedding is None:
                fallback_text = "falling back to Qwen verification" if self.face_qwen_fallback else "Qwen fallback disabled"
                rospy.logwarn("Owner face embedding is invalid; %s", fallback_text)
                return
            self.face_ready = True
            rospy.loginfo(
                "Lightweight face verification ready: model=%s ctx_id=%d threshold=%.2f",
                self.face_model_name,
                self.face_ctx_id,
                self.face_accept_threshold,
            )
        except Exception as exc:
            self.face_app = None
            self.owner_face_embedding = None
            self.face_ready = False
            fallback_text = "falling back to Qwen" if self.face_qwen_fallback else "Qwen fallback disabled"
            rospy.logwarn("Failed to initialize lightweight face verification; %s: %s", fallback_text, exc)

    def init_action_recognizer(self):
        if not self.action_recognition_enabled:
            rospy.loginfo("Owner action recognition disabled")
            return
        if not self.action_model_path:
            rospy.logwarn("Owner action recognition disabled: action_model_path is empty")
            return
        if not os.path.exists(self.action_model_path):
            rospy.logwarn(
                "Owner action pose model not found: %s. Download yolo11n-pose.pt or set action_model_path.",
                self.action_model_path,
            )
            return

        if self.action_require_gpu and str(self.action_device).startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    rospy.logwarn("Owner action recognition requires GPU, but PyTorch CUDA is unavailable")
                    return
            except Exception as exc:
                rospy.logwarn("Owner action recognition cannot check CUDA availability: %s", exc)
                return

        try:
            from ultralytics import YOLO
            self.action_pose_model = YOLO(self.action_model_path)
            self.action_ready = True
            rospy.loginfo(
                "Owner action recognition ready: model=%s device=%s imgsz=%d",
                self.action_model_path,
                self.action_device,
                self.action_imgsz,
            )
        except Exception as exc:
            self.action_pose_model = None
            self.action_ready = False
            rospy.logwarn("Failed to initialize owner action recognition: %s", exc)

    def snapshot_image(self):
        with self.lock:
            if self.latest_image is None:
                return None
            return self.latest_image.copy()

    def model_states_callback(self, msg):
        with self.lock:
            self.latest_model_states = msg

    def snapshot_owner_action_image(self):
        with self.lock:
            if self.latest_image is None:
                return None
            image = self.latest_image.copy()
            detections = list(self.latest_detections)

        if not self.action_use_owner_roi or not detections:
            return image

        height, width = image.shape[:2]
        best = None
        best_score = -1e9
        target_center = self.owner_track_center if self.owner_track_center is not None else 0.5
        for det in detections:
            if det.class_name and det.class_name != "person":
                continue
            if det.score < self.detection_min_score:
                continue
            xmin = clamp(int(det.xmin), 0, width - 1)
            ymin = clamp(int(det.ymin), 0, height - 1)
            xmax = clamp(int(det.xmax), 0, width - 1)
            ymax = clamp(int(det.ymax), 0, height - 1)
            if xmax <= xmin or ymax <= ymin:
                continue
            center_norm = float(det.center_x) / max(1.0, float(width))
            area_ratio = float((xmax - xmin) * (ymax - ymin)) / max(1.0, float(width * height))
            score = float(det.score) + area_ratio * 2.0 - abs(center_norm - target_center) * 2.0
            if score > best_score:
                best_score = score
                best = (xmin, ymin, xmax, ymax, center_norm)

        if best is None:
            return image

        xmin, ymin, xmax, ymax, center_norm = best
        box_w = xmax - xmin
        box_h = ymax - ymin
        pad_x = int(box_w * clamp(self.action_roi_padding, 0.0, 1.0))
        pad_y = int(box_h * clamp(self.action_roi_padding, 0.0, 1.0))
        cx1 = clamp(xmin - pad_x, 0, width - 1)
        cy1 = clamp(ymin - pad_y, 0, height - 1)
        cx2 = clamp(xmax + pad_x, 0, width - 1)
        cy2 = clamp(ymax + pad_y, 0, height - 1)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return image
        self.owner_track_center = center_norm
        return crop

    def action_from_name(self, name):
        text = str(name).lower()
        if "wave" in text or "waving" in text or "hand" in text:
            return "waving", 0.98, "simulation model name"
        if "fall" in text or "fallen" in text or "摔" in text:
            return "falling", 0.98, "simulation model name"
        if "lying" in text or "lie" in text or "lay" in text or "bed" in text or "躺" in text:
            return "lying", 0.98, "simulation model name"
        if "sitting" in text or "seated" in text or "sit" in text or "chair" in text or "坐" in text:
            return "sitting", 0.98, "simulation model name"
        return None

    def classify_action_from_sim_hint(self):
        if not self.action_sim_hint_enabled or ModelStates is None:
            return None
        with self.lock:
            states = self.latest_model_states
        if states is None:
            return None

        names = list(states.name)
        lower_names = [name.lower() for name in names]
        robot_index = None
        for index, name in enumerate(lower_names):
            if name in self.action_sim_robot_names or any(robot_name in name for robot_name in self.action_sim_robot_names):
                robot_index = index
                break
        if robot_index is None:
            return None

        robot_pose = states.pose[robot_index]
        robot_yaw = yaw_from_quaternion(robot_pose.orientation)
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        best = None
        best_score = -1e9
        for index, name in enumerate(lower_names):
            if index == robot_index:
                continue
            if not any(keyword in name for keyword in self.action_sim_human_keywords):
                continue
            verdict = self.action_from_name(name)
            if verdict is None:
                continue
            pose = states.pose[index]
            dx = pose.position.x - robot_pose.position.x
            dy = pose.position.y - robot_pose.position.y
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy
            if local_x <= 0:
                continue
            distance = math.hypot(dx, dy)
            if distance > self.action_sim_max_distance:
                continue
            bearing = math.atan2(local_y, local_x)
            if abs(bearing) > self.action_sim_front_bearing_limit:
                continue
            score = -abs(bearing) - distance * 0.03
            if score > best_score:
                best_score = score
                best = (name, verdict, distance, bearing)
        if best is None:
            return None
        name, verdict, distance, bearing = best
        label, confidence, reason = verdict
        rospy.loginfo(
            "Simulation action hint: model=%s label=%s distance=%.2f bearing=%.2f",
            name,
            label,
            distance,
            bearing,
        )
        return label, confidence, reason

    def wait_for_sim_action_hint(self):
        if not self.action_sim_hint_enabled:
            return None
        deadline = time.time() + max(0.0, self.action_sim_hint_timeout)
        while not rospy.is_shutdown() and time.time() <= deadline:
            verdict = self.classify_action_from_sim_hint()
            if verdict is not None:
                return verdict
            rospy.sleep(0.05)
        rospy.loginfo(
            "Simulation action hint did not produce a valid action within %.1fs",
            self.action_sim_hint_timeout,
        )
        return None

    def select_owner_pose(self, result, image_width):
        if result is None or result.keypoints is None or result.boxes is None:
            return None
        try:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
        except Exception as exc:
            rospy.logwarn("Failed to read pose result tensors: %s", exc)
            return None

        count = min(len(keypoints_xy), len(boxes_xyxy))
        if count <= 0:
            return None

        best = None
        best_score = -1e9
        target_center = self.owner_track_center
        for index in range(count):
            kxy = keypoints_xy[index]
            kconf = keypoints_conf[index]
            valid_count = int(np.sum(kconf >= self.action_min_keypoint_conf))
            if valid_count < 5:
                continue

            bbox = boxes_xyxy[index]
            center_norm = ((float(bbox[0]) + float(bbox[2])) * 0.5) / max(1.0, float(image_width))
            center_penalty = abs(center_norm - target_center) if target_center is not None else abs(center_norm - 0.5)
            score = float(boxes_conf[index]) + valid_count * 0.04 - center_penalty * 1.5
            if score > best_score:
                best_score = score
                best = {
                    "keypoints_xy": kxy,
                    "keypoints_conf": kconf,
                    "bbox": bbox,
                    "box_conf": float(boxes_conf[index]),
                    "center_norm": center_norm,
                    "valid_count": valid_count,
                }
        return best

    @staticmethod
    def angle_at(point_a, point_b, point_c):
        vec_a = np.asarray(point_a, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
        vec_c = np.asarray(point_c, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_c))
        if denom <= 1e-6:
            return None
        cosine = clamp(float(np.dot(vec_a, vec_c) / denom), -1.0, 1.0)
        return math.degrees(math.acos(cosine))

    @staticmethod
    def median_value(values):
        cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if not cleaned:
            return None
        return float(np.median(np.asarray(cleaned, dtype=np.float32)))

    def build_pose_feature(self, pose, image_shape):
        height, width = image_shape[:2]
        kxy = pose["keypoints_xy"]
        kconf = pose["keypoints_conf"]
        bbox = pose["bbox"]
        box_w = max(1.0, float(bbox[2]) - float(bbox[0]))
        box_h = max(1.0, float(bbox[3]) - float(bbox[1]))
        aspect = box_w / box_h
        center_y_norm = ((float(bbox[1]) + float(bbox[3])) * 0.5) / max(1.0, float(height))

        def point(index):
            if index >= len(kxy) or float(kconf[index]) < self.action_min_keypoint_conf:
                return None
            return (float(kxy[index][0]), float(kxy[index][1]))

        def midpoint(left, right):
            if left is None or right is None:
                return None
            return ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)

        left_shoulder = point(5)
        right_shoulder = point(6)
        left_wrist = point(9)
        right_wrist = point(10)
        left_hip = point(11)
        right_hip = point(12)
        left_knee = point(13)
        right_knee = point(14)
        left_ankle = point(15)
        right_ankle = point(16)
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        knee_mid = midpoint(left_knee, right_knee)

        torso_verticality = None
        if shoulder_mid is not None and hip_mid is not None:
            torso_dx = hip_mid[0] - shoulder_mid[0]
            torso_dy = hip_mid[1] - shoulder_mid[1]
            torso_dist = math.hypot(torso_dx, torso_dy)
            if torso_dist > 1e-6:
                torso_verticality = abs(torso_dy) / torso_dist

        knee_angles = []
        left_knee_angle = self.angle_at(left_hip, left_knee, left_ankle) if left_hip and left_knee and left_ankle else None
        right_knee_angle = self.angle_at(right_hip, right_knee, right_ankle) if right_hip and right_knee and right_ankle else None
        if left_knee_angle is not None:
            knee_angles.append(left_knee_angle)
        if right_knee_angle is not None:
            knee_angles.append(right_knee_angle)
        knee_angle = self.median_value(knee_angles)

        left_wrist_above = bool(left_wrist and left_shoulder and left_wrist[1] < left_shoulder[1] - box_h * 0.05)
        right_wrist_above = bool(right_wrist and right_shoulder and right_wrist[1] < right_shoulder[1] - box_h * 0.05)
        left_wrist_x_norm = left_wrist[0] / max(1.0, float(width)) if left_wrist else None
        right_wrist_x_norm = right_wrist[0] / max(1.0, float(width)) if right_wrist else None

        torso_v = torso_verticality if torso_verticality is not None else 1.0
        lying_like = (torso_v < 0.42) or (aspect > 1.20 and torso_v < 0.68)

        knees_near_hips = False
        if hip_mid is not None and knee_mid is not None:
            knees_near_hips = abs(knee_mid[1] - hip_mid[1]) < box_h * 0.45
        bent_knee = knee_angle is not None and knee_angle < 150.0
        sitting_like = torso_v > 0.45 and (bent_knee or knees_near_hips) and not lying_like
        upright_like = torso_v > 0.65 and aspect < 1.05

        return {
            "aspect": aspect,
            "center_y_norm": center_y_norm,
            "torso_verticality": torso_verticality,
            "knee_angle": knee_angle,
            "left_wrist_above": left_wrist_above,
            "right_wrist_above": right_wrist_above,
            "left_wrist_x_norm": left_wrist_x_norm,
            "right_wrist_x_norm": right_wrist_x_norm,
            "lying_like": lying_like,
            "sitting_like": sitting_like,
            "upright_like": upright_like,
            "box_conf": pose["box_conf"],
            "valid_count": pose["valid_count"],
        }

    def predict_owner_pose_feature(self, image):
        if image is None or image.size == 0 or not self.action_ready or self.action_pose_model is None:
            return None
        try:
            use_half = self.action_half and str(self.action_device).startswith("cuda")
            results = self.action_pose_model.predict(
                image,
                imgsz=self.action_imgsz,
                conf=self.action_conf,
                iou=self.action_iou,
                device=self.action_device,
                half=use_half,
                max_det=max(1, self.action_max_det),
                verbose=False,
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Owner action pose inference failed: %s", exc)
            return None
        if not results:
            return None
        pose = self.select_owner_pose(results[0], image.shape[1])
        if pose is None:
            return None
        return self.build_pose_feature(pose, image.shape)

    def classify_owner_action(self, features):
        usable = [feature for feature in features if feature is not None]
        if not usable:
            return "unknown", 0.0, "no pose"
        if len(usable) < max(1, self.action_min_pose_samples):
            return "unknown", 0.15, "too few pose samples"

        sample_count = float(len(usable))
        first = usable[:max(1, len(usable) // 3)]
        last = usable[-max(1, len(usable) // 3):]

        first_center = self.median_value([feature["center_y_norm"] for feature in first])
        last_center = self.median_value([feature["center_y_norm"] for feature in last])
        first_torso = self.median_value([feature["torso_verticality"] for feature in first])
        last_torso = self.median_value([feature["torso_verticality"] for feature in last])
        first_aspect = self.median_value([feature["aspect"] for feature in first])
        last_aspect = self.median_value([feature["aspect"] for feature in last])
        last_lie_ratio = sum(1 for feature in last if feature["lying_like"]) / float(len(last))
        median_aspect = self.median_value([feature["aspect"] for feature in usable])
        median_torso = self.median_value([feature["torso_verticality"] for feature in usable])
        median_knee = self.median_value([feature["knee_angle"] for feature in usable])

        if first_center is not None and last_center is not None:
            center_drop = last_center - first_center
            torso_drop = (first_torso - last_torso) if first_torso is not None and last_torso is not None else 0.0
            aspect_gain = (last_aspect - first_aspect) if first_aspect is not None and last_aspect is not None else 0.0
            if center_drop > 0.12 and (torso_drop > 0.28 or aspect_gain > 0.40 or last_lie_ratio >= 0.70):
                confidence = clamp(0.55 + center_drop + max(0.0, torso_drop) + max(0.0, aspect_gain) * 0.4, 0.0, 0.98)
                return "falling", confidence, "dynamic drop"

        wave_scores = []
        for side in ("left", "right"):
            above_key = "%s_wrist_above" % side
            x_key = "%s_wrist_x_norm" % side
            wrist_x = [feature[x_key] for feature in usable if feature[above_key] and feature[x_key] is not None]
            if len(wrist_x) >= 3:
                x_range = max(wrist_x) - min(wrist_x)
                above_ratio = len(wrist_x) / sample_count
                if x_range > 0.11 and above_ratio >= 0.45:
                    wave_scores.append(clamp(0.50 + x_range * 2.0 + above_ratio * 0.25, 0.0, 0.95))
        if wave_scores:
            return "waving", max(wave_scores), "raised wrist motion"

        lie_ratio = sum(1 for feature in usable if feature["lying_like"]) / sample_count
        sit_ratio = sum(1 for feature in usable if feature["sitting_like"]) / sample_count
        required_ratio = clamp(self.action_static_required_ratio, 0.50, 0.95)
        rospy.loginfo(
            "Owner action feature summary: samples=%d lie_ratio=%.2f sit_ratio=%.2f aspect=%s torso=%s knee=%s",
            len(usable),
            lie_ratio,
            sit_ratio,
            "%.2f" % median_aspect if median_aspect is not None else "NA",
            "%.2f" % median_torso if median_torso is not None else "NA",
            "%.0f" % median_knee if median_knee is not None else "NA",
        )
        if lie_ratio >= required_ratio and (median_aspect is None or median_aspect > 1.05):
            return "lying", clamp(0.50 + lie_ratio * 0.45, 0.0, 0.95), "horizontal body posture"
        if sit_ratio >= required_ratio and median_torso is not None and median_torso > 0.45:
            return "sitting", clamp(0.50 + sit_ratio * 0.45, 0.0, 0.95), "bent seated posture"

        upright_ratio = sum(1 for feature in usable if feature["upright_like"]) / sample_count
        if self.action_report_standing and upright_ratio >= required_ratio:
            return "standing", clamp(0.45 + upright_ratio * 0.35, 0.0, 0.85), "upright posture"
        return "unknown", 0.25, "no reliable target action"

    def recognize_owner_action(self, owner_candidate):
        if owner_candidate is not None and "det" in owner_candidate:
            det = owner_candidate["det"]
            image_width = float(owner_candidate.get("image_width", 0.0))
            if image_width > 0:
                self.owner_track_center = float(det.center_x) / image_width

        if not self.action_recognition_enabled:
            return "unknown", 0.0, "disabled"

        self.stop_base()

        # Gazebo exposes semantic action information through model names.
        # This path is enabled only by the simulation launch.
        if self.action_sim_hint_priority and self.action_sim_hint_enabled:
            sim_verdict = self.wait_for_sim_action_hint()
            if sim_verdict is not None:
                label, confidence, reason = sim_verdict
                rospy.loginfo(
                    "Owner action verdict: label=%s confidence=%.2f reason=%s source=simulation_hint",
                    label,
                    confidence,
                    reason,
                )
                return label, confidence, reason

        if not self.action_ready or self.action_pose_model is None:
            return "unknown", 0.0, "pose unavailable"

        if self.action_pause_yolo:
            self.set_yolo_paused(True)
            rospy.sleep(0.15)

        features = []
        deadline = time.time() + max(0.5, self.action_sample_seconds)
        interval = 1.0 / max(1.0, self.action_sample_rate)
        next_sample = 0.0
        rospy.loginfo("Recognizing owner action for %.1fs with lightweight pose model", self.action_sample_seconds)
        try:
            while not rospy.is_shutdown() and time.time() < deadline:
                now = time.time()
                if now >= next_sample:
                    image = self.snapshot_owner_action_image()
                    feature = self.predict_owner_pose_feature(image)
                    if feature is not None:
                        features.append(feature)
                    next_sample = now + interval
                rospy.sleep(0.02)
        finally:
            if self.action_pause_yolo:
                self.set_yolo_paused(False)

        label, confidence, reason = self.classify_owner_action(features)
        rospy.loginfo(
            "Owner action verdict: label=%s confidence=%.2f reason=%s samples=%d",
            label,
            confidence,
            reason,
            len(features),
        )
        return label, confidence, reason

    @staticmethod
    def action_to_speech(label):
        messages = {
            "falling": "主人突然摔倒。",
            "waving": "主人正在挥手示意。",
            "lying": "主人当前处于躺下状态。",
            "sitting": "主人当前坐着。",
            "standing": "主人当前站立，没有检测到指定异常动作。",
            "unknown": "我已识别到主人，但动作不确定。",
        }
        return messages.get(label, messages["unknown"])

    def simulate_fall_assist(self):
        """Hold the simulation arm above the owner after a successful approach."""
        if not self.arm_assist_enabled:
            rospy.loginfo("Fall arm assist is disabled")
            return False

        msg = JointState()
        msg.name = ["lift", "gripper"]
        msg.position = [
            clamp(self.arm_assist_lift_height, 0.493, 1.036),
            clamp(self.arm_assist_gripper, 0.025, 0.16),
        ]
        msg.velocity = [0.5, 5.0]

        rospy.loginfo(
            "Fall arm assist: moving end effector above owner lift=%.2f gripper=%.2f",
            msg.position[0],
            msg.position[1],
        )
        deadline = time.time() + max(0.2, self.arm_assist_publish_seconds)
        while not rospy.is_shutdown() and time.time() < deadline:
            self.arm_pub.publish(msg)
            time.sleep(0.1)

        rospy.loginfo(
            "Fall arm assist: end effector is holding above owner for %.1fs",
            max(0.0, self.arm_assist_hold_seconds),
        )
        if self.arm_assist_hold_seconds > 0:
            time.sleep(self.arm_assist_hold_seconds)
        return True

    def crop_candidate_face_region(self, candidate):
        crop = candidate.get("crop")
        if crop is None or crop.size == 0:
            return crop
        height = crop.shape[0]
        top_ratio = clamp(float(self.face_crop_top_ratio), 0.35, 1.0)
        end_y = max(1, int(height * top_ratio))
        return crop[:end_y, :]

    def verify_candidate_with_face(self, candidate):
        if not self.face_ready or self.face_app is None or self.owner_face_embedding is None:
            return None, 0.0, "face unavailable"

        face_image = self.crop_candidate_face_region(candidate)
        if face_image is None or face_image.size == 0:
            return None, 0.0, "empty face crop"

        try:
            start = time.time()
            faces = self.face_app.get(face_image)
            elapsed_ms = (time.time() - start) * 1000.0
        except Exception as exc:
            rospy.logwarn("Face verification failed: %s", exc)
            return None, 0.0, "face verification error"

        candidate_face = self.select_largest_face(faces)
        if candidate_face is None:
            rospy.loginfo("Face verifier saw no face in candidate crop")
            return None, 0.0, "no face"

        candidate_embedding = self.normalize_embedding(candidate_face.embedding)
        if candidate_embedding is None:
            return None, 0.0, "invalid face embedding"

        similarity = float(np.dot(self.owner_face_embedding, candidate_embedding))
        if similarity >= self.face_accept_threshold:
            rospy.loginfo(
                "Face owner verdict: accepted similarity=%.3f elapsed=%.1fms",
                similarity,
                elapsed_ms,
            )
            return True, similarity, "face accepted"
        if self.face_fast_reject and similarity <= self.face_reject_threshold:
            rospy.loginfo(
                "Face owner verdict: rejected similarity=%.3f elapsed=%.1fms",
                similarity,
                elapsed_ms,
            )
            return False, similarity, "face rejected"

        rospy.loginfo(
            "Face owner verdict: uncertain similarity=%.3f elapsed=%.1fms",
            similarity,
            elapsed_ms,
        )
        return None, similarity, "face uncertain"

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image
            self.latest_image_stamp = msg.header.stamp

    def detections_callback(self, msg):
        with self.lock:
            self.latest_detections = list(msg.detections)

    def persons_3d_callback(self, msg):
        with self.lock:
            self.latest_persons_3d = msg

    def scan_callback(self, msg):
        with self.lock:
            self.latest_scan = msg

    def odom_callback(self, msg):
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        with self.lock:
            self.latest_yaw = yaw

    def get_latest_yaw(self):
        with self.lock:
            return self.latest_yaw

    def wait_for_tts_subscriber(self):
        if not self.say_wait_for_subscribers:
            return True
        deadline = time.time() + self.say_wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.say_pub.get_num_connections() > 0:
                return True
            time.sleep(0.05)
        rospy.logwarn("No subscribers connected to %s; TTS message may not be spoken", self.say_topic)
        return False

    def say(self, text, hold=None):
        rospy.loginfo("TTS: %s", text)
        self.wait_for_tts_subscriber()
        repeat_count = max(1, self.say_repeat_count)
        for index in range(repeat_count):
            self.say_pub.publish(String(data=text))
            if index + 1 < repeat_count and self.say_repeat_interval > 0:
                rospy.sleep(self.say_repeat_interval)
        delay = self.say_after_publish_delay if hold is None else float(hold)
        if delay > 0:
            rospy.sleep(delay)

    def set_yolo_paused(self, paused):
        self.pause_yolo_pub.publish(Bool(data=bool(paused)))

    def unload_qwen_model(self):
        if not self.llm_unload_after_request:
            return

        payload = {
            "model": self.llm_model,
            "prompt": "",
            "stream": False,
            "keep_alive": "0s",
        }
        request = urllib.request.Request(
            self.llm_unload_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                response.read()
            rospy.loginfo("Requested Ollama unload for model %s", self.llm_model)
        except Exception as exc:
            rospy.logwarn("Ollama unload request failed: %s", exc)
        if self.llm_unload_wait > 0:
            rospy.sleep(self.llm_unload_wait)

    def build_llm_options(self):
        options = {
            "temperature": 0,
            "top_p": self.llm_top_p,
            "num_predict": self.llm_max_tokens,
            "num_ctx": self.llm_num_ctx,
        }
        if self.llm_num_gpu >= 0:
            options["num_gpu"] = self.llm_num_gpu
        return options

    def prewarm_qwen_model(self):
        rospy.loginfo("Prewarming Qwen vision model %s", self.llm_model)
        if self.llm_pause_yolo:
            self.set_yolo_paused(True)
            rospy.sleep(0.2)
        payload = {
            "model": self.llm_model,
            "messages": [{
                "role": "user",
                "content": (
                    "/no_think\n"
                    "Fast readiness check. Compare image1 and image2. "
                    "Return JSON only: {\"is_owner\": true, \"confidence\": 1.0}"
                ),
                "images": [self.owner_image_b64, self.owner_image_b64],
            }],
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": self.llm_keep_alive,
            "options": self.build_llm_options(),
        }
        request = urllib.request.Request(
            self.llm_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            start = time.time()
            with urllib.request.urlopen(request, timeout=self.llm_timeout) as response:
                response.read()
            rospy.loginfo("Qwen prewarm finished in %.1fs", time.time() - start)
        except Exception as exc:
            rospy.logwarn("Qwen prewarm failed: %s", exc)
        finally:
            if self.llm_unload_after_request:
                self.unload_qwen_model()
            if self.llm_pause_yolo:
                self.set_yolo_paused(False)

    def stop_base(self):
        self.cmd_pub.publish(Twist())

    def load_waypoint_pose(self):
        if not os.path.exists(self.waypoint_file):
            raise RuntimeError("waypoint file not found: %s" % self.waypoint_file)

        root = ET.parse(self.waypoint_file).getroot()
        for waypoint in root.findall("Waypoint"):
            name = waypoint.findtext("Name", "")
            if name != self.waypoint_name:
                continue

            pose = Pose()
            pose.position.x = float(waypoint.findtext("Pos_x", "0"))
            pose.position.y = float(waypoint.findtext("Pos_y", "0"))
            pose.position.z = float(waypoint.findtext("Pos_z", "0"))
            pose.orientation.x = float(waypoint.findtext("Ori_x", "0"))
            pose.orientation.y = float(waypoint.findtext("Ori_y", "0"))
            pose.orientation.z = float(waypoint.findtext("Ori_z", "0"))
            pose.orientation.w = float(waypoint.findtext("Ori_w", "1"))
            return pose

        raise RuntimeError("waypoint not found: %s in %s" % (self.waypoint_name, self.waypoint_file))

    def navigate_to_waypoint(self):
        rospy.loginfo("Waiting for move_base action server")
        if not self.move_base.wait_for_server(rospy.Duration(20.0)):
            raise RuntimeError("move_base action server is not available")

        pose = self.load_waypoint_pose()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = pose

        rospy.loginfo("Navigating to waypoint %s: x=%.3f y=%.3f", self.waypoint_name, pose.position.x, pose.position.y)
        self.move_base.send_goal(goal)
        finished = self.move_base.wait_for_result(rospy.Duration(self.navigate_timeout))
        if not finished:
            self.move_base.cancel_goal()
            raise RuntimeError("move_base timed out while navigating to %s" % self.waypoint_name)

        state = self.move_base.get_state()
        if state != GoalStatus.SUCCEEDED:
            raise RuntimeError("move_base failed with state %s" % state)
        rospy.loginfo("Arrived at waypoint %s", self.waypoint_name)

    def snapshot_candidates(self):
        with self.lock:
            if self.latest_image is None or not self.latest_detections:
                return []
            image = self.latest_image.copy()
            detections = list(self.latest_detections)

        height, width = image.shape[:2]
        candidates = []
        for det in detections:
            if det.class_name and det.class_name != "person":
                continue
            if det.score < self.detection_min_score:
                continue

            xmin = clamp(int(det.xmin), 0, width - 1)
            ymin = clamp(int(det.ymin), 0, height - 1)
            xmax = clamp(int(det.xmax), 0, width - 1)
            ymax = clamp(int(det.ymax), 0, height - 1)
            if xmax <= xmin or ymax <= ymin:
                continue

            box_w = xmax - xmin
            box_h = ymax - ymin
            area_ratio = float(box_w * box_h) / float(width * height)
            if area_ratio < self.detection_min_area_ratio:
                continue

            pad_x = int(box_w * 0.18)
            pad_y = int(box_h * 0.18)
            cx1 = clamp(xmin - pad_x, 0, width - 1)
            cy1 = clamp(ymin - pad_y, 0, height - 1)
            cx2 = clamp(xmax + pad_x, 0, width - 1)
            cy2 = clamp(ymax + pad_y, 0, height - 1)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            center_error = abs((float(det.center_x) / float(width)) - 0.5)
            priority = float(det.score) + area_ratio * 4.0 - center_error * 0.2
            candidates.append({
                "det": det,
                "crop": crop,
                "score": float(det.score),
                "area_ratio": area_ratio,
                "priority": priority,
                "image_width": width,
                "image_height": height,
            })

        candidates.sort(key=lambda item: item["priority"], reverse=True)
        return candidates[:max(1, self.verify_top_k)]

    def resize_for_llm(self, image):
        height, width = image.shape[:2]
        max_side = max(height, width)
        if max_side <= self.llm_image_max_side:
            return image
        scale = float(self.llm_image_max_side) / float(max_side)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def encode_jpeg_b64(self, image):
        image = self.resize_for_llm(image)
        quality = clamp(int(self.llm_jpeg_quality), 40, 95)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("failed to JPEG encode candidate image")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def parse_llm_json(text):
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            cleaned = match.group(0)
        elif cleaned.startswith("{"):
            cleaned = cleaned + "}"
        return json.loads(cleaned)

    @staticmethod
    def parse_owner_verdict(content):
        try:
            parsed = Task1FindOwner.parse_llm_json(content)
            return bool(parsed.get("is_owner", False)), float(parsed.get("confidence", 0.0)), str(parsed.get("reason", ""))
        except Exception:
            owner_match = re.search(r'"?is_owner"?\s*:\s*(true|false)', content, re.I)
            confidence_match = re.search(r'"?confidence"?\s*:\s*([0-9]+(?:\.[0-9]+)?)', content, re.I)
            if owner_match or confidence_match:
                is_owner = owner_match is not None and owner_match.group(1).lower() == "true"
                confidence = float(confidence_match.group(1)) if confidence_match else 0.0
                return is_owner, confidence, "regex fallback"
            raise

    def verify_candidate_with_qwen(self, candidate, manage_resources=True):
        candidate_b64 = self.encode_jpeg_b64(candidate["crop"])
        prompt = (
            "/no_think\n"
            "Fast decision. Return JSON only. "
            "Schema: {\"is_owner\": boolean, \"confidence\": number}. "
            "Image1 is the owner reference. Image2 is the current candidate. "
            "Compare visible appearance, clothing, hair, glasses, and body outline. "
            "If uncertain or the face/body is unclear, return false with low confidence."
        )
        payload = {
            "model": self.llm_model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [self.owner_image_b64, candidate_b64],
            }],
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": self.llm_keep_alive,
            "options": self.build_llm_options(),
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.llm_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        rospy.loginfo("Calling Qwen vision model %s for owner verification", self.llm_model)
        if manage_resources:
            if self.llm_pause_yolo:
                self.set_yolo_paused(True)
                rospy.sleep(0.2)
        try:
            start = time.time()
            with urllib.request.urlopen(request, timeout=self.llm_timeout) as response:
                body = response.read().decode("utf-8")
            rospy.loginfo("Qwen vision request finished in %.1fs", time.time() - start)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            rospy.logwarn("Qwen/Ollama HTTP %s: %s", exc.code, body[:500])
            return False, 0.0, "ollama http error"
        except urllib.error.URLError as exc:
            rospy.logwarn("Qwen/Ollama request failed: %s", exc)
            return False, 0.0, "ollama request failed"
        finally:
            if manage_resources:
                self.unload_qwen_model()
                if self.llm_pause_yolo:
                    self.set_yolo_paused(False)

        try:
            result = json.loads(body)
            content = result.get("message", {}).get("content", "") or result.get("response", "")
            is_owner, confidence, reason = self.parse_owner_verdict(content)
        except Exception as exc:
            rospy.logwarn("Failed to parse Qwen response: %s; raw response: %s", exc, body[:300])
            return False, 0.0, "parse failed"

        rospy.loginfo("Qwen owner verdict: is_owner=%s confidence=%.2f reason=%s", is_owner, confidence, reason)
        return is_owner and confidence >= self.owner_accept_threshold, confidence, reason

    def verify_candidate(self, candidate, manage_resources=True):
        face_decision, face_score, face_reason = self.verify_candidate_with_face(candidate)
        if face_decision is True:
            return True, face_score, face_reason
        if face_decision is False:
            return False, face_score, face_reason
        if not self.face_qwen_fallback:
            return False, face_score, face_reason
        return self.verify_candidate_with_qwen(candidate, manage_resources=manage_resources)

    def wait_for_odom_yaw(self, timeout=3.0):
        deadline = time.time() + timeout
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            yaw = self.get_latest_yaw()
            if yaw is not None:
                return yaw
            rate.sleep()
        return None

    def rotate_to_yaw(self, target_yaw, timeout=18.0):
        deadline = time.time() + timeout
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and time.time() < deadline:
            current_yaw = self.get_latest_yaw()
            if current_yaw is None:
                return False

            error = signed_angle_diff(target_yaw, current_yaw)
            if abs(error) < 0.08:
                self.stop_base()
                return True

            twist = Twist()
            twist.angular.z = clamp(1.2 * error, -self.return_angular_speed, self.return_angular_speed)
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        return False

    def merge_scan_candidates(self, pool, candidates, yaw):
        for candidate in candidates:
            item = dict(candidate)
            item["yaw"] = yaw
            item["priority"] = float(item.get("priority", 0.0))

            replaced = False
            for index, existing in enumerate(pool):
                existing_yaw = existing.get("yaw")
                same_view = yaw is not None and existing_yaw is not None and abs(signed_angle_diff(yaw, existing_yaw)) < 0.35
                same_center = abs(
                    (float(item["det"].center_x) / float(item["image_width"]))
                    - (float(existing["det"].center_x) / float(existing["image_width"]))
                ) < 0.15
                if same_view and same_center:
                    if item["priority"] > existing.get("priority", 0.0):
                        pool[index] = item
                    replaced = True
                    break
            if not replaced:
                pool.append(item)

        pool.sort(key=lambda value: value.get("priority", 0.0), reverse=True)
        return pool[:max(1, self.scan_candidate_pool_size)]

    def verify_candidate_pool(self, pool):
        if not pool:
            return None, None

        for candidate in sorted(pool, key=lambda value: value.get("priority", 0.0), reverse=True)[:max(1, self.verify_after_scan_top_k)]:
            accepted, confidence, reason = self.verify_candidate(candidate, manage_resources=True)
            if accepted:
                det = candidate["det"]
                self.owner_track_center = float(det.center_x) / float(candidate["image_width"])
                rospy.loginfo(
                    "Owner accepted at bbox=(%d,%d,%d,%d), confidence=%.2f reason=%s",
                    det.xmin,
                    det.ymin,
                    det.xmax,
                    det.ymax,
                    confidence,
                    reason,
                )
                return candidate, candidate.get("yaw")
        return None, None

    def scan_for_owner_by_time(self):
        deadline = time.time() + self.scan_duration
        last_verify_time = 0.0
        last_collect_time = 0.0
        owner_candidate = None
        owner_yaw = None
        scan_candidate_pool = []
        rate = rospy.Rate(10)

        rospy.logwarn("No odom yaw received on %s; falling back to timed scan", self.odom_topic)
        while not rospy.is_shutdown() and time.time() < deadline:
            twist = Twist()
            twist.angular.z = self.scan_angular_speed
            self.cmd_pub.publish(twist)

            now = time.time()
            if now - last_collect_time >= self.candidate_collection_interval:
                candidates = self.snapshot_candidates()
                if candidates:
                    scan_candidate_pool = self.merge_scan_candidates(scan_candidate_pool, candidates, self.get_latest_yaw())
                last_collect_time = now

            if self.verify_during_scan and owner_candidate is None and now - last_verify_time >= self.candidate_cooldown:
                candidates = self.snapshot_candidates()
                if candidates:
                    self.stop_base()
                    rospy.sleep(0.2)
                    current_pool = self.merge_scan_candidates([], candidates, self.get_latest_yaw())
                    owner_candidate, owner_yaw = self.verify_candidate_pool(current_pool)
                    last_verify_time = time.time()
                    if owner_candidate is not None:
                        return owner_candidate
            rate.sleep()

        self.stop_base()
        if owner_candidate is None:
            owner_candidate, owner_yaw = self.verify_candidate_pool(scan_candidate_pool)
        if owner_candidate is not None and owner_yaw is not None and self.scan_return_to_owner:
            self.rotate_to_yaw(owner_yaw)
        return owner_candidate

    def scan_for_owner(self):
        rospy.sleep(self.scan_after_arrival_delay)
        start_yaw = self.wait_for_odom_yaw()
        if start_yaw is None:
            return self.scan_for_owner_by_time()

        deadline = time.time() + self.scan_timeout
        last_verify_time = 0.0
        last_yaw = start_yaw
        rotated_angle = 0.0
        owner_candidate = None
        owner_yaw = None
        last_collect_time = 0.0
        scan_candidate_pool = []
        rate = rospy.Rate(10)

        rospy.loginfo(
            "Scanning %.2f rad at %.2f rad/s using odom yaw from %s",
            self.scan_total_angle,
            self.scan_angular_speed,
            self.odom_topic,
        )
        while not rospy.is_shutdown() and rotated_angle < self.scan_total_angle and time.time() < deadline:
            twist = Twist()
            twist.angular.z = self.scan_angular_speed
            self.cmd_pub.publish(twist)

            current_yaw = self.get_latest_yaw()
            if current_yaw is not None:
                delta = signed_angle_diff(current_yaw, last_yaw)
                if abs(delta) < 0.7:
                    rotated_angle += abs(delta)
                last_yaw = current_yaw

            now = time.time()
            rospy.loginfo_throttle(3.0, "Owner scan rotated %.0f / %.0f deg", math.degrees(rotated_angle), math.degrees(self.scan_total_angle))
            if now - last_collect_time >= self.candidate_collection_interval:
                candidates = self.snapshot_candidates()
                if candidates:
                    scan_candidate_pool = self.merge_scan_candidates(scan_candidate_pool, candidates, current_yaw)
                last_collect_time = now

            if self.verify_during_scan and owner_candidate is None and now - last_verify_time >= self.candidate_cooldown:
                candidates = self.snapshot_candidates()
                if candidates:
                    self.stop_base()
                    rospy.sleep(0.2)
                    current_pool = self.merge_scan_candidates([], candidates, current_yaw)
                    owner_candidate, owner_yaw = self.verify_candidate_pool(current_pool)
                    last_verify_time = time.time()
                    if owner_candidate is not None:
                        return owner_candidate
            rate.sleep()

        self.stop_base()
        if rotated_angle < self.scan_total_angle:
            rospy.logwarn("Owner scan stopped before full angle: %.0f / %.0f deg", math.degrees(rotated_angle), math.degrees(self.scan_total_angle))

        if owner_candidate is None:
            rospy.loginfo("Full scan collected %d candidate views; verifying top %d", len(scan_candidate_pool), self.verify_after_scan_top_k)
            owner_candidate, owner_yaw = self.verify_candidate_pool(scan_candidate_pool)

        if owner_candidate is not None and owner_yaw is not None and self.scan_return_to_owner:
            rospy.loginfo("Full scan complete; rotating back to owner heading")
            self.rotate_to_yaw(owner_yaw)
        return owner_candidate

    def front_scan_distance(self):
        with self.lock:
            scan = self.latest_scan
        if scan is None or not scan.ranges:
            return None

        values = []
        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            if abs(angle) <= math.radians(12.0):
                if math.isfinite(distance) and scan.range_min < distance < scan.range_max:
                    values.append(distance)
        if not values:
            return None
        return min(values)

    def select_tracking_detection(self):
        with self.lock:
            if self.latest_image is None or not self.latest_detections:
                return None, None, None
            height, width = self.latest_image.shape[:2]
            detections = list(self.latest_detections)

        best = None
        best_score = -1.0
        for det in detections:
            if det.class_name and det.class_name != "person":
                continue
            if det.score < self.detection_min_score:
                continue
            center_norm = float(det.center_x) / float(width)
            area_ratio = max(0.0, float((det.xmax - det.xmin) * (det.ymax - det.ymin)) / float(width * height))
            if self.owner_track_center is None:
                tracking_penalty = abs(center_norm - 0.5)
            else:
                tracking_penalty = abs(center_norm - self.owner_track_center)
            score = float(det.score) + area_ratio * 5.0 - tracking_penalty * 1.4
            if score > best_score:
                best = det
                best_score = score
        return best, width, height

    def center_owner_in_camera(self, owner_candidate=None):
        if not self.center_owner_before_action:
            return True

        if owner_candidate is not None and "det" in owner_candidate:
            det = owner_candidate["det"]
            image_width = float(owner_candidate.get("image_width", 0.0))
            if image_width > 0:
                self.owner_track_center = float(det.center_x) / image_width

        rospy.loginfo("Centering owner in camera before action recognition")
        deadline = time.time() + max(0.5, self.center_owner_timeout)
        centered_count = 0
        rate = rospy.Rate(12)
        while not rospy.is_shutdown() and time.time() < deadline:
            det, width, _height = self.select_tracking_detection()
            twist = Twist()

            if det is None or width is None or width <= 0:
                last_center = self.owner_track_center if self.owner_track_center is not None else 0.5
                direction = 1.0 if last_center > 0.5 else -1.0
                twist.angular.z = -direction * self.center_owner_lost_turn_speed
                self.cmd_pub.publish(twist)
                centered_count = 0
                rate.sleep()
                continue

            center_norm = float(det.center_x) / float(width)
            self.owner_track_center = center_norm
            x_error = center_norm - 0.5
            if abs(x_error) <= self.center_owner_tolerance:
                self.stop_base()
                centered_count += 1
                if centered_count >= 3:
                    rospy.loginfo("Owner centered in camera: center=%.3f", center_norm)
                    return True
                rate.sleep()
                continue

            centered_count = 0
            twist.angular.z = clamp(
                -self.center_owner_angular_gain * x_error,
                -self.center_owner_max_angular_speed,
                self.center_owner_max_angular_speed,
            )
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        rospy.logwarn("Owner centering timed out; continuing with action recognition")
        return False

    def snapshot_owner_person3d(self, owner_candidate=None):
        with self.lock:
            persons_msg = self.latest_persons_3d
            image = self.latest_image

        if persons_msg is None or not persons_msg.persons:
            return None
        if persons_msg.header.stamp and persons_msg.header.stamp.to_sec() > 0:
            age = abs((rospy.Time.now() - persons_msg.header.stamp).to_sec())
            if age > self.approach_3d_max_age:
                rospy.logwarn_throttle(2.0, "3D person detections are stale: age=%.2fs", age)
                return None

        image_width = None
        if image is not None and len(image.shape) >= 2:
            image_width = float(image.shape[1])

        target_center = self.owner_track_center
        if owner_candidate is not None and "det" in owner_candidate:
            det = owner_candidate["det"]
            candidate_width = float(owner_candidate.get("image_width", 0.0))
            if candidate_width > 0:
                target_center = float(det.center_x) / candidate_width

        best = None
        best_score = -1e9
        for person in persons_msg.persons:
            if person.label and person.label != "person":
                continue
            if person.score < self.detection_min_score:
                continue
            x = float(person.x)
            y = float(person.y)
            z = float(person.z)
            if not all(math.isfinite(value) for value in (x, y, z)):
                continue

            converted = self.person3d_to_base_xy(person)
            if converted is None:
                continue
            person_x, person_y, distance, _mode, _source_frame = converted
            if distance < self.approach_min_person_distance or distance > self.approach_max_person_distance:
                continue
            if person_x <= 0.05:
                continue

            center_score = 0.0
            if target_center is not None and image_width is not None and image_width > 0:
                center_norm = float(person.center_x) / image_width
                center_score = -abs(center_norm - target_center) * 2.0
            score = float(person.score) + center_score - distance * 0.03
            if score > best_score:
                best_score = score
                best = person
        return best

    def person3d_to_base_xy(self, person):
        x = float(person.x)
        y = float(person.y)
        z = float(person.z)
        if not all(math.isfinite(value) for value in (x, y, z)):
            return None

        frame_id = person.header.frame_id or self.approach_base_frame
        frame_lower = frame_id.lower()
        horizontal = math.hypot(x, y)
        mode = self.approach_3d_coordinate_mode
        if mode not in ("auto", "base", "optical"):
            mode = "auto"

        # Kinect/OpenNI-style point clouds use optical camera coordinates:
        # x = image right, y = image down, z = forward depth. In simulation the
        # TF for this cloud can look like base_footprint while still retaining
        # optical axes, so auto mode also detects the depth-dominant case.
        depth_dominant = z > max(0.8, horizontal * 1.35) and z <= self.approach_max_person_distance
        use_optical = mode == "optical" or (mode == "auto" and ("optical" in frame_lower or depth_dominant))

        if use_optical:
            person_x = z
            person_y = self.approach_optical_lateral_sign * x
            source_frame = self.approach_base_frame
            used_mode = "optical"
        else:
            person_x = x
            person_y = y
            source_frame = frame_id
            used_mode = "base"

        distance = math.hypot(person_x, person_y)
        return person_x, person_y, distance, used_mode, source_frame

    def wait_for_owner_person3d(self, owner_candidate=None):
        deadline = time.time() + max(0.1, self.approach_3d_wait_timeout)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            person = self.snapshot_owner_person3d(owner_candidate)
            if person is not None:
                return person
            rate.sleep()
        return None

    def preferred_side_signs(self, person_y):
        preference = self.approach_side_preference.lower()
        if preference == "left":
            first = 1.0
        elif preference == "right":
            first = -1.0
        else:
            first = -1.0 if person_y > 0.0 else 1.0
        return [first, -first]

    def build_owner_goal_from_base_xy(self, person_x, person_y, source_frame, standoff, side_offset, side_sign):
        distance = math.hypot(person_x, person_y)
        if distance <= 1e-3:
            return None

        standoff = clamp(abs(standoff), 0.35, 2.5)
        side_offset = clamp(abs(side_offset), 0.0, 1.0)
        ux = person_x / distance
        uy = person_y / distance
        perp_x = -uy
        perp_y = ux

        goal_x = person_x - ux * standoff + perp_x * side_offset * side_sign
        goal_y = person_y - uy * standoff + perp_y * side_offset * side_sign
        goal_distance = math.hypot(goal_x, goal_y)
        if goal_distance < self.approach_min_goal_distance:
            return None

        face_x = person_x - goal_x
        face_y = person_y - goal_y
        goal_yaw = math.atan2(face_y, face_x)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = source_frame
        goal.target_pose.header.stamp = rospy.Time(0)
        goal.target_pose.pose.position.x = goal_x
        goal.target_pose.pose.position.y = goal_y
        goal.target_pose.pose.position.z = 0.0
        set_pose_yaw(goal.target_pose.pose, goal_yaw)
        return goal, goal_distance

    def owner_approach_goal_candidates(self, person):
        converted = self.person3d_to_base_xy(person)
        if converted is None:
            return []
        person_x, person_y, distance, used_mode, source_frame = converted
        standoff = clamp(abs(self.approach_standoff_distance), 0.35, 1.5)
        side_offset = clamp(abs(self.approach_side_offset), 0.0, 1.0)
        side_signs = self.preferred_side_signs(person_y)

        candidates = []
        seen = set()

        def add_candidate(candidate_standoff, candidate_side_offset, candidate_side_sign, staged):
            built = self.build_owner_goal_from_base_xy(
                person_x,
                person_y,
                source_frame,
                candidate_standoff,
                candidate_side_offset,
                candidate_side_sign,
            )
            if built is None:
                return
            local_goal, goal_distance = built
            key = (
                round(local_goal.target_pose.pose.position.x, 2),
                round(local_goal.target_pose.pose.position.y, 2),
                round(candidate_standoff, 2),
                round(candidate_side_offset, 2),
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append({
                "local_goal": local_goal,
                "person_distance": distance,
                "goal_distance": goal_distance,
                "standoff": candidate_standoff,
                "side_offset": candidate_side_offset,
                "coord_mode": used_mode,
                "person_x": person_x,
                "person_y": person_y,
                "staged": staged,
            })

        max_step = max(0.0, self.approach_max_goal_distance)
        final_goal_distance = max(0.0, distance - standoff)
        if max_step > 0.0 and final_goal_distance > max_step:
            # Long goals near furniture often fail global planning. First move a
            # short, plan-checkable step toward the owner, then refresh point cloud.
            staged_standoff = max(0.35, distance - max_step)
            add_candidate(staged_standoff, 0.0, side_signs[0], True)
            for side_sign in side_signs:
                add_candidate(staged_standoff, min(side_offset, 0.20), side_sign, True)
            return candidates

        standoffs = [standoff, max(standoff, 0.75), max(standoff, 0.95), max(standoff, 1.15)]
        for candidate_standoff in standoffs:
            for candidate_side_offset, signs in (
                (0.0, [side_signs[0]]),
                (side_offset, side_signs),
                (side_offset, [side_signs[1]]),
            ):
                for side_sign in signs:
                    add_candidate(candidate_standoff, candidate_side_offset, side_sign, False)
        return candidates

    def owner_side_goal_from_3d(self, person):
        candidates = self.owner_approach_goal_candidates(person)
        if not candidates:
            return None
        best = candidates[0]
        return (
            best["local_goal"],
            best["person_distance"],
            best["goal_distance"],
            best["standoff"],
            best["side_offset"],
            best["coord_mode"],
            best["person_x"],
            best["person_y"],
        )

    def transform_approach_goal(self, goal):
        source_frame = goal.target_pose.header.frame_id
        target_frame = self.approach_goal_frame
        if not target_frame or source_frame == target_frame:
            goal.target_pose.header.stamp = rospy.Time.now()
            return goal
        if self.tf_buffer is None:
            rospy.logwarn("Cannot transform 3D approach goal from %s to %s: tf2 unavailable", source_frame, target_frame)
            return None

        pose = PoseStamped()
        pose.header = goal.target_pose.header
        pose.pose = goal.target_pose.pose
        try:
            transformed_pose = self.tf_buffer.transform(pose, target_frame, rospy.Duration(0.5))
        except Exception as exc:
            rospy.logwarn("Cannot transform 3D approach goal from %s to %s: %s", source_frame, target_frame, exc)
            return None

        transformed_pose.header.stamp = rospy.Time.now()
        transformed_goal = MoveBaseGoal()
        transformed_goal.target_pose = transformed_pose
        return transformed_goal

    def current_robot_pose(self, target_frame):
        pose = PoseStamped()
        pose.header.frame_id = self.approach_base_frame
        pose.header.stamp = rospy.Time(0)
        pose.pose.orientation.w = 1.0
        if pose.header.frame_id == target_frame:
            pose.header.stamp = rospy.Time.now()
            return pose
        if self.tf_buffer is None:
            return None
        try:
            current = self.tf_buffer.transform(pose, target_frame, rospy.Duration(0.5))
            current.header.stamp = rospy.Time.now()
            return current
        except Exception as exc:
            rospy.logwarn("Cannot transform current robot pose from %s to %s: %s", self.approach_base_frame, target_frame, exc)
            return None

    def goal_has_global_plan(self, goal):
        if not self.approach_plan_check:
            return True
        frame_id = goal.target_pose.header.frame_id
        start = self.current_robot_pose(frame_id)
        if start is None:
            return True
        try:
            rospy.wait_for_service(self.approach_plan_service, timeout=0.2)
            response = self.make_plan(start, goal.target_pose, self.approach_plan_tolerance)
        except Exception as exc:
            rospy.logwarn_throttle(3.0, "Cannot check approach plan via %s: %s", self.approach_plan_service, exc)
            return True
        pose_count = len(response.plan.poses)
        if pose_count <= 1:
            rospy.logwarn(
                "Rejected 3D approach candidate with no global plan: frame=%s goal=(%.2f, %.2f)",
                frame_id,
                goal.target_pose.pose.position.x,
                goal.target_pose.pose.position.y,
            )
            return False
        rospy.loginfo_throttle(2.0, "3D approach candidate has global plan with %d poses", pose_count)
        return True

    def send_approach_goal(self, goal, goal_distance):
        timeout = max(3.0, min(self.approach_goal_timeout, self.approach_stage_timeout + goal_distance * 4.0))
        self.move_base.send_goal(goal)
        finished = self.move_base.wait_for_result(rospy.Duration(timeout))
        if not finished:
            self.move_base.cancel_goal()
            rospy.logwarn("3D owner approach stage timed out after %.1fs", timeout)
            return False

        state = self.move_base.get_state()
        if state != GoalStatus.SUCCEEDED:
            rospy.logwarn("3D owner approach stage failed with move_base state %s", state)
            return False
        return True

    def approach_owner_by_3d_servo(self, owner_candidate=None):
        rospy.logwarn("Falling back to local point-cloud servo approach")
        if self.approach_pause_yolo:
            self.set_yolo_paused(False)
            rospy.sleep(0.1)

        deadline = time.time() + max(1.0, self.approach_servo_timeout)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            person = self.snapshot_owner_person3d(owner_candidate)
            if person is None:
                self.stop_base()
                rate.sleep()
                continue
            converted = self.person3d_to_base_xy(person)
            if converted is None:
                self.stop_base()
                rate.sleep()
                continue

            person_x, person_y, distance, coord_mode, _source_frame = converted
            front_distance = self.front_scan_distance()
            stop_distance = max(0.45, self.approach_servo_stop_distance)
            if distance <= stop_distance or (front_distance is not None and front_distance <= stop_distance):
                self.stop_base()
                rospy.loginfo(
                    "Point-cloud servo approach complete: mode=%s dist=%.2f front=%.2f",
                    coord_mode,
                    distance,
                    front_distance if front_distance is not None else -1.0,
                )
                return True

            bearing = math.atan2(person_y, max(0.05, person_x))
            twist = Twist()
            twist.angular.z = clamp(self.approach_servo_angular_gain * bearing, -0.35, 0.35)
            if abs(bearing) < 0.55:
                clearance_scale = 1.0
                if front_distance is not None:
                    clearance_scale = clamp((front_distance - stop_distance) / 0.6, 0.0, 1.0)
                twist.linear.x = min(self.approach_servo_linear_speed, max(0.06, (distance - stop_distance) * 0.25)) * clearance_scale
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        rospy.logwarn("Point-cloud servo approach timed out")
        return False

    def align_owner_after_approach(self, owner_candidate=None):
        """Rotate in place until the latest 3D owner position is straight ahead."""
        if not self.approach_final_align_enabled:
            return True

        deadline = time.time() + max(0.5, self.approach_final_align_timeout)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            person = self.snapshot_owner_person3d(owner_candidate)
            if person is None:
                self.stop_base()
                rate.sleep()
                continue

            converted = self.person3d_to_base_xy(person)
            if converted is None:
                self.stop_base()
                rate.sleep()
                continue

            _person_x, person_y, _distance, _mode, _source_frame = converted
            bearing = math.atan2(person_y, max(0.05, _person_x))
            if abs(bearing) <= self.approach_final_align_tolerance:
                self.stop_base()
                rospy.loginfo("Final owner alignment complete: bearing=%.3f rad", bearing)
                return True

            twist = Twist()
            twist.angular.z = clamp(
                self.approach_final_align_angular_gain * bearing,
                -self.approach_final_align_max_speed,
                self.approach_final_align_max_speed,
            )
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        rospy.logwarn("Final owner alignment timed out")
        return False

    def refine_owner_stop_with_3d_lidar(self, owner_candidate=None):
        """Use 3D owner tracking for precision and lidar as a forward safety limit."""
        if not self.approach_fine_refine_enabled:
            return True

        rospy.loginfo(
            "Starting fine owner stop: target_distance=%.2fm lidar_stop=%.2fm",
            self.approach_standoff_distance,
            self.approach_fine_lidar_stop_distance + self.approach_fine_lidar_margin,
        )
        if self.approach_pause_yolo:
            self.set_yolo_paused(False)
            rospy.sleep(0.1)

        target_distance = clamp(abs(self.approach_standoff_distance), 0.35, 1.5)
        deadline = time.time() + max(1.0, self.approach_fine_refine_timeout)
        stable_cycles = 0
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() < deadline:
            person = self.snapshot_owner_person3d(owner_candidate)
            if person is None:
                self.stop_base()
                stable_cycles = 0
                rate.sleep()
                continue

            converted = self.person3d_to_base_xy(person)
            if converted is None:
                self.stop_base()
                stable_cycles = 0
                rate.sleep()
                continue

            person_x, person_y, distance, coord_mode, _source_frame = converted
            bearing = math.atan2(person_y, max(0.05, person_x))
            distance_error = distance - target_distance
            lidar_front = self.front_scan_distance()
            lidar_guard_distance = (
                self.approach_fine_lidar_stop_distance
                + self.approach_fine_lidar_margin
            )

            # The depth target controls precision. Lidar only prevents the base
            # from driving into a close obstacle or the person's lower body.
            lidar_blocked = (
                lidar_front is not None
                and lidar_front <= lidar_guard_distance
            )
            within_distance = abs(distance_error) <= self.approach_fine_distance_tolerance
            within_bearing = abs(bearing) <= self.approach_fine_bearing_tolerance

            if within_distance and within_bearing:
                stable_cycles += 1
                self.stop_base()
                if stable_cycles >= self.approach_fine_stable_cycles:
                    rospy.loginfo(
                        "Fine owner stop complete: mode=%s distance=%.2f error=%.2f bearing=%.3f lidar=%.2f",
                        coord_mode,
                        distance,
                        distance_error,
                        bearing,
                        lidar_front if lidar_front is not None else -1.0,
                    )
                    return True
                rate.sleep()
                continue

            stable_cycles = 0
            twist = Twist()
            twist.angular.z = clamp(
                self.approach_fine_angular_gain * bearing,
                -self.approach_fine_max_angular_speed,
                self.approach_fine_max_angular_speed,
            )

            if distance_error > self.approach_fine_distance_tolerance and not lidar_blocked:
                twist.linear.x = clamp(
                    self.approach_fine_linear_gain * distance_error,
                    0.0,
                    self.approach_fine_max_linear_speed,
                )
                if lidar_front is not None:
                    clearance = lidar_front - lidar_guard_distance
                    twist.linear.x = min(
                        twist.linear.x,
                        max(0.0, clearance) * 0.35,
                    )
            elif distance_error < -self.approach_fine_distance_tolerance:
                twist.linear.x = -min(
                    self.approach_fine_max_reverse_speed,
                    abs(self.approach_fine_linear_gain * distance_error),
                )

            if lidar_blocked and twist.linear.x > 0.0:
                twist.linear.x = 0.0
                rospy.logwarn_throttle(
                    1.0,
                    "Fine owner stop blocked by lidar: front=%.2f stop=%.2f",
                    lidar_front,
                    lidar_guard_distance,
                )

            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        rospy.logwarn("Fine owner stop timed out; keeping the one-shot navigation result")
        return True

    def approach_owner_by_3d(self, owner_candidate=None):
        if not self.approach_enabled:
            return True
        if not self.approach_use_3d:
            return self.approach_owner()

        rospy.loginfo("Approaching owner using point-cloud 3D position from %s", self.persons_3d_topic)
        try:
            for stage_index in range(self.approach_3d_max_stages):
                if self.approach_pause_yolo:
                    self.set_yolo_paused(False)
                    rospy.sleep(0.1)

                with self.lock:
                    self.latest_persons_3d = None
                person = self.wait_for_owner_person3d(owner_candidate)
                if person is None:
                    rospy.logwarn("No usable owner 3D position from point cloud")
                    if self.approach_servo_fallback:
                        return self.approach_owner_by_3d_servo(owner_candidate)
                    if self.approach_fallback_2d:
                        return self.approach_owner()
                    return False

                candidates = self.owner_approach_goal_candidates(person)
                if not candidates:
                    rospy.loginfo("Owner is already close enough for 3D approach")
                    self.stop_base()
                    return True

                sent_goal = False
                for candidate_index, candidate in enumerate(candidates):
                    local_goal = candidate["local_goal"]
                    goal = self.transform_approach_goal(local_goal)
                    if goal is None:
                        continue
                    if not self.goal_has_global_plan(goal):
                        continue

                    rospy.loginfo(
                        "Owner 3D raw frame=%s raw=(%.2f, %.2f, %.2f) mode=%s base_xy=(%.2f, %.2f) dist=%.2f; stage=%d candidate=%d staged=%s local goal=(%.2f, %.2f) standoff=%.2f side=%.2f; move_base frame=%s goal=(%.2f, %.2f)",
                        person.header.frame_id,
                        person.x,
                        person.y,
                        person.z,
                        candidate["coord_mode"],
                        candidate["person_x"],
                        candidate["person_y"],
                        candidate["person_distance"],
                        stage_index + 1,
                        candidate_index + 1,
                        candidate["staged"],
                        local_goal.target_pose.pose.position.x,
                        local_goal.target_pose.pose.position.y,
                        candidate["standoff"],
                        candidate["side_offset"],
                        goal.target_pose.header.frame_id,
                        goal.target_pose.pose.position.x,
                        goal.target_pose.pose.position.y,
                    )

                    if self.approach_pause_yolo:
                        self.set_yolo_paused(True)
                        rospy.sleep(0.1)
                    sent_goal = True
                    if self.send_approach_goal(goal, candidate["goal_distance"]):
                        rospy.loginfo("3D owner approach stage complete: goal_distance=%.2f staged=%s", candidate["goal_distance"], candidate["staged"])
                        self.stop_base()
                        if candidate["staged"]:
                            break
                        self.refine_owner_stop_with_3d_lidar(owner_candidate)
                        return True
                    rospy.logwarn("Trying next 3D approach candidate")
                    if self.approach_pause_yolo:
                        self.set_yolo_paused(False)
                        rospy.sleep(0.1)

                if not sent_goal:
                    rospy.logwarn("No plannable 3D approach candidate found")
                    if self.approach_servo_fallback:
                        return self.approach_owner_by_3d_servo(owner_candidate)
                    if self.approach_fallback_2d:
                        return self.approach_owner()
                    return False

            rospy.logwarn("3D owner approach reached max stages without final close goal")
            if self.approach_servo_fallback:
                return self.approach_owner_by_3d_servo(owner_candidate)
            return False
        finally:
            if self.approach_pause_yolo:
                self.set_yolo_paused(False)

    def approach_owner(self):
        if not self.approach_enabled:
            return True

        rospy.loginfo("Approaching owner using camera bbox and front laser distance")
        deadline = time.time() + self.approach_timeout
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            det, width, height = self.select_tracking_detection()
            front_distance = self.front_scan_distance()

            if det is None or width is None:
                twist = Twist()
                twist.angular.z = 0.18
                self.cmd_pub.publish(twist)
                rate.sleep()
                continue

            center_norm = float(det.center_x) / float(width)
            self.owner_track_center = center_norm
            x_error = (center_norm - 0.5) * 2.0
            height_ratio = max(0.0, float(det.ymax - det.ymin) / float(height))
            centered = abs(x_error) <= self.approach_center_tolerance * 2.0
            close_by_scan = front_distance is not None and front_distance <= self.approach_stop_distance
            close_by_image = height_ratio >= self.approach_target_height_ratio

            if centered and (close_by_scan or close_by_image):
                self.stop_base()
                rospy.loginfo("Approach complete: front_distance=%s height_ratio=%.2f", front_distance, height_ratio)
                return True

            twist = Twist()
            twist.angular.z = clamp(-self.approach_angular_gain * x_error, -0.45, 0.45)
            if abs(x_error) < 0.75 and not close_by_scan and not close_by_image:
                twist.linear.x = self.approach_linear_speed * max(0.25, 1.0 - abs(x_error))
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        rospy.logwarn("Approach timed out")
        return False

    def run(self):
        if self.speak_on_start:
            self.say("我开始前往客厅寻找主人。")

        self.navigate_to_waypoint()

        if self.speak_on_arrival:
            self.say("我已到达客厅，开始环顾寻找主人。")

        owner_candidate = self.scan_for_owner()
        if owner_candidate is None:
            self.say("我没有确认主人，请再给我一次机会。")
            return False

        if self.speak_on_owner_found:
            self.say("我已经识别到主人。")

        self.center_owner_in_camera(owner_candidate)
        action_label, action_confidence, action_reason = self.recognize_owner_action(owner_candidate)
        rospy.loginfo(
            "Owner action summary: label=%s confidence=%.2f reason=%s",
            action_label,
            action_confidence,
            action_reason,
        )
        self.say(self.action_to_speech(action_label), hold=self.action_speech_hold)
        if self.approach_enabled:
            approached = self.approach_owner_by_3d(owner_candidate)
            if approached:
                self.align_owner_after_approach(owner_candidate)
                if action_label == "falling":
                    self.simulate_fall_assist()
                elif action_label == "waving":
                    self.say("请问您需要什么帮助？")
                elif action_label in ("lying", "sitting"):
                    self.say("请指示。")
            else:
                self.say("我已经识别到主人位置，但暂时无法安全靠近。")
        return True


def main():
    rospy.init_node("task1_find_owner")
    node = None
    try:
        node = Task1FindOwner()
        success = node.run()
        rospy.loginfo("task1_find_owner finished: success=%s", success)
    except Exception as exc:
        if node is not None:
            node.stop_base()
        rospy.logfatal("task1_find_owner failed: %s", exc)
        try:
            pub = rospy.Publisher(rospy.get_param("~say_topic", "/voice/say"), String, queue_size=1)
            rospy.sleep(0.5)
            pub.publish(String(data="任务启动失败，请检查导航、识别模型和主人照片。"))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
