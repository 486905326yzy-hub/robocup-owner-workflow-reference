#!/usr/bin/python3.9
# coding=utf-8

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import actionlib
import cv2
import numpy as np
import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from person_reid_owner_test import PersonReidOwnerTest


class OwnerVoiceReidTest(PersonReidOwnerTest):
    def __init__(self):
        super().__init__()

        self.asr_topic = rospy.get_param("~asr_topic", "/voice/asr_text")
        self.answer_timeout = max(1.0, float(rospy.get_param("~answer_timeout", 12.0)))
        self.asr_wait_for_publishers = bool(rospy.get_param("~asr_wait_for_publishers", True))
        self.asr_wait_timeout = max(0.0, float(rospy.get_param("~asr_wait_timeout", 45.0)))
        self.max_name_retries = max(
            1,
            int(rospy.get_param("~max_name_retries", rospy.get_param("~max_retries", 3))),
        )
        self.max_name_length = max(1, int(rospy.get_param("~max_name_length", 12)))
        self.owner_count = max(1, int(rospy.get_param("~owner_count", 3)))
        self.owner_profiles = []
        self.skipped_owner_indices = []
        self.current_owner_index = 1
        self.last_match_owner_index = None
        self.last_match_method = None
        self.base_profile_path = self.profile_path
        self.base_metadata_path = self.metadata_path

        self.face_verify_enabled = bool(rospy.get_param("~face_verify_enabled", True))
        self.face_auto_download = bool(rospy.get_param("~face_auto_download", False))
        self.face_model_name = rospy.get_param("~face_model_name", "buffalo_sc")
        self.face_model_root = os.path.expanduser(rospy.get_param("~face_model_root", "~/.insightface"))
        self.face_ctx_id = int(rospy.get_param("~face_ctx_id", -1))
        self.face_det_size = int(rospy.get_param("~face_det_size", 480))
        self.face_det_thresh = float(rospy.get_param("~face_det_thresh", 0.35))
        self.face_accept_threshold = float(rospy.get_param("~face_accept_threshold", 0.45))
        self.face_reject_threshold = float(rospy.get_param("~face_reject_threshold", 0.25))
        self.face_fast_reject = bool(rospy.get_param("~face_fast_reject", True))
        self.face_min_reid_score = float(rospy.get_param("~face_min_reid_score", 0.45))
        self.face_identity_weight = max(
            0.0,
            min(1.0, float(rospy.get_param("~face_identity_weight", 0.70))),
        )
        self.lying_face_identity_weight = max(
            0.0,
            min(1.0, float(rospy.get_param("~lying_face_identity_weight", 0.10))),
        )
        self.owner_score_margin_threshold = max(
            0.0,
            float(rospy.get_param("~owner_score_margin_threshold", 0.04)),
        )
        self.face_crop_padding = float(rospy.get_param("~face_crop_padding", 0.20))
        self.face_crop_top_ratio = float(rospy.get_param("~face_crop_top_ratio", 0.68))
        # Team side-face/weak-face fallback: identity switches to OSNet only
        # when InsightFace explicitly reports a limited-quality face.
        self.side_face_reid_enabled = bool(rospy.get_param("~side_face_reid_enabled", True))
        self.side_face_nose_offset_threshold = max(
            0.0, float(rospy.get_param("~side_face_nose_offset_threshold", 0.35))
        )
        self.side_face_min_face_size = max(
            0.0, float(rospy.get_param("~side_face_min_face_size", 32.0))
        )
        self.side_face_min_det_score = max(
            0.0, min(1.0, float(rospy.get_param("~side_face_min_det_score", 0.55)))
        )
        self.side_face_reid_threshold = float(
            rospy.get_param("~side_face_reid_threshold", 0.75)
        )
        self.side_face_reid_margin = max(
            self.owner_score_margin_threshold,
            float(rospy.get_param("~side_face_reid_margin", 0.08)),
        )
        self.side_face_reid_required_consecutive = max(
            self.match_required_consecutive,
            int(rospy.get_param("~side_face_reid_required_consecutive", 3)),
        )
        self.side_face_fusion_enabled = bool(
            rospy.get_param("~side_face_fusion_enabled", True)
        )
        self.side_face_fusion_weight = max(
            0.0, min(1.0, float(rospy.get_param("~side_face_fusion_weight", 0.12)))
        )
        self.side_face_face_score_offset = max(
            0.0, min(0.50, float(rospy.get_param("~side_face_face_score_offset", 0.32)))
        )
        self.side_face_face_score_floor = max(
            0.0, min(1.0, float(rospy.get_param("~side_face_face_score_floor", 0.40)))
        )
        self.body_query_variants_enabled = bool(
            rospy.get_param("~body_query_variants_enabled", True)
        )
        # Dedicated close-range face enrollment happens before the existing
        # full-body Re-ID recording. It uses the already-loaded InsightFace
        # model and falls back safely to Re-ID if no clear face is available.
        self.face_close_registration_enabled = bool(
            rospy.get_param("~face_close_registration_enabled", True)
        )
        self.face_close_record_seconds = max(
            0.8, float(rospy.get_param("~face_close_record_seconds", 3.2))
        )
        self.face_close_sample_count = max(
            1, int(rospy.get_param("~face_close_sample_count", 3))
        )
        self.face_close_sample_interval = max(
            0.10, float(rospy.get_param("~face_close_sample_interval", 0.45))
        )
        self.face_close_min_samples = max(
            1, int(rospy.get_param("~face_close_min_samples", 4))
        )
        self.face_close_min_size = max(
            32, int(rospy.get_param("~face_close_min_size", 60))
        )
        self.face_close_start_text = rospy.get_param(
            "~face_close_start_text", "请靠近相机约半米，正对镜头，开始登记人脸。"
        )
        self.face_close_front_text = rospy.get_param(
            "~face_close_front_text", "请正对镜头，保持自然表情。"
        )
        self.face_close_left_text = rospy.get_param(
            "~face_close_left_text", "请缓慢向左转头。"
        )
        self.face_close_right_text = rospy.get_param(
            "~face_close_right_text", "请缓慢向右转头。"
        )
        self.face_close_done_text = rospy.get_param(
            "~face_close_done_text", "人脸登记完成，请退后站到相机前方，开始记录外观特征。"
        )
        self.face_close_fallback_text = rospy.get_param(
            "~face_close_fallback_text", "近距离人脸采集不足，将继续记录外观特征。"
        )
        self.face_close_poses = [
            ("front", "正面", self.face_close_front_text),
            ("left", "左侧", self.face_close_left_text),
            ("right", "右侧", self.face_close_right_text),
        ]

        self.face_app = None
        self.face_model_ready = False
        self.face_ready = False
        self.owner_face_embedding = None
        self.owner_face_embedding_bank = None
        self.current_record_face_embeddings = []
        self.current_record_pose_id = ""
        self.current_record_pose_name = ""

        default_action_launch_file = os.path.abspath(
            os.path.join(
                SCRIPT_DIR,
                "..",
                "..",
                "offline_voice_bridge",
                "launch",
                "qwen_action_recognition.launch",
            )
        )
        self.action_recognition_enabled = bool(
            rospy.get_param("~action_recognition_enabled", True)
        )
        self.action_backend = str(rospy.get_param("~action_backend", "qwen")).strip().lower()
        if self.action_backend not in {"qwen", "yolo_pose"}:
            rospy.logwarn("Unknown action backend %s; using qwen", self.action_backend)
            self.action_backend = "qwen"
        self.action_launch_file = os.path.abspath(
            os.path.expanduser(
                rospy.get_param("~action_launch_file", default_action_launch_file)
            )
        )
        self.action_node_name = str(
            rospy.get_param("~action_node_name", "owner_action_qwen")
        ).strip()
        self.action_result_topic = str(
            rospy.get_param(
                "~action_result_topic",
                "/owner_voice_reid_test/action_result",
            )
        ).strip()
        self.action_timeout = max(
            15.0,
            float(rospy.get_param("~action_timeout", 150.0)),
        )
        self.action_speech_grace = max(
            0.2,
            float(rospy.get_param("~action_speech_grace", 1.0)),
        )
        self.action_show_window = bool(
            rospy.get_param("~action_show_window", False)
        )
        self.action_llm_url = str(
            rospy.get_param(
                "~action_llm_url",
                "http://127.0.0.1:11434/api/chat",
            )
        ).strip()
        self.action_llm_model = str(
            rospy.get_param("~action_llm_model", "qwen3.5:2b")
        ).strip()
        self.action_pointcloud_enabled = bool(
            rospy.get_param("~action_pointcloud_enabled", False)
        )
        self.action_startup_settle_seconds = max(
            0.0,
            float(rospy.get_param("~action_startup_settle_seconds", 0.6)),
        )
        self.action_capture_delay = max(
            0.0,
            float(rospy.get_param("~action_capture_delay", 0.3)),
        )
        self.action_capture_duration = max(
            0.0,
            float(rospy.get_param("~action_capture_duration", 5.0)),
        )
        self.action_capture_frame_count = max(
            1,
            int(rospy.get_param("~action_capture_frame_count", 9)),
        )
        self.action_llm_frame_count = max(
            1,
            int(rospy.get_param("~action_llm_frame_count", 5)),
        )
        self.action_pose_enabled = bool(
            rospy.get_param("~action_pose_enabled", True)
        )
        self.action_pose_model_path = str(
            rospy.get_param("~action_pose_model_path", "")
        ).strip()
        self.action_pose_device = str(
            rospy.get_param("~action_pose_device", "cpu")
        ).strip()
        self.action_pose_image_size = max(
            160,
            int(rospy.get_param("~action_pose_image_size", 416)),
        )
        self.action_pose_confidence = float(
            rospy.get_param("~action_pose_confidence", 0.25)
        )
        self.action_pose_iou = float(rospy.get_param("~action_pose_iou", 0.45))
        self.action_pose_max_detections = max(
            1,
            int(rospy.get_param("~action_pose_max_detections", 4)),
        )
        self.action_llm_timeout = max(
            1.0,
            float(rospy.get_param("~action_llm_timeout", 90.0)),
        )
        self.action_llm_keep_alive = str(
            rospy.get_param("~action_llm_keep_alive", "30m")
        ).strip()
        self.action_llm_max_tokens = max(
            1,
            int(rospy.get_param("~action_llm_max_tokens", 32)),
        )
        self.action_llm_num_ctx = max(
            512,
            int(rospy.get_param("~action_llm_num_ctx", 8192)),
        )
        self.action_jpeg_quality = max(
            1,
            min(100, int(rospy.get_param("~action_jpeg_quality", 65))),
        )
        self.action_image_max_width = max(
            0,
            int(rospy.get_param("~action_image_max_width", 384)),
        )
        self.action_model_warmup_retries = max(
            1,
            int(rospy.get_param("~action_model_warmup_retries", 3)),
        )
        self.action_model_warmup_retry_delay = max(
            0.0,
            float(rospy.get_param("~action_model_warmup_retry_delay", 2.0)),
        )
        self.action_use_owner_roi = bool(
            rospy.get_param("~action_use_owner_roi", True)
        )
        self.action_roi_padding = max(
            0.0,
            min(1.0, float(rospy.get_param("~action_roi_padding", 0.55))),
        )
        self.action_yolo_pause_settle_seconds = max(
            0.0,
            float(rospy.get_param("~action_yolo_pause_settle_seconds", 0.8)),
        )
        self.action_result_connect_timeout = max(
            0.0,
            float(rospy.get_param("~action_result_connect_timeout", 10.0)),
        )
        self.action_pause_yolo = bool(
            rospy.get_param("~action_pause_yolo", True)
        )
        self.yolo_pause_topic = str(
            rospy.get_param("~yolo_pause_topic", "/yoloworld/pause")
        ).strip()
        self.action_launch_parent = None
        self.action_process = None
        self.action_result_sub = None
        self.action_result_event = threading.Event()
        self.action_result = None
        self.action_launch_lock = threading.Lock()
        self.action_completed = False

        self.show_yolo_window = bool(rospy.get_param("~show_yolo_window", True))
        self.yolo_window_name = rospy.get_param("~yolo_window_name", "Owner YOLO-Person Box")
        self.yolo_window_ready = False
        self.yolo_window_failed = False

        self.navigate_enabled = bool(rospy.get_param("~navigate_enabled", True))
        self.waypoint_name = str(rospy.get_param("~waypoint_name", "living_room")).strip()
        self.waypoint_file = os.path.expanduser(
            rospy.get_param("~waypoint_file", "/home/ubuntu20/waypoints.xml")
        )
        self.move_base_server_timeout = max(
            1.0,
            float(rospy.get_param("~move_base_server_timeout", 25.0)),
        )
        self.navigate_timeout = max(1.0, float(rospy.get_param("~navigate_timeout", 120.0)))
        self.clear_costmaps_before_navigation = bool(
            rospy.get_param("~clear_costmaps_before_navigation", True)
        )
        self.clear_costmaps_service = rospy.get_param(
            "~clear_costmaps_service", "/move_base/clear_costmaps"
        )
        self.clear_costmaps_timeout = max(
            0.1,
            float(rospy.get_param("~clear_costmaps_timeout", 5.0)),
        )
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.scan_angular_speed = float(rospy.get_param("~scan_angular_speed", -0.25))
        self.scan_total_angle = max(0.1, float(rospy.get_param("~scan_total_angle", math.pi)))
        self.scan_timeout = max(1.0, float(rospy.get_param("~scan_timeout", 35.0)))
        self.scan_step_angle = max(0.02, float(rospy.get_param("~scan_step_angle", 0.18)))
        self.scan_pause = max(0.0, float(rospy.get_param("~scan_pause", 0.35)))
        self.scan_after_arrival_delay = max(
            0.0,
            float(rospy.get_param("~scan_after_arrival_delay", 0.2)),
        )
        self.scan_return_to_owner = bool(rospy.get_param("~scan_return_to_owner", True))
        self.center_owner_enabled = bool(rospy.get_param("~center_owner_enabled", True))
        self.center_owner_timeout = max(0.5, float(rospy.get_param("~center_owner_timeout", 5.0)))
        self.center_owner_tolerance = max(
            0.01,
            float(rospy.get_param("~center_owner_tolerance", 0.06)),
        )
        self.center_owner_angular_gain = float(
            rospy.get_param("~center_owner_angular_gain", 0.65)
        )
        self.center_owner_max_angular_speed = abs(
            float(rospy.get_param("~center_owner_max_angular_speed", 0.30))
        )
        self.center_owner_lost_turn_speed = abs(
            float(rospy.get_param("~center_owner_lost_turn_speed", 0.10))
        )
        self.owner_not_found_text = rospy.get_param(
            "~owner_not_found_text", "没有找到主人。"
        )
        self.navigation_failed_text = rospy.get_param(
            "~navigation_failed_text", "导航到客厅失败。"
        )
        self.latest_yaw = None
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        self.name_prompt_text = rospy.get_param(
            "~name_prompt_text",
            "请说姓名",
        )
        self.indexed_name_prompt_text = rospy.get_param(
            "~indexed_name_prompt_text",
            "请说第%s位主人的姓名。",
        )
        self.name_confirm_prompt_text = rospy.get_param(
            "~name_confirm_prompt_text",
            "姓名%s是否正确？正确请说正确，错误请重新说姓名。",
        )
        # Kept as a configurable inter-prompt pause. Do not close ASR before
        # confirmation: owners commonly answer "正确" immediately after TTS.
        # Long prompt echoes are rejected by the confirmation matcher instead.
        self.name_confirm_reply_delay = max(
            0.0,
            float(rospy.get_param("~name_confirm_reply_delay", 0.0)),
        )
        self.name_retry_text = rospy.get_param(
            "~name_retry_text",
            "请按照姓名加名字的格式回答，例如姓名张三。",
        )
        # Suppress repeated name-format prompts caused by the robot's own TTS
        # echo. ASR keeps listening continuously; only the spoken reminder is
        # rate-limited.
        self.name_retry_prompt_interval = max(
            0.0,
            float(rospy.get_param("~name_retry_prompt_interval", 8.0)),
        )
        self._last_name_retry_prompt_at = float("-inf")
        self.name_timeout_text = rospy.get_param("~name_timeout_text", "没有听到主人姓名。")
        self.name_invalid_text = rospy.get_param(
            "~name_invalid_text",
            "没有识别到姓名格式，请以姓名开头再说一次。",
        )
        self.name_failed_text = rospy.get_param(
            "~name_failed_text",
            "没有记录到主人姓名，测试结束。",
        )
        self.owner_skip_text = rospy.get_param(
            "~owner_skip_text",
            "已跳过第%s位主人的记录。",
        )
        self.first_owner_skip_text = rospy.get_param(
            "~first_owner_skip_text",
            "第一位主人不能跳过，请说姓名。",
        )
        self.asr_not_ready_text = rospy.get_param(
            "~asr_not_ready_text",
            "语音识别没有连接，请检查离线语音节点。",
        )
        self.name_recorded_text = rospy.get_param(
            "~name_recorded_text",
            "已记录主人姓名，%s。接下来开始近距离人脸登记。",
        )
        self.front_recording_text = rospy.get_param(
            "~front_recording_text",
            "请退后站到相机前方，正面面对相机，开始记录正面外观特征。",
        )
        self.front_to_side_recording_text = rospy.get_param(
            "~front_to_side_recording_text",
            "请从正面慢慢转向侧面。",
        )
        self.side_recording_text = rospy.get_param(
            "~side_recording_text",
            "请侧身面对相机，开始记录侧身特征。",
        )
        self.front_to_side_record_seconds = max(
            self.record_seconds,
            float(rospy.get_param("~front_to_side_record_seconds", 4.5)),
        )
        self.front_to_side_record_sample_count = max(
            self.record_min_samples,
            int(rospy.get_param("~front_to_side_record_sample_count", 10)),
        )
        self.front_to_side_record_sample_interval = max(
            0.05,
            float(rospy.get_param("~front_to_side_record_sample_interval", 0.35)),
        )
        self.owner_record_poses = [
            ("front", "正面", self.front_recording_text),
            ("front_to_side", "正面转侧面", self.front_to_side_recording_text),
            ("side", "侧身", self.side_recording_text),
        ]
        self.all_profiles_recorded_text = rospy.get_param(
            "~all_profiles_recorded_text",
            "主人信息已全部记录完成",
        )
        self.named_owner_found_text = rospy.get_param(
            "~named_owner_found_text",
            "识别到主人，%s。",
        )

        self.owner_name = ""
        self.latest_name_answer = None
        self.accepting_name_answer = False
        self.name_condition = threading.Condition()

        self.result_pub = rospy.Publisher(
            rospy.get_param("~result_topic", "/owner_voice_reid_test/result"),
            String,
            queue_size=5,
            latch=True,
        )
        self.yolo_pause_pub = rospy.Publisher(
            self.yolo_pause_topic,
            Bool,
            queue_size=1,
            latch=True,
        )
        self.asr_sub = rospy.Subscriber(self.asr_topic, String, self.asr_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.on_shutdown(self.shutdown_action_recognition)
        rospy.on_shutdown(self.close_yolo_window)

    def wait_for_asr(self):
        if not self.asr_wait_for_publishers:
            return True
        deadline = time.time() + self.asr_wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            try:
                if self.asr_sub.get_num_connections() > 0:
                    self.publish_status("asr_ready", topic=self.asr_topic)
                    return True
            except AttributeError:
                return True
            rospy.sleep(0.1)
        self.publish_status("asr_not_ready", topic=self.asr_topic, timeout=self.asr_wait_timeout)
        rospy.logerr("No ASR publisher connected on %s", self.asr_topic)
        self.speak(self.asr_not_ready_text, wait=True)
        return False

    def asr_callback(self, message):
        answer = message.data.strip()
        if not answer:
            return
        rospy.loginfo("ASR: %s", answer)
        with self.name_condition:
            if not self.accepting_name_answer:
                return
            # Preserve a spoken skip command if ASR later hears the robot prompt.
            if self.is_skip_owner_answer(self.latest_name_answer or ""):
                return
            self.latest_name_answer = answer
            self.name_condition.notify_all()

    def action_result_callback(self, message):
        try:
            result = json.loads(message.data)
        except (TypeError, ValueError) as exc:
            rospy.logwarn("Invalid Qwen action result: %s", exc)
            return
        if not isinstance(result, dict):
            rospy.logwarn("Ignoring non-object Qwen action result")
            return
        self.action_result = result
        self.action_result_event.set()

    @staticmethod
    def roslaunch_bool(value):
        return "true" if bool(value) else "false"

    @staticmethod
    def action_owner_roi(owner_result):
        candidate = owner_result.get("candidate", {}) if owner_result else {}
        bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
        if not bbox or len(bbox) < 4:
            return ""
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        except (TypeError, ValueError):
            return ""
        if x2 <= x1 or y2 <= y1:
            return ""
        return "%d,%d,%d,%d" % (x1, y1, x2, y2)

    def start_action_recognition(self, owner_result=None):
        if not self.action_recognition_enabled:
            self.publish_status("owner_action_disabled")
            return False
        if self.action_backend == "yolo_pose":
            action_script = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "offline_voice_bridge", "scripts", "yolo_pose_action_recognition.py")
            )
        else:
            action_script = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "offline_voice_bridge", "scripts", "qwen_action_recognition.py")
            )
        if not os.path.isfile(action_script):
            rospy.logwarn("Action script not found: %s", action_script)
            self.publish_status("owner_action_unavailable", message="action script not found")
            return False

        self.shutdown_action_recognition()
        self.action_result = None
        self.action_result_event.clear()
        self.action_result_sub = rospy.Subscriber(
            self.action_result_topic,
            String,
            self.action_result_callback,
            queue_size=1,
        )

        try:
            owner_roi = self.action_owner_roi(owner_result)
            if self.action_backend == "yolo_pose":
                command = [
                    sys.executable, action_script,
                    "__name:=%s" % self.action_node_name,
                    "_image_topic:=%s" % self.image_topic,
                    "_result_topic:=%s" % self.action_result_topic,
                    "_model_path:=%s" % self.action_pose_model_path,
                    "_device:=%s" % self.action_pose_device,
                    "_image_size:=%d" % self.action_pose_image_size,
                    "_confidence:=%.3f" % self.action_pose_confidence,
                    "_iou:=%.3f" % self.action_pose_iou,
                    "_capture_seconds:=%.3f" % self.action_capture_duration,
                    "_sample_count:=%d" % self.action_capture_frame_count,
                    "_owner_roi:=%s" % owner_roi,
                    "_roi_padding:=%.3f" % self.action_roi_padding,
                ]
            else:
                command = [
                    sys.executable, action_script,
                    "__name:=%s" % self.action_node_name,
                    "_image_topic:=%s" % self.image_topic,
                    "_result_topic:=%s" % self.action_result_topic,
                    "_llm_url:=%s" % self.action_llm_url,
                    "_llm_model:=%s" % self.action_llm_model,
                    "_llm_timeout:=%.3f" % self.action_llm_timeout,
                    "_llm_keep_alive:=%s" % self.action_llm_keep_alive,
                    "_llm_max_tokens:=%d" % self.action_llm_max_tokens,
                    "_capture_delay:=%.3f" % self.action_capture_delay,
                    "_capture_duration:=%.3f" % self.action_capture_duration,
                    "_capture_frame_count:=%d" % self.action_capture_frame_count,
                    "_llm_frame_count:=%d" % self.action_llm_frame_count,
                    "_jpeg_quality:=%d" % self.action_jpeg_quality,
                    "_image_max_width:=%d" % self.action_image_max_width,
                    "_owner_roi:=%s" % owner_roi,
                    "_roi_padding:=%.3f" % self.action_roi_padding,
                ]
            self.action_process = subprocess.Popen(command)
            connect_deadline = time.time() + self.action_result_connect_timeout
            while (
                not rospy.is_shutdown()
                and time.time() < connect_deadline
                and self.action_result_sub.get_num_connections() == 0
            ):
                rospy.sleep(0.05)
            if self.action_result_sub.get_num_connections() == 0:
                rospy.logwarn(
                    "Qwen action result topic has no publisher yet: %s",
                    self.action_result_topic,
                )
            rospy.loginfo(
                "Started %s action node: name=%s result_topic=%s owner_roi=%s",
                self.action_backend,
                self.action_node_name,
                self.action_result_topic,
                owner_roi or "disabled",
            )
            self.publish_status(
                "owner_action_started",
                node=self.action_node_name,
                result_topic=self.action_result_topic,
            )
            return True
        except Exception as exc:
            rospy.logwarn("Failed to start Qwen action node: %s", exc)
            self.publish_status("owner_action_unavailable", message=str(exc))
            self.shutdown_action_recognition()
            return False

    def set_yolo_paused(self, paused):
        if not self.action_pause_yolo or self.yolo_pause_pub is None:
            return False
        try:
            self.yolo_pause_pub.publish(Bool(data=bool(paused)))
            rospy.loginfo("YOLO-World %s for owner action recognition", "paused" if paused else "resumed")
            return True
        except Exception as exc:
            rospy.logwarn("Failed to set YOLO-World pause=%s: %s", paused, exc)
            return False

    def shutdown_action_recognition(self):
        with self.action_launch_lock:
            launch_parent = self.action_launch_parent
            self.action_launch_parent = None
            action_process = self.action_process
            self.action_process = None
            result_sub = self.action_result_sub
            self.action_result_sub = None

        if result_sub is not None:
            try:
                result_sub.unregister()
            except Exception:
                pass
        if launch_parent is not None:
            try:
                launch_parent.shutdown()
                rospy.loginfo("Stopped Qwen action node")
            except Exception as exc:
                rospy.logwarn("Failed to stop Qwen action node: %s", exc)
        if action_process is not None:
            try:
                if action_process.poll() is None:
                    action_process.terminate()
                    action_process.wait(timeout=5.0)
                rospy.loginfo("Stopped Qwen action node")
            except Exception as exc:
                rospy.logwarn("Failed to stop Qwen action process: %s", exc)

    def run_owner_action_recognition(self, owner_result):
        if self.action_completed:
            return self.action_result
        self.action_completed = True
        owner_name = owner_result.get("owner_name", self.owner_name) if owner_result else self.owner_name
        yolo_paused = False
        if self.action_pause_yolo:
            yolo_paused = self.set_yolo_paused(True)
            if yolo_paused and self.action_yolo_pause_settle_seconds > 0.0:
                rospy.loginfo(
                    "Waiting %.2fs for YOLO to settle before action recognition",
                    self.action_yolo_pause_settle_seconds,
                )
                rospy.sleep(self.action_yolo_pause_settle_seconds)
        if not self.start_action_recognition(owner_result):
            if yolo_paused:
                self.set_yolo_paused(False)
            return None

        deadline = time.time() + self.action_timeout
        rate = rospy.Rate(10)
        try:
            while not rospy.is_shutdown() and time.time() < deadline:
                if self.action_result_event.is_set():
                    break
                rate.sleep()
            result = self.action_result
            if result is None:
                rospy.logwarn(
                    "Timed out waiting for Qwen action result after %.1fs",
                    self.action_timeout,
                )
                self.publish_status(
                    "owner_action_timeout",
                    owner_name=owner_name,
                    timeout=self.action_timeout,
                )
                return None

            rospy.loginfo(
                "Owner action recognized: owner=%s action=%s place=%s",
                owner_name or "unknown",
                result.get("action", "unknown"),
                result.get("place", "unknown"),
            )
            self.publish_status(
                "owner_action_finished",
                owner_index=owner_result.get("owner_index") if owner_result else None,
                owner_name=owner_name,
                action=result.get("action", "unknown"),
                place=result.get("place", "unknown"),
                speech=result.get("speech", ""),
                recognizer=result.get("recognizer", "qwen"),
                latency_sec=result.get("total_sec"),
            )
            rospy.sleep(self.action_speech_grace)
            return result
        finally:
            self.shutdown_action_recognition()
            if yolo_paused:
                self.set_yolo_paused(False)

    def odom_callback(self, message):
        orientation = message.pose.pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self.lock:
            self.latest_yaw = yaw

    def init_yolo_window(self):
        if not self.show_yolo_window or self.yolo_window_failed:
            return
        try:
            cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_NORMAL)
            self.yolo_window_ready = True
            rospy.loginfo("YOLO debug window enabled: %s", self.yolo_window_name)
        except Exception as exc:
            self.yolo_window_ready = False
            self.yolo_window_failed = True
            rospy.logwarn("Unable to create YOLO debug window: %s", exc)

    def close_yolo_window(self):
        if not self.yolo_window_ready:
            return
        try:
            cv2.destroyWindow(self.yolo_window_name)
        except Exception:
            pass
        self.yolo_window_ready = False

    def update_yolo_window(self, status_text=""):
        if not self.yolo_window_ready:
            return
        with self.lock:
            if self.latest_image is None:
                return
            image = self.latest_image.copy()
            detections = list(self.latest_detections)

        height, width = image.shape[:2]
        for detection in detections:
            label = str(getattr(detection, "class_name", "person") or "person")
            score = float(getattr(detection, "score", 0.0) or 0.0)
            x1 = max(0, min(width - 1, int(getattr(detection, "xmin", 0))))
            y1 = max(0, min(height - 1, int(getattr(detection, "ymin", 0))))
            x2 = max(0, min(width - 1, int(getattr(detection, "xmax", 0))))
            y2 = max(0, min(height - 1, int(getattr(detection, "ymax", 0))))
            if x2 <= x1 or y2 <= y1:
                continue
            color = (0, 220, 0) if label.lower() == "person" else (0, 180, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = "%s %.2f" % (label, score)
            cv2.putText(
                image,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        header = "YOLO: %d persons" % sum(
            1
            for detection in detections
            if str(getattr(detection, "class_name", "person") or "person").lower() == "person"
        )
        if status_text:
            header += " | " + status_text
        cv2.putText(
            image,
            header,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        try:
            cv2.imshow(self.yolo_window_name, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rospy.signal_shutdown("YOLO debug window closed by user")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "YOLO debug window update failed: %s", exc)
            self.close_yolo_window()

    def get_latest_yaw(self):
        with self.lock:
            return self.latest_yaw

    def stop_base(self):
        self.cmd_pub.publish(Twist())

    def load_waypoint_pose(self, waypoint_name=None):
        target_name = str(waypoint_name or self.waypoint_name).strip()
        if not os.path.exists(self.waypoint_file):
            raise RuntimeError("waypoint file not found: %s" % self.waypoint_file)
        root = ET.parse(self.waypoint_file).getroot()
        for waypoint in root.findall("Waypoint"):
            if waypoint.findtext("Name", "").strip() != target_name:
                continue
            return {
                "x": float(waypoint.findtext("Pos_x", "0")),
                "y": float(waypoint.findtext("Pos_y", "0")),
                "z": float(waypoint.findtext("Ori_z", "0")),
                "w": float(waypoint.findtext("Ori_w", "1")),
            }
        raise RuntimeError("waypoint not found: %s in %s" % (target_name, self.waypoint_file))

    def clear_move_base_costmaps(self, reason):
        if not self.clear_costmaps_before_navigation:
            return
        try:
            rospy.wait_for_service(self.clear_costmaps_service, timeout=self.clear_costmaps_timeout)
            clear_costmaps = rospy.ServiceProxy(self.clear_costmaps_service, Empty)
            clear_costmaps()
            rospy.loginfo("Cleared move_base costmaps: %s", reason)
        except Exception as exc:
            rospy.logwarn("Unable to clear move_base costmaps %s: %s", reason, exc)

    def navigate_to_waypoint(self, waypoint_name=None):
        target_name = str(waypoint_name or self.waypoint_name).strip()
        if not self.navigate_enabled:
            rospy.loginfo("Navigation disabled; assuming robot is already at %s", target_name)
            return True

        rospy.loginfo("Waiting for move_base action server")
        if not self.move_base.wait_for_server(rospy.Duration(self.move_base_server_timeout)):
            raise RuntimeError("move_base action server is not available")

        pose = self.load_waypoint_pose(target_name)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = pose["x"]
        goal.target_pose.pose.position.y = pose["y"]
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.z = pose["z"]
        goal.target_pose.pose.orientation.w = pose["w"]

        self.clear_move_base_costmaps("before navigating to %s" % target_name)
        self.move_base.send_goal(goal)
        deadline = time.time() + self.navigate_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            self.update_yolo_window("导航到%s" % target_name)
            if self.move_base.wait_for_result(rospy.Duration(0.2)):
                break

        if rospy.is_shutdown():
            self.move_base.cancel_goal()
            self.stop_base()
            return False
        if not self.move_base.wait_for_result(rospy.Duration(0.0)):
            self.move_base.cancel_goal()
            self.stop_base()
            raise RuntimeError("move_base timed out while navigating to %s" % target_name)

        state = self.move_base.get_state()
        if state != GoalStatus.SUCCEEDED:
            self.stop_base()
            raise RuntimeError("move_base failed with state %s" % state)

        self.stop_base()
        rospy.loginfo("Arrived at waypoint %s", target_name)
        return True

    @staticmethod
    def candidate_center_norm(candidate, image_width):
        bbox = candidate.get("bbox", [0, 0, image_width, 1])
        return ((float(bbox[0]) + float(bbox[2])) * 0.5) / max(1.0, float(image_width))

    def center_owner_in_camera(self, owner_result):
        if not self.center_owner_enabled:
            return True

        initial_candidate = owner_result.get("candidate", {}) if owner_result else {}
        image, _detections = self.snapshot()
        if image is None:
            return False
        image_width = image.shape[1]
        last_center = self.candidate_center_norm(initial_candidate, image_width)
        latest_candidate = dict(initial_candidate)
        deadline = time.time() + self.center_owner_timeout
        centered_count = 0
        rate = rospy.Rate(12)

        while not rospy.is_shutdown() and time.time() < deadline:
            image, detections = self.snapshot()
            if image is None:
                rate.sleep()
                continue
            candidates = self.person_candidates(image, detections)
            if candidates:
                candidate = min(
                    candidates,
                    key=lambda item: abs(self.candidate_center_norm(item, image.shape[1]) - last_center),
                )
                latest_candidate = dict(candidate)
                last_center = self.candidate_center_norm(candidate, image.shape[1])
                x_error = last_center - 0.5
                if abs(x_error) <= self.center_owner_tolerance:
                    self.stop_base()
                    centered_count += 1
                    self.update_yolo_window("主人已居中")
                    if centered_count >= 3:
                        if owner_result is not None:
                            updated_candidate = dict(owner_result.get("candidate", {}))
                            updated_candidate.update(latest_candidate)
                            owner_result["candidate"] = updated_candidate
                        rospy.loginfo(
                            "Updated action ROI after centering: bbox=%s",
                            latest_candidate.get("bbox"),
                        )
                        rospy.loginfo("Owner centered in camera: center=%.3f", last_center)
                        return True
                else:
                    centered_count = 0
                    twist = Twist()
                    twist.angular.z = max(
                        -self.center_owner_max_angular_speed,
                        min(
                            self.center_owner_max_angular_speed,
                            -self.center_owner_angular_gain * x_error,
                        ),
                    )
                    self.cmd_pub.publish(twist)
                    self.update_yolo_window("主人居中中")
            else:
                centered_count = 0
                direction = 1.0 if last_center > 0.5 else -1.0
                twist = Twist()
                twist.angular.z = -direction * self.center_owner_lost_turn_speed
                self.cmd_pub.publish(twist)
                self.update_yolo_window("重新寻找主人")
            rate.sleep()

        self.stop_base()
        if owner_result is not None and latest_candidate:
            updated_candidate = dict(owner_result.get("candidate", {}))
            updated_candidate.update(latest_candidate)
            owner_result["candidate"] = updated_candidate
        rospy.logwarn("Owner centering timed out")
        return False

    def scan_for_owner(self):
        rospy.sleep(self.scan_after_arrival_delay)
        start_yaw = self.get_latest_yaw()
        deadline = time.time() + self.scan_timeout
        last_yaw = start_yaw
        rotated_angle = 0.0
        last_eval_time = 0.0
        last_match_owner_index = None
        match_consecutive_count = 0
        step_duration = self.scan_step_angle / max(abs(self.scan_angular_speed), 1e-3)
        moving = True
        phase_deadline = time.time() + step_duration
        rate = rospy.Rate(10)

        rospy.loginfo(
            "Scanning for owner: total_angle=%.2f rad step_angle=%.2f rad "
            "angular_speed=%.2f rad/s pause=%.2fs",
            self.scan_total_angle,
            self.scan_step_angle,
            self.scan_angular_speed,
            self.scan_pause,
        )
        while not rospy.is_shutdown() and time.time() < deadline:
            now = time.time()
            if moving:
                twist = Twist()
                twist.angular.z = self.scan_angular_speed
                self.cmd_pub.publish(twist)
                self.update_yolo_window("小步旋转寻找主人")
            else:
                self.stop_base()
                self.update_yolo_window("停顿观察画面")

            current_yaw = self.get_latest_yaw()
            if start_yaw is not None and current_yaw is not None and last_yaw is not None:
                delta = math.atan2(
                    math.sin(current_yaw - last_yaw),
                    math.cos(current_yaw - last_yaw),
                )
                if abs(delta) < 0.7:
                    rotated_angle += abs(delta)
                last_yaw = current_yaw

            now = time.time()
            if now - last_eval_time >= self.match_check_interval:
                result = self.evaluate_current_frame()
                last_eval_time = now
                matched = result is not None and self.result_is_match(result, self.match_threshold)
                if matched:
                    owner_index = result.get("owner_index")
                    required_consecutive = result.get(
                        "required_consecutive",
                        self.match_required_consecutive,
                    )
                    if last_match_owner_index == owner_index:
                        match_consecutive_count += 1
                    else:
                        match_consecutive_count = 1
                    last_match_owner_index = owner_index
                    self.publish_status(
                        "scan_match_score",
                        owner_index=owner_index,
                        name=result.get("owner_name", ""),
                        score=result.get("score"),
                        identity_score=result.get("identity_score"),
                        reid_score=result.get("reid_score"),
                        face_score=result.get("face_score"),
                        owner_score_margin=result.get("owner_score_margin"),
                        consecutive=match_consecutive_count,
                        required_consecutive=required_consecutive,
                    )
                    if match_consecutive_count >= required_consecutive:
                        self.stop_base()
                        rospy.sleep(0.2)
                        self.center_owner_in_camera(result)
                        self.announce_owner_result(result)
                        self.run_owner_action_recognition(result)
                        return result
                else:
                    match_consecutive_count = 0
                    last_match_owner_index = None

            if start_yaw is not None and rotated_angle >= self.scan_total_angle:
                break

            now = time.time()
            if now >= phase_deadline:
                if moving:
                    self.stop_base()
                    moving = False
                    phase_deadline = now + self.scan_pause
                else:
                    moving = True
                    phase_deadline = now + step_duration
            rate.sleep()

        self.stop_base()
        if start_yaw is not None:
            rospy.loginfo("Owner scan completed: rotated %.2f rad", rotated_angle)
        else:
            rospy.logwarn("No odom yaw received on %s; scan ended by timeout", self.odom_topic)
        self.speak(self.owner_not_found_text, wait=True)
        self.publish_status("owner_not_found", rotated_angle=rotated_angle)
        return None

    @staticmethod
    def clean_text(text):
        return re.sub(r"\s+", "", text.strip().strip("，。！？、,.!?;；:： "))

    @staticmethod
    def clamp_value(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def normalize_embedding(embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6 or not np.isfinite(norm):
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
            rospy.loginfo("InsightFace fusion disabled")
            return

        model_dir = os.path.join(self.face_model_root, "models", self.face_model_name)
        cached_models = []
        if os.path.isdir(model_dir):
            cached_models = [name for name in os.listdir(model_dir) if name.endswith(".onnx")]
        if not cached_models and not self.face_auto_download:
            rospy.logwarn(
                "InsightFace model %s is not cached under %s. Face fusion will be disabled.",
                self.face_model_name,
                model_dir,
            )
            return

        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            rospy.logwarn("InsightFace is not available: %s. Face fusion will be disabled.", exc)
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
            self.face_model_ready = True
            rospy.loginfo(
                "InsightFace fusion ready: model=%s ctx_id=%d det_size=%d threshold=%.2f",
                self.face_model_name,
                self.face_ctx_id,
                self.face_det_size,
                self.face_accept_threshold,
            )
        except Exception as exc:
            self.face_app = None
            self.face_model_ready = False
            rospy.logwarn("Failed to initialize InsightFace fusion: %s", exc)

    def candidate_face_regions(self, crop, candidate=None):
        if crop is None or crop.size == 0:
            return []
        height, width = crop.shape[:2]
        top_ratio = self.clamp_value(float(self.face_crop_top_ratio), 0.35, 1.0)
        padding = self.clamp_value(float(self.face_crop_padding), 0.0, 0.40)

        upper_y2 = max(1, int(height * top_ratio))
        pad_x = int(width * padding)
        x1 = int(self.clamp_value(0 - pad_x, 0, width - 1))
        x2 = int(self.clamp_value(width + pad_x, x1 + 1, width))
        regions = [
            ("upper", crop[0:upper_y2, x1:x2]),
            ("full", crop),
        ]
        if self.side_face_reid_enabled:
            regions.extend(
                [
                    ("upper_flip", cv2.flip(crop[0:upper_y2, x1:x2], 1)),
                    ("full_flip", cv2.flip(crop, 1)),
                ]
            )
        return [(name, image) for name, image in regions if image is not None and image.size > 0]

    def face_information_quality(self, face, face_count, ambiguity_ratio=None):
        reasons = []
        bbox = getattr(face, "bbox", None)
        min_size = 0.0
        if bbox is not None and len(bbox) >= 4:
            min_size = min(
                max(0.0, float(bbox[2]) - float(bbox[0])),
                max(0.0, float(bbox[3]) - float(bbox[1])),
            )
        det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        if not math.isfinite(min_size) or min_size < self.side_face_min_face_size:
            reasons.append("small_face")
        if not math.isfinite(det_score) or det_score < self.side_face_min_det_score:
            reasons.append("low_detection_confidence")
        if face_count > 1 and (
            ambiguity_ratio is None or ambiguity_ratio > 0.45
        ):
            reasons.append("multiple_faces")
        nose_offset = None
        keypoints = getattr(face, "kps", None)
        if keypoints is not None and len(keypoints) >= 3:
            eye_delta_x = float(keypoints[1][0]) - float(keypoints[0][0])
            eye_delta_y = float(keypoints[1][1]) - float(keypoints[0][1])
            eye_distance_squared = eye_delta_x ** 2 + eye_delta_y ** 2
            if math.isfinite(eye_distance_squared) and eye_distance_squared > 1e-6:
                nose_delta_x = float(keypoints[2][0]) - (
                    float(keypoints[0][0]) + float(keypoints[1][0])
                ) * 0.5
                nose_delta_y = float(keypoints[2][1]) - (
                    float(keypoints[0][1]) + float(keypoints[1][1])
                ) * 0.5
                offset = abs(
                    nose_delta_x * eye_delta_x + nose_delta_y * eye_delta_y
                ) / eye_distance_squared
                if math.isfinite(offset):
                    nose_offset = offset
        if nose_offset is None:
            reasons.append("unreliable_landmarks")
        elif nose_offset >= self.side_face_nose_offset_threshold:
            reasons.append("side_face")
        return {
            "limited": bool(reasons),
            "reasons": reasons,
            "nose_offset": nose_offset,
            "min_size": min_size if math.isfinite(min_size) else None,
            "det_score": det_score if math.isfinite(det_score) else None,
        }

    def extract_face_embedding(self, crop, candidate=None):
        if not self.face_model_ready or self.face_app is None:
            return None, 0.0, "face unavailable"

        best_embedding = None
        best_region_embeddings = []
        best_area = 0.0
        best_region = ""
        best_face_count = 0
        # Retained for the local close-face registration gate.
        best_bbox = None
        best_det_score = 0.0
        elapsed_ms = 0.0
        face_quality = None
        detection_succeeded = False
        for region_name, face_image in self.candidate_face_regions(crop, candidate=candidate):
            try:
                start = time.time()
                faces = self.face_app.get(face_image)
                detection_succeeded = True
                elapsed_ms += (time.time() - start) * 1000.0
            except Exception as exc:
                rospy.logwarn("InsightFace failed on %s crop: %s", region_name, exc)
                continue

            face = self.select_largest_face(faces)
            if face is None:
                continue
            embedding = self.normalize_embedding(face.embedding)
            if embedding is None:
                continue
            region_quality = None
            if self.side_face_reid_enabled:
                face_areas = []
                for detected_face in faces:
                    bbox = getattr(detected_face, "bbox", None)
                    if bbox is None or len(bbox) < 4:
                        continue
                    face_width = max(0.0, float(bbox[2]) - float(bbox[0]))
                    face_height = max(0.0, float(bbox[3]) - float(bbox[1]))
                    face_areas.append(face_width * face_height)
                face_areas.sort(reverse=True)
                ambiguity_ratio = None
                if len(face_areas) >= 2 and face_areas[0] > 1e-6:
                    ambiguity_ratio = face_areas[1] / face_areas[0]
                region_quality = self.face_information_quality(
                    face,
                    len(faces),
                    ambiguity_ratio=ambiguity_ratio,
                )
            bbox = getattr(face, "bbox", None)
            if bbox is not None and len(bbox) >= 4:
                area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                    0.0,
                    float(bbox[3]) - float(bbox[1]),
                )
            else:
                area = 1.0
            if area > best_area:
                best_embedding = embedding
                best_region_embeddings = [embedding]
                best_area = area
                best_region = region_name
                best_face_count = len(faces) if faces is not None else 1
                best_bbox = (
                    [float(value) for value in bbox[:4]]
                    if bbox is not None and len(bbox) >= 4
                    else None
                )
                best_det_score = float(getattr(face, "det_score", 0.0) or 0.0)
                face_quality = region_quality

        if best_embedding is None:
            reason = "no face" if detection_succeeded else "face detection failed"
            return None, elapsed_ms, reason
        return {
            "embedding": best_embedding,
            "embeddings": best_region_embeddings,
            "region": best_region,
            "face_count": best_face_count,
            "bbox": best_bbox,
            "det_score": best_det_score,
            "area": best_area,
            "quality": face_quality,
        }, elapsed_ms, "face ready"

    def parse_owner_name(self, answer):
        text = self.clean_text(answer)
        match = re.match(r"^(?:主人)?姓名(?:是|叫|为|:|：)?(.+)$", text)
        if not match:
            return ""
        name = match.group(1).strip("，。！？、,.!?;；:： ")
        name = re.split(r"[，。！？、,.!?;；:：]", name, maxsplit=1)[0]
        name = re.sub(r"^(?:是|叫|为)", "", name).strip()
        name = self.clean_text(name)
        if not name:
            return ""
        return name[: self.max_name_length]

    def is_skip_owner_answer(self, answer):
        return self.clean_text(answer) in ("跳过", "跳過")

    def speak_name_retry_if_due(self):
        now = time.monotonic()
        elapsed = now - self._last_name_retry_prompt_at
        if elapsed < self.name_retry_prompt_interval:
            rospy.loginfo(
                "Suppressing repeated name reminder: %.1fs remaining",
                self.name_retry_prompt_interval - elapsed,
            )
            return
        self._last_name_retry_prompt_at = now
        self.speak(self.name_retry_text, wait=True)

    @staticmethod
    def format_with_name(template, name):
        if "%s" in template:
            return template % name
        if "{name}" in template:
            return template.format(name=name)
        return "%s%s" % (template, name)

    @staticmethod
    def owner_index_label(owner_index):
        labels = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        if 1 <= owner_index <= len(labels):
            return labels[owner_index - 1]
        return str(owner_index)

    def format_owner_index_text(self, template, owner_index):
        label = self.owner_index_label(owner_index)
        if "%s" in template:
            return template % label
        if "%d" in template:
            return template % owner_index
        if "{index}" in template or "{label}" in template:
            return template.format(index=owner_index, label=label)
        return template

    def owner_profile_paths(self, owner_index):
        profile_root, profile_ext = os.path.splitext(self.base_profile_path)
        metadata_root, metadata_ext = os.path.splitext(self.base_metadata_path)
        if self.owner_count == 1:
            return self.base_profile_path, self.base_metadata_path
        return (
            "%s_%d%s" % (profile_root, owner_index, profile_ext or ".npz"),
            "%s_%d%s" % (metadata_root, owner_index, metadata_ext or ".json"),
        )

    def select_owner_profile_path(self, owner_index):
        self.current_owner_index = owner_index
        self.profile_path, self.metadata_path = self.owner_profile_paths(owner_index)

    def wait_for_owner_name(self, owner_index=None):
        with self.name_condition:
            self.latest_name_answer = None
            self.accepting_name_answer = False

        if owner_index is None:
            prompt_text = self.name_prompt_text
        else:
            prompt_text = self.format_owner_index_text(self.indexed_name_prompt_text, owner_index)
        candidate_name = ""
        candidate_raw = ""
        self.speak(prompt_text, wait=True)

        while not rospy.is_shutdown():
            with self.name_condition:
                self.latest_name_answer = None
                self.accepting_name_answer = True
                while self.latest_name_answer is None and not rospy.is_shutdown():
                    self.name_condition.wait(timeout=0.2)
                    waiting_for = "确认姓名" if candidate_name else "姓名"
                    self.update_yolo_window(
                        "等待第%s位主人%s" % (self.owner_index_label(self.current_owner_index), waiting_for)
                    )
                raw_answer = self.latest_name_answer or ""
                self.latest_name_answer = None
                self.accepting_name_answer = False

            if not raw_answer:
                continue
            if self.is_skip_owner_answer(raw_answer):
                self.skipped_owner_indices.append(self.current_owner_index)
                payload = {
                    "event": "owner_skipped",
                    "owner_index": self.current_owner_index,
                    "raw": raw_answer,
                }
                self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                self.publish_status("owner_skipped", owner_index=self.current_owner_index, raw=raw_answer)
                self.speak(self.format_owner_index_text(self.owner_skip_text, self.current_owner_index), wait=True)
                rospy.loginfo("Owner %d enrollment skipped by voice command", self.current_owner_index)
                return None

            if candidate_name:
                normalized = self.clean_text(raw_answer)
                if normalized in ("正确", "正确的", "是", "是的", "对", "对的", "正確", "正確的"):
                    self.owner_name = candidate_name
                    payload = {
                        "event": "owner_name_recorded",
                        "owner_index": self.current_owner_index,
                        "name": candidate_name,
                        "raw": candidate_raw,
                        "confirmation": raw_answer,
                    }
                    self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                    self.publish_status(
                        "owner_name_recorded", owner_index=self.current_owner_index,
                        name=candidate_name, raw=candidate_raw, confirmation=raw_answer,
                    )
                    self.speak(self.format_with_name(self.name_recorded_text, candidate_name), wait=True)
                    rospy.loginfo("Owner %d name confirmed: %s", self.current_owner_index, candidate_name)
                    return candidate_name
                corrected_name = self.parse_owner_name(raw_answer)
                if corrected_name:
                    candidate_name, candidate_raw = corrected_name, raw_answer
                elif normalized in ("错误", "不对", "不正确", "不是", "重说", "重新说"):
                    candidate_name = ""
                    candidate_raw = ""
                    self.speak_name_retry_if_due()
                    continue
                else:
                    self.publish_status(
                        "owner_name_confirmation_ignored", owner_index=self.current_owner_index,
                        raw=raw_answer, pending_name=candidate_name,
                    )
                    # A speaker echo usually contains the whole question. Keep
                    # listening instead of replaying it or closing the window.
                    continue
            else:
                candidate_name = self.parse_owner_name(raw_answer)
                candidate_raw = raw_answer if candidate_name else ""
                if not candidate_name:
                    self.publish_status("owner_name_ignored", raw=raw_answer)
                    self.speak_name_retry_if_due()
                    continue

            self.speak(self.format_with_name(self.name_confirm_prompt_text, candidate_name), wait=True)

        self.speak(self.name_failed_text, wait=True)
        return ""

    def save_crop(self, crop, directory, prefix, index):
        path = super().save_crop(crop, directory, prefix, index)
        if prefix.startswith("owner") and self.face_model_ready:
            face_data, elapsed_ms, reason = self.extract_face_embedding(crop)
            if face_data is None:
                rospy.logwarn(
                    "No usable face embedding for owner %d %s sample %d: %s elapsed=%.1fms",
                    self.current_owner_index,
                    self.current_record_pose_name or prefix,
                    index,
                    reason,
                    elapsed_ms,
                )
                return path
            self.current_record_face_embeddings.append(
                {
                    "sample": index,
                    "pose": self.current_record_pose_id,
                    "pose_name": self.current_record_pose_name,
                    "image": path,
                    "region": face_data.get("region", ""),
                    "face_count": face_data.get("face_count", 0),
                    "stage": "reid",
                    "embedding": face_data["embedding"],
                }
            )
            self.publish_status(
                "face_recording_sample",
                owner_index=self.current_owner_index,
                sample=index,
                pose=self.current_record_pose_id,
                pose_name=self.current_record_pose_name,
                region=face_data.get("region", ""),
                faces=face_data.get("face_count", 0),
                elapsed_ms=elapsed_ms,
            )
        return path

    def record_owner_pose(
        self,
        pose_id,
        pose_name,
        prompt_text,
        sample_offset,
        duration=None,
        sample_count=None,
        sample_interval=None,
    ):
        self.current_record_pose_id = pose_id
        self.current_record_pose_name = pose_name
        self.publish_status(
            "recording_pose_started",
            owner_index=self.current_owner_index,
            name=self.owner_name,
            pose=pose_id,
            pose_name=pose_name,
        )
        self.speak(prompt_text, wait=True)
        self.play_ding()

        embeddings = []
        color_embeddings = []
        sample_meta = []
        sample_dir = self.profile_sample_dir()
        capture_duration = self.record_seconds if duration is None else max(0.5, float(duration))
        target_sample_count = self.record_sample_count if sample_count is None else max(1, int(sample_count))
        capture_interval = (
            self.record_sample_interval
            if sample_interval is None
            else max(0.05, float(sample_interval))
        )
        deadline = time.time() + capture_duration
        next_sample_time = 0.0
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and time.time() < deadline and len(embeddings) < target_sample_count:
            now = time.time()
            if now < next_sample_time:
                self.update_yolo_window("登记%s" % pose_name)
                rate.sleep()
                continue
            next_sample_time = now + capture_interval
            image, detections = self.snapshot()
            if image is None:
                rospy.logwarn_throttle(1.0, "Waiting for fresh camera image on %s", self.image_topic)
                self.update_yolo_window("等待相机")
                rate.sleep()
                continue
            candidates = self.person_candidates(image, detections)
            if not candidates:
                rospy.logwarn_throttle(1.0, "Waiting for person detection on %s", self.detections_topic)
                self.update_yolo_window("等待YOLO人体框")
                rate.sleep()
                continue
            crop, crop_bbox = self.crop_candidate(image, candidates[0])
            if crop is None:
                rate.sleep()
                continue
            extracted = self.extract_embeddings([crop])
            if not extracted:
                rate.sleep()
                continue
            pose_sample_index = len(embeddings) + 1
            sample_index = sample_offset + pose_sample_index
            path = self.save_crop(crop, sample_dir, "owner_%s" % pose_id, sample_index)
            color_embedding = self.color_hist_embedding(crop)
            if color_embedding is not None:
                color_embeddings.append(color_embedding)
            embeddings.append(extracted[0])
            sample_meta.append(
                {
                    "index": sample_index,
                    "pose": pose_id,
                    "pose_name": pose_name,
                    "pose_sample": pose_sample_index,
                    "image": path,
                    "bbox": candidates[0]["bbox"],
                    "crop_bbox": crop_bbox,
                    "det_score": candidates[0]["score"],
                    "area_ratio": candidates[0]["area_ratio"],
                }
            )
            self.publish_status(
                "recording_sample",
                owner_index=self.current_owner_index,
                sample=sample_index,
                pose=pose_id,
                pose_name=pose_name,
                pose_sample=pose_sample_index,
                required=self.record_min_samples,
                target=target_sample_count,
            )
            rospy.loginfo(
                "Owner %d %s Re-ID sample %d captured: %s",
                self.current_owner_index,
                pose_name,
                pose_sample_index,
                path or "not saved",
            )
            self.update_yolo_window("登记%s" % pose_name)
            rate.sleep()

        self.current_record_pose_id = ""
        self.current_record_pose_name = ""
        if len(embeddings) < self.record_min_samples:
            self.publish_status(
                "recording_pose_failed",
                owner_index=self.current_owner_index,
                pose=pose_id,
                pose_name=pose_name,
                samples=len(embeddings),
                required=self.record_min_samples,
            )
            self.speak(self.record_failed_text, wait=True)
            raise RuntimeError(
                "only captured %d/%d usable %s Re-ID samples"
                % (len(embeddings), self.record_min_samples, pose_name)
            )

        self.publish_status(
            "recording_pose_done",
            owner_index=self.current_owner_index,
            pose=pose_id,
            pose_name=pose_name,
            samples=len(embeddings),
        )
        return embeddings, color_embeddings, sample_meta

    def record_close_face_pose(self, pose_id, pose_name, prompt_text):
        """Capture clear face embeddings before the owner steps back for Re-ID."""
        self.publish_status(
            "close_face_pose_started",
            owner_index=self.current_owner_index,
            name=self.owner_name,
            pose=pose_id,
            pose_name=pose_name,
        )
        self.speak(prompt_text, wait=True)
        self.play_ding()
        sample_dir = self.profile_sample_dir()
        captured = 0
        deadline = time.time() + self.face_close_record_seconds
        next_sample_time = 0.0
        rate = rospy.Rate(30)
        while (not rospy.is_shutdown() and time.time() < deadline and
               captured < self.face_close_sample_count):
            now = time.time()
            if now < next_sample_time:
                self.update_yolo_window("近距人脸登记%s" % pose_name)
                rate.sleep()
                continue
            next_sample_time = now + self.face_close_sample_interval
            image, _detections = self.snapshot()
            if image is None:
                rospy.logwarn_throttle(1.0, "Waiting for camera image for close face enrollment")
                rate.sleep()
                continue
            face_data, elapsed_ms, reason = self.extract_face_embedding(image)
            bbox = face_data.get("bbox") if face_data is not None else None
            face_width = 0.0
            face_height = 0.0
            if bbox is not None and len(bbox) >= 4:
                face_width = max(0.0, float(bbox[2]) - float(bbox[0]))
                face_height = max(0.0, float(bbox[3]) - float(bbox[1]))
            if (face_data is None or min(face_width, face_height) < self.face_close_min_size):
                rospy.logwarn_throttle(
                    1.0,
                    "Close face sample rejected for owner %d %s: %s size=%.0fx%.0f min=%d elapsed=%.1fms",
                    self.current_owner_index, pose_name, reason, face_width, face_height,
                    self.face_close_min_size, elapsed_ms,
                )
                self.update_yolo_window("请靠近相机，脸部清晰可见")
                rate.sleep()
                continue
            sample_id = "face_close_%s_%02d" % (pose_id, captured + 1)
            path = super().save_crop(image, sample_dir, sample_id, captured + 1)
            self.current_record_face_embeddings.append(
                {
                    "sample": sample_id,
                    "pose": "face_close_%s" % pose_id,
                    "pose_name": "近距人脸%s" % pose_name,
                    "stage": "close_face",
                    "image": path,
                    "region": face_data.get("region", ""),
                    "face_count": face_data.get("face_count", 0),
                    "face_bbox": bbox,
                    "face_size": [round(face_width, 1), round(face_height, 1)],
                    "det_score": face_data.get("det_score", 0.0),
                    "embedding": face_data["embedding"],
                }
            )
            captured += 1
            self.publish_status(
                "close_face_sample",
                owner_index=self.current_owner_index,
                pose=pose_id,
                pose_name=pose_name,
                sample=captured,
                target=self.face_close_sample_count,
                face_size=[round(face_width, 1), round(face_height, 1)],
                det_score=face_data.get("det_score", 0.0),
                elapsed_ms=elapsed_ms,
            )
            rospy.loginfo(
                "Owner %d close face %s sample %d captured: %s size=%.0fx%.0f",
                self.current_owner_index, pose_name, captured, path or "not saved",
                face_width, face_height,
            )
            self.update_yolo_window("近距人脸登记%s" % pose_name)
            rate.sleep()
        self.publish_status(
            "close_face_pose_done",
            owner_index=self.current_owner_index,
            pose=pose_id,
            pose_name=pose_name,
            samples=captured,
        )
        return captured

    def record_close_face_enrollment(self):
        if not self.face_close_registration_enabled:
            return 0
        if not self.face_model_ready:
            rospy.logwarn("Close face enrollment skipped because InsightFace is unavailable")
            self.publish_status("close_face_skipped", owner_index=self.current_owner_index, reason="face unavailable")
            return 0
        self.publish_status(
            "close_face_started",
            owner_index=self.current_owner_index,
            name=self.owner_name,
            poses=[pose_name for _, pose_name, _ in self.face_close_poses],
        )
        self.speak(self.face_close_start_text, wait=True)
        total = 0
        for pose_id, pose_name, prompt_text in self.face_close_poses:
            total += self.record_close_face_pose(pose_id, pose_name, prompt_text)
        if total >= self.face_close_min_samples:
            self.speak(self.face_close_done_text, wait=True)
            self.publish_status(
                "close_face_done", owner_index=self.current_owner_index,
                samples=total, required=self.face_close_min_samples,
            )
        else:
            self.speak(self.face_close_fallback_text, wait=True)
            self.publish_status(
                "close_face_insufficient", owner_index=self.current_owner_index,
                samples=total, required=self.face_close_min_samples,
            )
        return total

    def record_owner(self):
        close_face_samples = self.record_close_face_enrollment()
        self.publish_status(
            "recording_started",
            close_face_samples=close_face_samples,
            owner_index=self.current_owner_index,
            name=self.owner_name,
            poses=[pose_name for _, pose_name, _ in self.owner_record_poses],
        )
        all_embeddings = []
        all_color_embeddings = []
        all_sample_meta = []
        for pose_id, pose_name, prompt_text in self.owner_record_poses:
            duration = None
            sample_count = None
            sample_interval = None
            if pose_id == "front_to_side":
                duration = self.front_to_side_record_seconds
                sample_count = self.front_to_side_record_sample_count
                sample_interval = self.front_to_side_record_sample_interval
            embeddings, color_embeddings, sample_meta = self.record_owner_pose(
                pose_id,
                pose_name,
                prompt_text,
                len(all_embeddings),
                duration=duration,
                sample_count=sample_count,
                sample_interval=sample_interval,
            )
            all_embeddings.extend(embeddings)
            all_color_embeddings.extend(color_embeddings)
            all_sample_meta.extend(sample_meta)

        self.save_owner_profile(
            all_embeddings,
            all_sample_meta,
            color_embeddings=all_color_embeddings,
        )
        self.speak(self.record_done_text, wait=True)
        self.publish_status(
            "recording_done",
            owner_index=self.current_owner_index,
            samples=len(all_embeddings),
            poses=[pose_id for pose_id, _, _ in self.owner_record_poses],
        )

    def remember_current_owner_profile(self):
        profile = {
            "index": self.current_owner_index,
            "name": self.owner_name,
            "embedding": self.owner_embedding,
            "embedding_bank": self.owner_embedding_bank,
            "color_embedding": self.owner_color_embedding,
            "face_embedding": self.owner_face_embedding,
            "face_embedding_bank": self.owner_face_embedding_bank,
            "metadata": dict(self.owner_profile_meta),
            "profile_path": self.profile_path,
            "metadata_path": self.metadata_path,
        }
        existing_index = next(
            (index for index, item in enumerate(self.owner_profiles) if item["index"] == self.current_owner_index),
            None,
        )
        if existing_index is None:
            self.owner_profiles.append(profile)
        else:
            self.owner_profiles[existing_index] = profile
        self.owner_profiles.sort(key=lambda item: item["index"])

    def record_all_owners(self):
        self.owner_profiles = []
        self.skipped_owner_indices = []
        for owner_index in range(1, self.owner_count + 1):
            if rospy.is_shutdown():
                return
            self.select_owner_profile_path(owner_index)
            self.owner_name = ""
            self.owner_embedding = None
            self.owner_embedding_bank = None
            self.owner_color_embedding = None
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.current_record_face_embeddings = []
            self.owner_profile_meta = {}
            owner_name = self.wait_for_owner_name(owner_index)
            if owner_name is None:
                continue
            if not owner_name:
                return
            self.record_owner()
            self.remember_current_owner_profile()
        if not rospy.is_shutdown():
            self.face_ready = any(profile.get("face_embedding_bank") is not None for profile in self.owner_profiles)
            self.speak(self.all_profiles_recorded_text, wait=True)
            self.publish_status(
                "all_profiles_recorded",
                count=len(self.owner_profiles),
                registered_count=len(self.owner_profiles),
                skipped_count=len(self.skipped_owner_indices),
                skipped_owner_indices=list(self.skipped_owner_indices),
                face_ready=self.face_ready,
            )

    def load_all_owner_profiles(self):
        loaded_profiles = []
        self.owner_profiles = []
        original_owner_name = self.owner_name
        for owner_index in range(1, self.owner_count + 1):
            self.select_owner_profile_path(owner_index)
            self.owner_name = ""
            self.owner_embedding = None
            self.owner_embedding_bank = None
            self.owner_color_embedding = None
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.owner_profile_meta = {}
            if not self.load_owner_profile():
                self.owner_name = original_owner_name
                self.owner_profiles = []
                return False
            self.owner_name = self.owner_profile_meta.get("owner_name", "")
            self.remember_current_owner_profile()
            loaded_profiles.append(self.owner_name or "主人%d" % owner_index)
        self.owner_profiles = sorted(self.owner_profiles, key=lambda item: item["index"])
        self.face_ready = any(profile.get("face_embedding_bank") is not None for profile in self.owner_profiles)
        self.publish_status(
            "all_profiles_loaded",
            count=len(self.owner_profiles),
            names=loaded_profiles,
            face_ready=self.face_ready,
        )
        rospy.loginfo("Loaded %d owner profiles: %s", len(self.owner_profiles), ", ".join(loaded_profiles))
        return len(self.owner_profiles) == self.owner_count

    def save_owner_profile(self, embeddings, sample_meta, color_embeddings=None):
        super().save_owner_profile(embeddings, sample_meta, color_embeddings=color_embeddings)
        face_embeddings = [item["embedding"] for item in self.current_record_face_embeddings]
        if face_embeddings:
            mean_face_embedding = self.normalize_embedding(np.mean(np.vstack(face_embeddings), axis=0))
            if mean_face_embedding is not None:
                face_bank = [mean_face_embedding]
                face_bank.extend(face_embeddings)
                self.owner_face_embedding = mean_face_embedding
                self.owner_face_embedding_bank = np.vstack(face_bank).astype(np.float32)
                with np.load(self.profile_path, allow_pickle=False) as profile_file:
                    npz_payload = {name: profile_file[name] for name in profile_file.files}
                npz_payload["face_embedding"] = self.owner_face_embedding.astype(np.float32)
                npz_payload["face_embedding_bank"] = self.owner_face_embedding_bank
                np.savez(self.profile_path, **npz_payload)
                self.owner_profile_meta["has_face_embedding"] = True
                self.owner_profile_meta["face_samples"] = [
                    {
                        "sample": item["sample"],
                        "pose": item["pose"],
                        "pose_name": item["pose_name"],
                        "image": item["image"],
                        "region": item["region"],
                        "face_count": item["face_count"],
                        "stage": item.get("stage", "reid"),
                        "face_size": item.get("face_size"),
                        "det_score": item.get("det_score"),
                    }
                    for item in self.current_record_face_embeddings
                ]
        else:
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.owner_profile_meta["has_face_embedding"] = False
            self.owner_profile_meta["face_samples"] = []
            if self.face_model_ready:
                rospy.logwarn(
                    "Owner %d profile saved without face embedding; Re-ID fallback remains enabled.",
                    self.current_owner_index,
                )
        self.owner_profile_meta["face_close_registration"] = {
            "enabled": self.face_close_registration_enabled,
            "required_samples": self.face_close_min_samples,
            "min_face_size": self.face_close_min_size,
            "captured_samples": sum(
                1 for item in self.current_record_face_embeddings
                if item.get("stage") == "close_face"
            ),
        }
        self.owner_profile_meta["record_poses"] = [
            {"pose": pose_id, "pose_name": pose_name}
            for pose_id, pose_name, _ in self.owner_record_poses
        ]
        if self.owner_name:
            self.owner_profile_meta["owner_index"] = self.current_owner_index
            self.owner_profile_meta["owner_name"] = self.owner_name
            with open(self.metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(self.owner_profile_meta, metadata_file, ensure_ascii=False, indent=2)
            self.publish_status(
                "profile_named",
                owner_index=self.current_owner_index,
                name=self.owner_name,
                profile_path=self.profile_path,
            )

    def load_owner_profile(self):
        loaded = super().load_owner_profile()
        self.owner_face_embedding = None
        self.owner_face_embedding_bank = None
        if loaded:
            try:
                with np.load(self.profile_path, allow_pickle=False) as profile_file:
                    if "face_embedding" in profile_file.files:
                        self.owner_face_embedding = self.normalize_embedding(profile_file["face_embedding"])
                    if "face_embedding_bank" in profile_file.files:
                        bank = []
                        for vector in np.asarray(profile_file["face_embedding_bank"]):
                            normalized = self.normalize_embedding(vector)
                            if normalized is not None:
                                bank.append(normalized)
                        if bank:
                            self.owner_face_embedding_bank = np.vstack(bank).astype(np.float32)
            except Exception as exc:
                rospy.logwarn("Failed to load owner face embedding from %s: %s", self.profile_path, exc)
            self.face_ready = self.owner_face_embedding_bank is not None
            if not self.owner_name:
                self.owner_name = self.owner_profile_meta.get("owner_name", "")
        return loaded

    def owner_found_message(self):
        name = self.owner_name or self.owner_profile_meta.get("owner_name", "主人")
        return self.format_with_name(self.named_owner_found_text, name)

    def owner_reid_similarity(self, owner_profile, embedding):
        embedding_bank = owner_profile.get("embedding_bank")
        if embedding_bank is not None:
            scores = np.dot(embedding_bank, embedding)
            return float(np.max(scores))
        owner_embedding = owner_profile.get("embedding")
        if owner_embedding is None:
            return -1.0
        return float(np.dot(owner_embedding, embedding))

    def owner_color_similarity(self, owner_profile, crop):
        color_embedding = owner_profile.get("color_embedding")
        if color_embedding is None:
            return None
        query_color_embedding = self.color_hist_embedding(crop)
        if query_color_embedding is None:
            return None
        return float(np.dot(color_embedding, query_color_embedding))

    def reid_query_variants(self, crop, lying_pose):
        variants = list(super().reid_query_variants(crop, lying_pose))
        if lying_pose or not self.body_query_variants_enabled:
            return variants
        height, width = crop.shape[:2]
        if height < 4 or width < 4:
            return variants
        upper_height = max(2, int(height * 0.82))
        upper_body = crop[:upper_height, :]
        variants.extend(
            [
                ("body_flip", cv2.flip(crop, 1)),
                ("upper_body", upper_body),
                ("upper_body_flip", cv2.flip(upper_body, 1)),
            ]
        )
        return variants

    def owner_face_similarity(self, owner_profile, face_embedding):
        if face_embedding is None:
            return None
        face_bank = owner_profile.get("face_embedding_bank")
        if face_bank is None:
            face_embedding_single = owner_profile.get("face_embedding")
            if face_embedding_single is None:
                return None
            if isinstance(face_embedding, (list, tuple)):
                return float(
                    max(np.dot(face_embedding_single, embedding) for embedding in face_embedding)
                )
            return float(np.dot(face_embedding_single, face_embedding))
        if isinstance(face_embedding, (list, tuple)):
            return float(
                max(
                    np.max(np.dot(face_bank, embedding))
                    for embedding in face_embedding
                )
            )
        return float(np.max(np.dot(face_bank, face_embedding)))

    def score_owner_match(
        self, owner_profile, embedding, crop, lying_pose, face_embedding=None,
        reid_fallback=False, face_quality=None,
    ):
        reid_score = self.owner_reid_similarity(owner_profile, embedding)
        if lying_pose:
            score = reid_score
            identity_score = score
        else:
            score = reid_score
        face_score = self.owner_face_similarity(owner_profile, face_embedding)
        if not lying_pose:
            if reid_fallback:
                if (
                    self.side_face_fusion_enabled
                    and face_score is not None
                    and face_quality
                ):
                    face_weight = self.side_face_fusion_weight
                    if "small_face" in face_quality.get("reasons", []):
                        face_weight *= 0.5
                    if "low_detection_confidence" in face_quality.get("reasons", []):
                        face_weight *= 0.5
                    face_weight = max(0.0, min(0.35, face_weight))
                    reid_weight = max(0.0, 1.0 - face_weight)
                    identity_score = (
                        reid_weight * reid_score
                        + face_weight * face_score
                    )
                    if face_score >= self.side_face_face_score_floor:
                        calibrated_face_score = min(
                            0.99,
                            face_score + self.side_face_face_score_offset,
                        )
                        identity_score = max(
                            identity_score,
                            calibrated_face_score,
                        )
                else:
                    identity_score = reid_score
            else:
                identity_score = face_score if face_score is not None else -1.0
        return score, reid_score, None, face_score, identity_score

    def result_is_match(self, result, default_threshold):
        if result is None:
            return False
        score_margin = result.get("owner_score_margin")
        margin_threshold = self.owner_score_margin_threshold
        reid_fallback = result.get("match_method") == "side_face_reid"
        if reid_fallback:
            margin_threshold = self.side_face_reid_margin
        if (
            len(self.owner_profiles) > 1
            and (
                (reid_fallback and score_margin is None)
                or (
                    score_margin is not None
                    and float(score_margin) < margin_threshold
                )
            )
        ):
            return False

        face_score = result.get("face_score")
        lying_pose = bool(result.get("candidate", {}).get("lying_pose", False))
        if reid_fallback and not lying_pose:
            if not self.side_face_reid_enabled or self.reid_backend != "torchreid":
                return False
            return PersonReidOwnerTest.result_is_match(
                result, self.side_face_reid_threshold
            )
        if lying_pose:
            return PersonReidOwnerTest.result_is_match(result, default_threshold)

        if self.side_face_reid_enabled and (result.get("face_quality") or {}).get("limited"):
            return False
        if face_score is None:
            return False
        return float(face_score) >= self.face_accept_threshold

    def evaluate_current_frame(self):
        if not self.owner_profiles:
            return super().evaluate_current_frame()

        image, detections = self.snapshot()
        if image is None:
            return None
        candidates = self.person_candidates(image, detections)
        if not candidates:
            return None
        query_crops = []
        records = []
        for candidate in candidates:
            crop, crop_bbox = self.crop_candidate(
                image,
                candidate,
                padding=self.crop_padding,
            )
            if crop is None:
                continue
            lying_pose = self.is_lying_candidate(candidate)
            if lying_pose and self.lying_crop_padding > self.crop_padding:
                crop, crop_bbox = self.crop_candidate(
                    image,
                    candidate,
                    padding=self.lying_crop_padding,
                )
                if crop is None:
                    continue
            meta = dict(candidate)
            meta["crop_bbox"] = crop_bbox
            meta["lying_pose"] = lying_pose
            variants = self.reid_query_variants(crop, lying_pose)
            query_indexes = []
            variant_names = []
            for variant_name, variant_crop in variants:
                query_indexes.append(len(query_crops))
                variant_names.append(variant_name)
                query_crops.append(variant_crop)
            records.append(
                {
                    "crop": crop,
                    "meta": meta,
                    "query_indexes": query_indexes,
                    "variant_names": variant_names,
                }
            )
        if not query_crops:
            return None
        embeddings = self.extract_embeddings(query_crops)
        if len(embeddings) != len(query_crops):
            return None

        results = []
        face_profiles_ready = any(profile.get("face_embedding_bank") is not None for profile in self.owner_profiles)
        for record in records:
            lying_pose = record["meta"].get("lying_pose", False)
            face_embedding = None
            face_region = ""
            face_count = 0
            face_elapsed_ms = 0.0
            face_reason = "face disabled"
            face_quality = None
            if self.face_model_ready and face_profiles_ready:
                face_data, face_elapsed_ms, face_reason = self.extract_face_embedding(
                    record["crop"],
                    candidate=record["meta"],
                )
                if face_data is not None:
                    face_embedding = face_data.get("embeddings") or face_data["embedding"]
                    face_region = face_data.get("region", "")
                    face_count = face_data.get("face_count", 0)
                    face_quality = face_data.get("quality")
                elif face_reason == "no face":
                    face_quality = {"limited": True, "reasons": ["no_face"]}
            reid_fallback = bool(
                self.side_face_reid_enabled
                and not lying_pose
                and self.face_model_ready
                and face_profiles_ready
                and self.reid_backend == "torchreid"
                and face_quality
                and face_quality.get("limited")
                and "multiple_faces" not in face_quality.get("reasons", [])
            )
            for owner_profile in self.owner_profiles:
                variant_scores = []
                for query_index, variant_name in zip(record["query_indexes"], record["variant_names"]):
                    score, reid_score, color_score, face_score, identity_score = self.score_owner_match(
                        owner_profile,
                        embeddings[query_index],
                        record["crop"],
                        lying_pose,
                        face_embedding=face_embedding,
                        reid_fallback=reid_fallback,
                        face_quality=face_quality,
                    )
                    if (
                        self.side_face_reid_enabled
                        and not lying_pose
                        and not reid_fallback
                        and face_quality
                        and face_quality.get("limited")
                    ):
                        identity_score = -1.0
                    variant_scores.append(
                        (identity_score, score, reid_score, color_score, face_score, variant_name)
                    )
                identity_score, score, reid_score, color_score, face_score, best_variant = max(
                    variant_scores,
                    key=lambda item: item[0],
                )
                if lying_pose:
                    match_threshold = self.lying_match_threshold
                    min_reid_score = self.lying_min_reid_score
                    required_consecutive = self.lying_required_consecutive
                    match_method = "lying_reid"
                elif reid_fallback:
                    match_threshold = self.side_face_reid_threshold
                    min_reid_score = self.side_face_reid_threshold
                    required_consecutive = self.side_face_reid_required_consecutive
                    match_method = "side_face_reid"
                else:
                    match_threshold = self.match_threshold
                    min_reid_score = -1.0
                    required_consecutive = self.match_required_consecutive
                    match_method = "face"
                primary_score = identity_score if not lying_pose else score
                result = {
                    "score": float(primary_score),
                    "identity_score": float(identity_score),
                    "reid_score": float(reid_score),
                    "color_score": None if color_score is None else float(color_score),
                    "face_score": None if face_score is None else float(face_score),
                    "face_region": face_region,
                    "face_count": face_count,
                    "face_elapsed_ms": face_elapsed_ms,
                    "face_reason": face_reason,
                    "face_quality": face_quality,
                    "match_method": match_method,
                    "match_threshold": match_threshold,
                    "min_reid_score": min_reid_score,
                    "required_consecutive": required_consecutive,
                    "best_variant": best_variant,
                    "candidate": record["meta"],
                    "crop": record["crop"],
                    "num_candidates": len(records),
                    "owner_index": owner_profile["index"],
                    "owner_name": owner_profile.get("name") or "主人%d" % owner_profile["index"],
                }
                results.append(result)
        if not results:
            return None

        if self.side_face_reid_enabled:
            best_by_method = {}
            for match_method in ("face", "lying_reid", "side_face_reid"):
                method_results = [
                    result for result in results if result["match_method"] == match_method
                ]
                if not method_results:
                    continue
                owner_best_scores = {}
                for result in method_results:
                    owner_index = result["owner_index"]
                    score = result["identity_score"]
                    if owner_index not in owner_best_scores or score > owner_best_scores[owner_index]:
                        owner_best_scores[owner_index] = score
                ranked_owner_scores = sorted(owner_best_scores.values(), reverse=True)
                score_margin = None
                if len(ranked_owner_scores) >= 2:
                    score_margin = ranked_owner_scores[0] - ranked_owner_scores[1]
                for result in method_results:
                    result["owner_score_margin"] = score_margin
                if match_method == "side_face_reid":
                    for record in records:
                        candidate_results = [
                            result for result in method_results
                            if result["candidate"] is record["meta"]
                        ]
                        candidate_owner_scores = sorted(
                            (result["identity_score"] for result in candidate_results),
                            reverse=True,
                        )
                        if len(candidate_owner_scores) >= 2:
                            candidate_margin = candidate_owner_scores[0] - candidate_owner_scores[1]
                            for result in candidate_results:
                                result["owner_score_margin"] = min(
                                    candidate_margin,
                                    result["owner_score_margin"],
                                )
                best_by_method[match_method] = max(
                    method_results, key=lambda item: item["identity_score"]
                )
            for match_method in ("face", "lying_reid", "side_face_reid"):
                best_result = best_by_method.get(match_method)
                if best_result and self.result_is_match(best_result, self.match_threshold):
                    return best_result
            return max(results, key=lambda item: item["identity_score"])

        owner_best_scores = {}
        for result in results:
            owner_index = result["owner_index"]
            score = result["identity_score"]
            if owner_index not in owner_best_scores or score > owner_best_scores[owner_index]:
                owner_best_scores[owner_index] = score
        ranked_owner_scores = sorted(owner_best_scores.values(), reverse=True)
        score_margin = None
        if len(ranked_owner_scores) >= 2:
            score_margin = ranked_owner_scores[0] - ranked_owner_scores[1]

        best_result = max(results, key=lambda item: item["identity_score"])
        if best_result["match_method"] != "side_face_reid":
            best_result["owner_score_margin"] = score_margin
        return best_result

    def announce_owner_result(self, result):
        score = result["score"]
        owner_index = result.get("owner_index", self.current_owner_index)
        owner_name = result.get("owner_name", self.owner_name)
        crop_path = self.maybe_save_match_crop(result["crop"], score)
        self.owner_name = owner_name
        message = self.format_with_name(self.named_owner_found_text, owner_name or "主人")
        rospy.loginfo(
            "Owner recognized: index=%s name=%s score=%.3f crop=%s",
            owner_index,
            owner_name or "unknown",
            score,
            crop_path or "not saved",
        )
        self.speak(message, wait=False)
        payload = {
            "event": "owner_recognized",
            "owner_index": owner_index,
            "name": owner_name,
            "score": score,
            "identity_score": result.get("identity_score", score),
            "reid_score": result.get("reid_score", score),
            "color_score": result.get("color_score"),
            "face_score": result.get("face_score"),
            "face_region": result.get("face_region", ""),
            "face_count": result.get("face_count", 0),
            "owner_score_margin": result.get("owner_score_margin"),
            "lying_pose": result.get("candidate", {}).get("lying_pose", False),
            "best_variant": result.get("best_variant", "raw"),
            "crop": crop_path,
        }
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.publish_status(
            "owner_recognized",
            owner_index=owner_index,
            name=owner_name,
            score=score,
            identity_score=result.get("identity_score", score),
            reid_score=result.get("reid_score", score),
            color_score=result.get("color_score"),
            face_score=result.get("face_score"),
            face_region=result.get("face_region", ""),
            face_count=result.get("face_count", 0),
            owner_score_margin=result.get("owner_score_margin"),
            lying_pose=result.get("candidate", {}).get("lying_pose", False),
            best_variant=result.get("best_variant", "raw"),
            crop=crop_path,
        )
        self.last_announce_time = time.time()

    def recognition_loop(self):
        self.publish_status(
            "recognition_started",
            threshold=self.match_threshold,
            owner_count=len(self.owner_profiles) or 1,
            names=[profile.get("name", "") for profile in self.owner_profiles],
            face_model_ready=self.face_model_ready,
            face_ready=self.face_ready,
            face_weight=self.face_identity_weight,
            owner_margin_threshold=self.owner_score_margin_threshold,
        )
        rate = rospy.Rate(max(1.0, 1.0 / self.match_check_interval))
        while not rospy.is_shutdown():
            self.update_yolo_window("识别中")
            result = self.evaluate_current_frame()
            if result is None:
                self.match_consecutive_count = 0
                self.last_match_owner_index = None
                rate.sleep()
                continue

            score = result["score"]
            owner_index = result.get("owner_index", self.current_owner_index)
            owner_name = result.get("owner_name", self.owner_name)
            match_threshold = result.get("match_threshold", self.match_threshold)
            required_consecutive = result.get("required_consecutive", self.match_required_consecutive)
            matched = self.result_is_match(result, self.match_threshold)
            if matched:
                match_method = result.get("match_method")
                if (
                    self.last_match_owner_index == owner_index
                    and self.last_match_method == match_method
                ):
                    self.match_consecutive_count += 1
                else:
                    self.match_consecutive_count = 1
                self.last_match_owner_index = owner_index
                self.last_match_method = match_method
            else:
                self.match_consecutive_count = 0
                self.last_match_owner_index = None
                self.last_match_method = None
            self.publish_status(
                "match_score",
                score=score,
                identity_score=result.get("identity_score", score),
                reid_score=result.get("reid_score", score),
                color_score=result.get("color_score"),
                face_score=result.get("face_score"),
                face_region=result.get("face_region", ""),
                face_count=result.get("face_count", 0),
                face_elapsed_ms=result.get("face_elapsed_ms", 0.0),
                owner_score_margin=result.get("owner_score_margin"),
                matched=matched,
                consecutive=self.match_consecutive_count,
                threshold=match_threshold,
                required_consecutive=required_consecutive,
                lying_pose=result.get("candidate", {}).get("lying_pose", False),
                best_variant=result.get("best_variant", "raw"),
                owner_index=owner_index,
                name=owner_name,
            )

            if matched and self.match_consecutive_count >= required_consecutive:
                now = time.time()
                if now - self.last_announce_time >= self.announce_cooldown:
                    crop_path = self.maybe_save_match_crop(result["crop"], score)
                    self.owner_name = owner_name
                    message = self.format_with_name(self.named_owner_found_text, owner_name or "主人")
                    rospy.loginfo(
                        "Owner recognized: index=%s name=%s score=%.3f crop=%s",
                        owner_index,
                        owner_name or "unknown",
                        score,
                        crop_path or "not saved",
                    )
                    self.speak(message, wait=False)
                    payload = {
                        "event": "owner_recognized",
                        "owner_index": owner_index,
                        "name": owner_name,
                        "score": score,
                        "identity_score": result.get("identity_score", score),
                        "reid_score": result.get("reid_score", score),
                        "color_score": result.get("color_score"),
                        "face_score": result.get("face_score"),
                        "face_region": result.get("face_region", ""),
                        "face_count": result.get("face_count", 0),
                        "owner_score_margin": result.get("owner_score_margin"),
                        "lying_pose": result.get("candidate", {}).get("lying_pose", False),
                        "best_variant": result.get("best_variant", "raw"),
                        "crop": crop_path,
                    }
                    self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                    self.publish_status(
                        "owner_recognized",
                        owner_index=owner_index,
                        name=owner_name,
                        score=score,
                        identity_score=result.get("identity_score", score),
                        reid_score=result.get("reid_score", score),
                        color_score=result.get("color_score"),
                        face_score=result.get("face_score"),
                        face_region=result.get("face_region", ""),
                        face_count=result.get("face_count", 0),
                        owner_score_margin=result.get("owner_score_margin"),
                        lying_pose=result.get("candidate", {}).get("lying_pose", False),
                        best_variant=result.get("best_variant", "raw"),
                        crop=crop_path,
                    )
                    self.last_announce_time = now
                    if not self.action_completed:
                        self.run_owner_action_recognition(result)
                    if self.stop_after_first_match:
                        return
            rate.sleep()

    def run(self):
        self.wait_for_tts()
        if not self.wait_for_asr():
            return
        self.wait_for_camera_inputs()
        self.init_yolo_window()
        self.init_reid_backend()
        self.init_face_recognizer()
        if self.reuse_existing_profile and self.load_all_owner_profiles():
            rospy.loginfo("Loaded existing owner Re-ID profiles from %s", self.profile_dir)
        else:
            self.record_all_owners()
        if not self.owner_profiles:
            if self.owner_embedding is not None:
                self.remember_current_owner_profile()
            else:
                raise RuntimeError("owner profiles are not ready")
        if self.navigate_enabled:
            try:
                self.navigate_to_waypoint(self.waypoint_name)
            except Exception as exc:
                self.stop_base()
                self.speak(self.navigation_failed_text, wait=True)
                self.publish_status("navigation_failed", message=str(exc), waypoint=self.waypoint_name)
                raise
            self.scan_for_owner()
        else:
            self.recognition_loop()


def main():
    rospy.init_node("owner_voice_reid_test")
    node = OwnerVoiceReidTest()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("Owner voice Re-ID test failed: %s", exc)
        node.publish_status(
            "error",
            message=str(exc),
            owner_index=getattr(node, "current_owner_index", 0),
            name=getattr(node, "owner_name", ""),
        )
        raise


if __name__ == "__main__":
    main()
