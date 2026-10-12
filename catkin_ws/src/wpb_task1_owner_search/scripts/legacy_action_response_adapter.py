#!/usr/bin/env python3.9
# coding: utf-8
"""Safe bridge from the Re-ID/YOLO-Pose workflow to the legacy action response.

The legacy task controller owns the tested Kinect point-cloud, lidar-guard and
move_base approach code.  This adapter deliberately leaves that controller
unchanged and supplies it with the owner box confirmed by the Re-ID workflow.
"""

import json
import math
import time
import urllib.error
import urllib.request

import rospy

from task1_find_owner_real import RealOwnerSearchBeforeAction


class LegacyActionResponseAdapter(RealOwnerSearchBeforeAction):
    """Use a Re-ID-confirmed image box to run the legacy action response."""

    # The formal task loads gallery images because it performs identity
    # recognition itself. Identity has already been verified by Re-ID before
    # this adapter is created, so no duplicate gallery or model is needed.
    def load_owner_images(self):
        return []

    def init_face_recognizer(self):
        # Re-ID already validated identity. Mark the legacy identity gate ready
        # without loading a second face model or touching its configuration.
        self.face_ready = True
        self.face_app = None

    def init_action_recognizer(self):
        # YOLO-Pose has already supplied the action label.
        self.action_model = None

    def start_electrical_switch_preload(self):
        # Parent construction defaults to direct_asr. Defer its preload until
        # this adapter has explicitly selected the shared topic ASR below.
        return None

    def __init__(self):
        super().__init__()
        self.external_owner_bbox_norm = None
        # Reuse the workflow ASR topic, then classify the transcript with the
        # launch-configured local Qwen model (qwen3.5:0.8b by default).
        self.electrical_switch_instruction_source = "topic"
        self.waving_help_dialogue_enabled = bool(
            rospy.get_param("~waving_help_dialogue_enabled", True)
        )
        self.waving_help_listen_seconds = max(
            2.0, float(rospy.get_param("~waving_help_listen_seconds", 8.0))
        )
        self.waving_help_echo_guard_seconds = max(
            0.0, float(rospy.get_param("~waving_help_echo_guard_seconds", 0.8))
        )
        self.waving_help_repeat_prefix = str(
            rospy.get_param("~waving_help_repeat_prefix", "我理解您的需求是：")
        ).strip()
        self.waving_help_no_input_text = str(
            rospy.get_param("~waving_help_no_input_text", "抱歉，我没有听清您的需求。")
        ).strip()
        self.waving_help_max_repeat_chars = max(
            12, int(rospy.get_param("~waving_help_max_repeat_chars", 80))
        )
        self.team_waving_two_stage_enabled = bool(
            rospy.get_param("~team_waving_two_stage_enabled", True)
        )
        self.team_waving_coarse_standoff_distance = max(
            self.waving_approach_safety_radius + 0.20,
            float(rospy.get_param("~team_waving_coarse_standoff_distance", 1.00)),
        )
        self.team_waving_refine_pause_seconds = max(
            0.0, float(rospy.get_param("~team_waving_refine_pause_seconds", 0.25))
        )
        # Keep the two mandatory stages aligned with the team's optimized
        # workflow: help dialogue is entered only after precise approach.
        self.team_waving_precise_stage_enabled = bool(
            rospy.get_param("~team_waving_precise_stage_enabled", True)
        )
        self.team_waving_coarse_dialogue_fallback_enabled = bool(
            rospy.get_param("~team_waving_coarse_dialogue_fallback_enabled", True)
        )
        self.team_waving_standing_standoff_distance = max(
            self.waving_approach_safety_radius,
            min(0.50, float(rospy.get_param("~standing_waving_standoff_distance", 0.48))),
        )
        self.team_waving_seated_standoff_distance = max(
            self.waving_approach_safety_radius + 0.05,
            float(rospy.get_param("~seated_waving_standoff_distance", 0.75)),
        )
        # Now that the source is topic, the inherited preload warms Qwen only.
        super().start_electrical_switch_preload()
        rospy.loginfo(
            "Electrical-switch dialogue: source=%s qwen_enabled=%s model=%s num_gpu=%s",
            self.electrical_switch_instruction_source,
            self.electrical_switch_ollama_enabled,
            self.electrical_switch_ollama_model,
            self.electrical_switch_ollama_num_gpu,
        )

    def is_waving_help_prompt_echo(self, text):
        """Return true only for ASR text that is clearly our own help prompt."""
        normalized = "".join(ch for ch in str(text or "") if ch.isalnum())
        prompt = "".join(ch for ch in self.approach_help_prompt if ch.isalnum())
        if not normalized or not prompt:
            return False
        return normalized in prompt or prompt in normalized

    def collect_waving_help_transcript(self, sequence_before_prompt):
        """Wait for a real reply while ignoring delayed speaker echo."""
        if self.waving_help_echo_guard_seconds > 0.0:
            rospy.sleep(self.waving_help_echo_guard_seconds)
        self.play_electrical_switch_ready_ding()
        deadline = time.time() + self.waving_help_listen_seconds
        sequence = int(sequence_before_prompt)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                entries = [entry for entry in self.asr_history if entry["sequence"] > sequence]
            for entry in entries:
                sequence = max(sequence, int(entry["sequence"]))
                text = str(entry.get("text", "")).strip()
                if not text:
                    continue
                if self.is_waving_help_prompt_echo(text):
                    rospy.loginfo("Ignoring waving-help TTS echo: %s", text)
                    continue
                rospy.loginfo("Waving-help received owner transcript: %s", text)
                return text
            rate.sleep()
        return ""

    @staticmethod
    def normalize_waving_help_repeat(text):
        """Compare wording while tolerating only whitespace and punctuation."""
        return "".join(ch for ch in str(text or "") if ch.isalnum())

    def summarize_waving_help_request(self, transcript):
        """Ask Qwen to repeat the ASR request without any semantic rewrite.

        Qwen output is accepted only when its words are identical to the ASR
        input (aside from punctuation/whitespace); otherwise the exact ASR
        text is spoken. This makes the model a constrained formatter rather
        than an intent interpreter.
        """
        raw_text = str(transcript or "").strip()
        if not raw_text:
            return "", "empty"
        if not self.should_try_electrical_switch_ollama():
            return raw_text, "asr_fallback"
        prompt = (
            "/no_think\n"
            "你是逐字复述器。repeat 必须与‘主人原话’使用完全相同的中文词语；"
            "仅可添加或删除标点和空格。绝不能推理、判断状态、改写动词、补充礼貌语、"
            "承诺执行或解释。\n"
            "示例：主人原话=把房间的灯打开，输出={\"repeat\":\"把房间的灯打开\"}。\n"
            "禁止输出={\"repeat\":\"房间里的灯是开着的\"}。\n"
            "只输出 JSON，不要解释：{\"repeat\":\"...\"}。\n"
            "主人原话：%s" % raw_text
        )
        payload = {
            "model": self.electrical_switch_ollama_model,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": self.electrical_switch_ollama_keep_alive,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0,
                "top_p": 0.7,
                "num_predict": 64,
                "num_ctx": 1024,
                "num_gpu": max(0, self.electrical_switch_ollama_num_gpu),
            },
        }
        request = urllib.request.Request(
            self.electrical_switch_ollama_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=max(1.0, self.electrical_switch_ollama_timeout)
            ) as response:
                raw_response = response.read().decode("utf-8")
            content = json.loads(raw_response).get("message", {}).get("content", "").strip()
            try:
                repeat_text = str(json.loads(content).get("repeat", "")).strip()
            except (TypeError, ValueError):
                repeat_text = ""
            if not repeat_text or len(repeat_text) > self.waving_help_max_repeat_chars:
                rospy.logwarn("Invalid Qwen help repeat; using raw ASR text: %s", content)
                return raw_text, "asr_fallback"
            if self.normalize_waving_help_repeat(repeat_text) != self.normalize_waving_help_repeat(raw_text):
                rospy.logwarn(
                    "Qwen altered help wording; rejecting rewrite: raw=%s qwen=%s",
                    raw_text,
                    repeat_text,
                )
                return raw_text, "asr_verbatim_guard"
            self.electrical_switch_ollama_available = True
            return repeat_text, "qwen_verbatim"
        except (OSError, ValueError, TypeError, urllib.error.URLError) as exc:
            self.electrical_switch_ollama_available = False
            self.electrical_switch_ollama_last_failure = time.time()
            rospy.logwarn("Qwen help repeat failed; using raw ASR text: %s", exc)
            return raw_text, "asr_fallback"

    def run_waving_help_dialogue(self):
        if not self.waving_help_dialogue_enabled:
            return {"status": "disabled", "raw_text": "", "repeat_text": ""}
        # Establish the ASR boundary before speaking: an echo recognized late
        # by Whisper must still be visible to, and rejected by, the listener.
        with self.lock:
            sequence_before_prompt = self.asr_sequence
        self.say(self.approach_help_prompt, hold=self.waving_help_pause_seconds)
        transcript = self.collect_waving_help_transcript(sequence_before_prompt)
        if not transcript:
            self.say(self.waving_help_no_input_text)
            rospy.logwarn("Waving-help dialogue received no ASR transcript")
            return {"status": "no_input", "raw_text": "", "repeat_text": ""}
        repeat_text, source = self.summarize_waving_help_request(transcript)
        spoken = "%s%s" % (self.waving_help_repeat_prefix, repeat_text)
        self.say(spoken)
        rospy.loginfo(
            "Waving-help dialogue completed: source=%s raw=%s repeat=%s",
            source, transcript, repeat_text,
        )
        return {"status": "completed", "source": source, "raw_text": transcript, "repeat_text": repeat_text}

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        area_a = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
        area_b = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
        return inter / max(1.0, area_a + area_b - inter)

    def set_external_owner_context(self, context):
        bbox = (context or {}).get("bbox")
        image_width = float((context or {}).get("image_width", 0) or 0)
        image_height = float((context or {}).get("image_height", 0) or 0)
        if not bbox or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
            self.external_owner_bbox_norm = None
            return
        self.external_owner_bbox_norm = [
            max(0.0, min(1.0, float(bbox[0]) / image_width)),
            max(0.0, min(1.0, float(bbox[1]) / image_height)),
            max(0.0, min(1.0, float(bbox[2]) / image_width)),
            max(0.0, min(1.0, float(bbox[3]) / image_height)),
        ]
        self.owner_track_center = (self.external_owner_bbox_norm[0] + self.external_owner_bbox_norm[2]) * 0.5

    def select_tracking_detection(self):
        if self.external_owner_bbox_norm is None:
            return super().select_tracking_detection()
        with self.lock:
            if self.latest_image is None or not self.latest_detections:
                return None, None, None
            height, width = self.latest_image.shape[:2]
            detections = list(self.latest_detections)
        target = [
            self.external_owner_bbox_norm[0] * width,
            self.external_owner_bbox_norm[1] * height,
            self.external_owner_bbox_norm[2] * width,
            self.external_owner_bbox_norm[3] * height,
        ]
        best, best_iou = None, 0.0
        for det in detections:
            if getattr(det, "class_name", "") not in ("", "person"):
                continue
            if float(getattr(det, "score", 0.0) or 0.0) < self.detection_min_score:
                continue
            candidate = [float(det.xmin), float(det.ymin), float(det.xmax), float(det.ymax)]
            iou = self._bbox_iou(target, candidate)
            if iou > best_iou:
                best, best_iou = det, iou
        if best is not None:
            return best, width, height
        # The person may have moved after recognition. The legacy tracker then
        # falls back to its centre-biased selection instead of moving blindly.
        return super().select_tracking_detection()

    def resolve_waving_owner_position(self, owner_candidate=None):
        """Get one fresh, Re-ID-associated Kinect position for an approach stage."""
        deadline = time.time() + max(1.0, self.approach_timeout)
        rate = rospy.Rate(10)
        fallback_used = False
        self.last_approach_failure_reason = "no valid owner position before approach"
        while not rospy.is_shutdown() and time.time() < deadline:
            det, width, height = self.select_tracking_detection()
            if (det is None or width is None or height is None) and owner_candidate is not None and not fallback_used:
                det = owner_candidate.get("det")
                width = owner_candidate.get("image_width")
                height = owner_candidate.get("image_height")
                fallback_used = det is not None
            if det is None or not width or not height:
                self.last_approach_failure_reason = "no owner detection before approach"
                rate.sleep()
                continue
            self.owner_track_center = float(det.center_x) / float(width)
            position = self.estimate_owner_position_from_pointcloud(det, width, height)
            if position is not None:
                return position
            self.last_approach_failure_reason = self.last_pointcloud_reason or "no valid owner point-cloud position"
            rate.sleep()
        return None

    def approach_waving_owner_two_stage(self, owner_candidate=None, action_result=None):
        """Use the team's two-stage, multi-candidate waving approach."""
        if not self.team_waving_two_stage_enabled:
            return self.approach_waving_owner(owner_candidate)

        # The team implementation does not force a single map goal on the
        # robot-to-owner line. Both stages evaluate a front-facing fan of
        # standoff positions. This avoids oscillation when the owner's front
        # position is occupied in the local costmap.
        team_candidate_angles = [0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0]
        original_angles = list(self.waving_approach_candidate_angles_deg)
        original_distances = list(self.waving_approach_candidate_distances)
        original_reactive_fallback = self.waving_reactive_fallback_enabled
        try:
            self.waving_approach_candidate_angles_deg = team_candidate_angles
            # Each stage supplies its exact clearance. Do not mix distances
            # from the legacy profile while evaluating team candidate goals.
            self.waving_approach_candidate_distances = []
            self.waving_reactive_fallback_enabled = False
            rospy.loginfo(
                "Team-style waving approach stage 1/2: candidate coarse standoff=%.2fm angles=%s",
                self.team_waving_coarse_standoff_distance,
                team_candidate_angles,
            )
            coarse_ok = self.approach_waving_owner(
                owner_candidate,
                target_distance=self.team_waving_coarse_standoff_distance,
                label="waving owner coarse candidate approach",
            )
        finally:
            self.waving_approach_candidate_angles_deg = original_angles
            self.waving_approach_candidate_distances = original_distances
            self.waving_reactive_fallback_enabled = original_reactive_fallback

        if not coarse_ok:
            self.stop_base()
            return False

        self.stop_base()
        if not self.team_waving_precise_stage_enabled:
            return True

        if self.team_waving_refine_pause_seconds > 0.0:
            rospy.sleep(self.team_waving_refine_pause_seconds)
        det, width, height = self.select_tracking_detection()
        refined_candidate = owner_candidate
        if det is not None and width and height:
            refined_candidate = {"det": det, "image_width": width, "image_height": height}
            rospy.loginfo("Team-style waving approach stage 2/2: refreshed person box")
        else:
            rospy.logwarn("Team-style waving approach stage 2/2: no refreshed box; using confirmed owner box")

        posture = str((action_result or {}).get("posture", "")).strip().lower()
        precise_standoff_distance = (
            self.team_waving_standing_standoff_distance
            if posture == "standing"
            else self.team_waving_seated_standoff_distance
        )

        # Same team candidate fan for the precise approach. Existing lidar
        # guard remains active and unguarded reactive movement stays disabled.
        original_angles = list(self.waving_approach_candidate_angles_deg)
        original_distances = list(self.waving_approach_candidate_distances)
        original_reactive_fallback = self.waving_reactive_fallback_enabled
        try:
            self.waving_approach_candidate_angles_deg = team_candidate_angles
            self.waving_approach_candidate_distances = []
            self.waving_reactive_fallback_enabled = False
            rospy.loginfo(
                "Team-style waving approach stage 2/2: team candidate precise standoff=%.2fm angles=%s",
                precise_standoff_distance,
                team_candidate_angles,
            )
            precise_ok = self.approach_waving_owner(
                refined_candidate,
                target_distance=precise_standoff_distance,
                label="waving owner precise team-candidate approach",
            )
            if precise_ok or not self.team_waving_coarse_dialogue_fallback_enabled:
                return precise_ok
            rospy.logwarn(
                "Waving precise approach failed after a successful coarse approach; "
                "using the verified coarse conversational position for help dialogue: %s",
                self.last_approach_failure_reason,
            )
            self.stop_base()
            return True
        finally:
            self.waving_approach_candidate_angles_deg = original_angles
            self.waving_approach_candidate_distances = original_distances
            self.waving_reactive_fallback_enabled = original_reactive_fallback

    def handle_pose_action(self, action_result, owner_context=None):
        action = str((action_result or {}).get("action", "unknown") or "unknown").strip().lower()
        self.set_external_owner_context(owner_context or {})
        self.stop_base()
        det, width, height = self.select_tracking_detection()
        owner_candidate = None
        if det is not None and width and height:
            owner_candidate = {"det": det, "image_width": width, "image_height": height}

        self.say(self.action_to_speech(action), hold=self.action_speech_hold)
        response = {"handled": True, "action": action, "approached": False, "detail": "reported"}
        if action in self.already_fallen_surface_labels:
            action, surface_reason, initial_position = self.classify_static_lying_surface(owner_candidate)
            response["action"] = action
            response["surface_reason"] = surface_reason
        else:
            initial_position = None

        if action in self.fall_approach_action_labels:
            approached = self.approach_fallen_owner(owner_candidate, initial_position=initial_position)
            response["approached"] = bool(approached)
            response["detail"] = self.last_approach_failure_reason if not approached else "approached"
            if action in self.fall_assist_arm_action_labels:
                # Keep the team's fall-response contract: a recognised fall
                # always triggers the assist-arm sequence after the approach
                # attempt, even if move_base could not finish the approach.
                if not approached:
                    rospy.logwarn(
                        "Proceeding to fall assist after incomplete fall approach: %s",
                        self.last_approach_failure_reason,
                    )
                self.perform_fall_assist_arm_motion()
                response["detail"] = (
                    "fall_assist_completed"
                    if approached
                    else "fall_assist_after_incomplete_approach"
                )
            elif action in ("lying", "sitting"):
                self.wait_for_electrical_switch_instruction()
                response["detail"] = "switch_interaction_completed"
        elif action == "waving":
            approached = self.approach_waving_owner_two_stage(
                owner_candidate, action_result=action_result
            )
            response["approached"] = bool(approached)
            response["detail"] = self.last_approach_failure_reason if not approached else "approached"
            if approached:
                dialogue = self.run_waving_help_dialogue()
                response["help_dialogue"] = dialogue
                response["detail"] = "help_dialogue_%s" % dialogue.get("status", "unknown")
            else:
                self.say("我看到您在挥手，但还没有成功靠近您。")
        self.stop_base()
        return response
