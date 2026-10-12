#!/usr/bin/env python3.9
# coding=utf-8

import json
import os
import sys
import threading
import time
from enum import Enum

import rospy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from owner_voice_reid_test import OwnerVoiceReidTest
from legacy_action_response_adapter import LegacyActionResponseAdapter


class WorkflowState(Enum):
    INITIALIZE = "initialize"
    WAIT_FOR_ASR = "wait_for_asr"
    WAIT_FOR_CAMERA = "wait_for_camera"
    PREPARE_MODELS = "prepare_models"
    REGISTER_OWNERS = "register_owners"
    NAVIGATE = "navigate"
    SEARCH = "search"
    OWNER_CONFIRMED = "owner_confirmed"
    RECOGNIZE_ACTION = "recognize_action"
    INTERACT = "interact"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizedOwnerVoiceReid(OwnerVoiceReidTest):
    """Workflow wrapper that keeps the original owner-search implementation intact."""

    def __init__(self):
        super().__init__()

        self.workflow_state_topic = str(
            rospy.get_param(
                "~optimized_state_topic",
                "/owner_voice_reid_optimized/state",
            )
        ).strip()
        self.workflow_status_topic = str(
            rospy.get_param(
                "~optimized_status_topic",
                "/owner_voice_reid_optimized/status",
            )
        ).strip()
        self.patrol_enabled = bool(rospy.get_param("~patrol_enabled", True))
        self.patrol_repeat = bool(rospy.get_param("~patrol_repeat", False))
        self.patrol_max_rounds = max(1, int(rospy.get_param("~patrol_max_rounds", 1)))
        self.patrol_waypoint_names = self.parse_waypoint_names(
            rospy.get_param(
                "~patrol_waypoint_names",
                ["living_room", "kitchen", "bedroom", "canteen"],
            )
        )
        if not self.patrol_waypoint_names:
            self.patrol_waypoint_names = [self.waypoint_name]
        self.exit_waypoint_name = str(
            rospy.get_param("~exit_waypoint_name", "exit")
        ).strip()
        self.return_to_exit_when_complete = bool(
            rospy.get_param("~return_to_exit_when_complete", False)
        )
        self.stop_patrol_on_owner_found = bool(
            rospy.get_param("~stop_patrol_on_owner_found", True)
        )
        self.action_followup_enabled = bool(
            rospy.get_param("~action_followup_enabled", True)
        )
        self.legacy_action_response_enabled = bool(
            rospy.get_param("~legacy_action_response_enabled", False)
        )
        self.legacy_action_response = None
        if self.legacy_action_response_enabled:
            try:
                self.legacy_action_response = LegacyActionResponseAdapter()
                rospy.loginfo("Legacy safe action-response adapter is ready")
            except Exception as exc:
                rospy.logerr("Legacy action-response adapter unavailable: %s", exc)
        self.waving_followup_text = str(
            rospy.get_param("~waving_followup_text", "我已看到您的示意，将继续执行搜索任务。")
        ).strip()
        self.static_action_followup_text = str(
            rospy.get_param("~static_action_followup_text", "我已记录主人的状态，将继续执行搜索任务。")
        ).strip()
        if not self.exit_waypoint_name:
            self.exit_waypoint_name = "exit"

        self.workflow_timeouts = {
            WorkflowState.INITIALIZE: 30.0,
            WorkflowState.WAIT_FOR_ASR: max(10.0, self.asr_wait_timeout + 5.0),
            WorkflowState.WAIT_FOR_CAMERA: max(10.0, self.startup_timeout + 5.0),
            WorkflowState.PREPARE_MODELS: 120.0,
            WorkflowState.REGISTER_OWNERS: max(120.0, self.owner_count * 180.0),
            WorkflowState.NAVIGATE: max(30.0, self.navigate_timeout + 10.0),
            WorkflowState.SEARCH: max(20.0, self.scan_timeout + 10.0),
            WorkflowState.OWNER_CONFIRMED: 20.0,
            WorkflowState.RECOGNIZE_ACTION: max(30.0, self.action_timeout + 10.0),
            WorkflowState.INTERACT: max(30.0, self.answer_timeout + 30.0),
            WorkflowState.COMPLETED: 30.0,
            WorkflowState.FAILED: 30.0,
        }
        configured_timeout = rospy.get_param("~optimized_state_timeout", 0.0)
        if float(configured_timeout) > 0.0:
            for state in self.workflow_timeouts:
                self.workflow_timeouts[state] = float(configured_timeout)

        self.workflow_state = WorkflowState.INITIALIZE
        self.workflow_state_started_at = time.time()
        self.workflow_state_timeout_reported = False
        self.workflow_lock = threading.RLock()
        self.workflow_state_pub = rospy.Publisher(
            self.workflow_status_topic,
            String,
            queue_size=5,
            latch=True,
        )
        self.workflow_event_pub = rospy.Publisher(
            self.workflow_state_topic,
            String,
            queue_size=10,
            latch=True,
        )
        self.workflow_watchdog = rospy.Timer(
            rospy.Duration(1.0),
            self.workflow_watchdog_callback,
        )
        self.current_patrol_waypoint = ""
        self.current_action_result = None
        self.patrol_results = []
        rospy.on_shutdown(self.stop_workflow_watchdog)

    @staticmethod
    def parse_waypoint_names(value):
        if isinstance(value, str):
            raw_names = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_names = value
        else:
            raw_names = [value]
        names = []
        for raw_name in raw_names:
            name = str(raw_name or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def stop_workflow_watchdog(self):
        timer = getattr(self, "workflow_watchdog", None)
        if timer is not None:
            timer.shutdown()

    def workflow_watchdog_callback(self, _event):
        with self.workflow_lock:
            state = self.workflow_state
            elapsed = time.time() - self.workflow_state_started_at
            timeout = self.workflow_timeouts.get(state, 0.0)
            if timeout <= 0.0 or elapsed <= timeout or self.workflow_state_timeout_reported:
                return
            self.workflow_state_timeout_reported = True
        rospy.logwarn(
            "Optimized owner workflow state timed out: state=%s elapsed=%.1fs timeout=%.1fs",
            state.value,
            elapsed,
            timeout,
        )
        self.publish_workflow_state(
            "state_timeout",
            state=state.value,
            elapsed_sec=round(elapsed, 3),
            timeout_sec=timeout,
        )

    def publish_workflow_state(self, event, **fields):
        with self.workflow_lock:
            state = self.workflow_state.value
            started_at = self.workflow_state_started_at
        payload = {
            "event": event,
            "state": state,
            "time": time.time(),
            "state_elapsed_sec": round(max(0.0, time.time() - started_at), 3),
        }
        if self.current_patrol_waypoint and "waypoint" not in fields:
            fields["waypoint"] = self.current_patrol_waypoint
        payload.update(fields)
        message = String(data=json.dumps(payload, ensure_ascii=False))
        self.workflow_event_pub.publish(message)
        self.workflow_state_pub.publish(message)

    def set_workflow_state(self, state, reason="", **fields):
        if not isinstance(state, WorkflowState):
            state = WorkflowState(str(state))
        with self.workflow_lock:
            previous = self.workflow_state.value
            self.workflow_state = state
            self.workflow_state_started_at = time.time()
            self.workflow_state_timeout_reported = False
        rospy.loginfo(
            "Optimized owner workflow: %s -> %s%s",
            previous,
            state.value,
            " (%s)" % reason if reason else "",
        )
        self.publish_workflow_state(
            "state_changed",
            previous_state=previous,
            reason=reason,
            **fields
        )
        super().publish_status(
            "workflow_state_changed",
            state=state.value,
            previous_state=previous,
            reason=reason,
            **fields
        )

    def navigate_to_waypoint(self, waypoint_name=None):
        self.set_workflow_state(
            WorkflowState.NAVIGATE,
            reason="navigate_to_waypoint",
            waypoint=waypoint_name or self.waypoint_name,
        )
        return super().navigate_to_waypoint(waypoint_name)

    def scan_for_owner(self):
        self.action_completed = False
        self.action_result = None
        self.action_result_event.clear()
        self.current_action_result = None
        self.set_workflow_state(WorkflowState.SEARCH, reason="scan_for_owner")
        result = super().scan_for_owner()
        return result

    def announce_owner_result(self, result):
        self.set_workflow_state(
            WorkflowState.OWNER_CONFIRMED,
            reason="owner_match_confirmed",
            owner_index=(result or {}).get("owner_index"),
            owner_name=(result or {}).get("owner_name", ""),
        )
        return super().announce_owner_result(result)

    def run_owner_action_recognition(self, owner_result):
        self.set_workflow_state(
            WorkflowState.RECOGNIZE_ACTION,
            reason="owner_confirmed",
            owner_index=(owner_result or {}).get("owner_index"),
        )
        result = super().run_owner_action_recognition(owner_result)
        self.current_action_result = self.normalize_action_result(result)
        if self.action_followup_enabled:
            self.handle_owner_action_interaction(self.current_action_result, owner_result)
        return self.current_action_result

    @staticmethod
    def normalize_action_result(action_result):
        """Map Qwen output to the same action contract used by the upstream flow."""
        if not isinstance(action_result, dict):
            return action_result
        normalized = dict(action_result)
        action = str(normalized.get("action", "unknown") or "unknown").strip().lower()
        place = str(normalized.get("place", "unknown") or "unknown").strip().lower()
        if action in {"fallen", "sudden_fall"}:
            action = "falling"
            place = "floor"
        elif action == "lying" and place == "floor":
            action = "lying_ground"
        elif action not in {"waving", "sitting", "lying", "lying_ground", "falling"}:
            action = "unknown"
        normalized["action"] = action
        normalized["place"] = place
        normalized["action_priority"] = "fall" if action in {"falling", "lying_ground"} else action
        return normalized

    @classmethod
    def action_result_is_certain(cls, action_result):
        if not isinstance(action_result, dict):
            return False
        recognizer = str(action_result.get("recognizer", "") or "").strip().lower()
        action = str(action_result.get("action", "unknown") or "unknown").strip().lower()
        supported_actions = {
            "waving",
            "sitting",
            "lying",
            "fallen",
            "sudden_fall",
            "falling",
            "lying_ground",
        }
        if recognizer == "yolo_pose":
            return action in supported_actions
        if recognizer == "qwen_vision":
            try:
                confidence = float(action_result.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            return action in supported_actions and confidence >= 0.75
        return False

    def build_action_owner_context(self, owner_result):
        candidate = (owner_result or {}).get("candidate") or {}
        bbox = candidate.get("bbox")
        image, _detections = self.snapshot()
        if image is None or not bbox:
            return {}
        height, width = image.shape[:2]
        return {"bbox": bbox, "image_width": width, "image_height": height}

    def handle_owner_action_interaction(self, action_result, owner_result):
        normalized = self.normalize_action_result(action_result)
        if not self.action_result_is_certain(normalized):
            self.publish_workflow_state("interaction_skipped", reason="action_uncertain")
            self.publish_status("owner_interaction_skipped", reason="action_uncertain")
            return False
        action = normalized.get("action", "unknown")
        place = normalized.get("place", "unknown")
        self.set_workflow_state(WorkflowState.INTERACT, reason="action_result_received", action=action, place=place)
        self.stop_base()
        if self.legacy_action_response is not None:
            try:
                outcome = self.legacy_action_response.handle_pose_action(
                    normalized, self.build_action_owner_context(owner_result)
                )
                self.publish_status("owner_action_interaction_finished", action=action, place=place,
                                    recognizer=normalized.get("recognizer", "unknown"), outcome=outcome)
                self.publish_workflow_state("interaction_finished", action=action, place=place,
                                            outcome=outcome, followup="continue_patrol")
                return bool(outcome.get("handled", False))
            except Exception as exc:
                rospy.logerr("Legacy action response failed safely: %s", exc)
                self.stop_base()
        speech = str(normalized.get("speech", "") or "").strip()
        if not speech:
            speech = {"waving": "主人正在挥手示意。", "sitting": "主人当前坐着。",
                      "lying": "识别到主人躺下。", "lying_ground": "识别到主人摔倒。",
                      "falling": "识别到主人摔倒。"}.get(action, "我已识别到主人，但动作不确定。")
        self.speak(speech, wait=True)
        followup = self.waving_followup_text if action == "waving" else self.static_action_followup_text
        if followup:
            self.speak(followup, wait=True)
        self.publish_status("owner_action_interaction_finished", action=action, place=place,
                            recognizer=normalized.get("recognizer", "unknown"), followup="continue_patrol")
        self.publish_workflow_state("interaction_finished", action=action, place=place, followup="continue_patrol")
        return True

    def reset_patrol_waypoint_state(self, waypoint_name):
        self.current_patrol_waypoint = waypoint_name
        self.current_action_result = None
        self.action_completed = False
        self.action_result = None
        self.action_result_event.clear()

    def run_patrol(self):
        rounds = self.patrol_max_rounds if self.patrol_repeat else 1
        self.patrol_results = []
        for round_index in range(rounds):
            for waypoint_index, waypoint_name in enumerate(self.patrol_waypoint_names, start=1):
                if rospy.is_shutdown():
                    return {
                        "owner_found": any(
                            item["owner_found"] for item in self.patrol_results
                        ),
                        "rooms_completed": len(self.patrol_results),
                        "interrupted": True,
                    }
                self.reset_patrol_waypoint_state(waypoint_name)
                self.publish_workflow_state(
                    "patrol_waypoint_started",
                    round=round_index + 1,
                    waypoint_index=waypoint_index,
                    waypoint=waypoint_name,
                )
                self.navigate_to_waypoint(waypoint_name)
                result = self.scan_for_owner()
                action_result = self.current_action_result
                owner_found = result is not None
                action_certain = self.action_result_is_certain(action_result)
                if owner_found and action_certain:
                    finish_reason = "interaction_completed"
                elif owner_found:
                    finish_reason = "action_uncertain"
                else:
                    finish_reason = "owner_not_found"
                room_result = {
                    "round": round_index + 1,
                    "waypoint_index": waypoint_index,
                    "waypoint": waypoint_name,
                    "owner_found": owner_found,
                    "action_certain": action_certain,
                    "action": (action_result or {}).get("action", "unknown"),
                    "place": (action_result or {}).get("place", "unknown"),
                    "finish_reason": finish_reason,
                }
                self.patrol_results.append(room_result)
                self.publish_workflow_state(
                    "patrol_waypoint_finished",
                    round=round_index + 1,
                    waypoint_index=waypoint_index,
                    waypoint=waypoint_name,
                    owner_found=owner_found,
                    action_certain=action_certain,
                    action=room_result["action"],
                    place=room_result["place"],
                    finish_reason=finish_reason,
                )
                if owner_found and self.stop_patrol_on_owner_found:
                    rospy.loginfo("Owner found at %s; stopping patrol at current room", waypoint_name)
                    return {
                        "owner_found": True,
                        "rooms_completed": len(self.patrol_results),
                        "interrupted": False,
                    }
        return {
            "owner_found": any(item["owner_found"] for item in self.patrol_results),
            "rooms_completed": len(self.patrol_results),
            "interrupted": False,
        }

    def run_single_waypoint(self):
        self.reset_patrol_waypoint_state(self.waypoint_name)
        self.navigate_to_waypoint(self.waypoint_name)
        result = self.scan_for_owner()
        action_result = self.current_action_result
        self.patrol_results = [
            {
                "round": 1,
                "waypoint_index": 1,
                "waypoint": self.waypoint_name,
                "owner_found": result is not None,
                "action_certain": self.action_result_is_certain(action_result),
                "action": (action_result or {}).get("action", "unknown"),
                "place": (action_result or {}).get("place", "unknown"),
            }
        ]
        return {
            "owner_found": result is not None,
            "rooms_completed": 1,
            "interrupted": False,
        }

    def run(self):
        self.set_workflow_state(WorkflowState.INITIALIZE, reason="startup")
        self.wait_for_tts()

        self.set_workflow_state(WorkflowState.WAIT_FOR_ASR, reason="wait_for_asr")
        if not self.wait_for_asr():
            self.set_workflow_state(WorkflowState.FAILED, reason="asr_not_ready")
            return

        self.set_workflow_state(WorkflowState.WAIT_FOR_CAMERA, reason="wait_for_camera")
        self.wait_for_camera_inputs()
        self.init_yolo_window()
        self.update_yolo_window("相机已连接")

        self.set_workflow_state(WorkflowState.PREPARE_MODELS, reason="initialize_reid_and_face")
        self.init_reid_backend()
        self.init_face_recognizer()

        self.set_workflow_state(WorkflowState.REGISTER_OWNERS, reason="load_or_record_profiles")
        if self.reuse_existing_profile and self.load_all_owner_profiles():
            rospy.loginfo("Loaded existing owner Re-ID profiles from %s", self.profile_dir)
        else:
            self.record_all_owners()
        if not self.owner_profiles:
            if self.owner_embedding is not None:
                self.remember_current_owner_profile()
            else:
                raise RuntimeError("owner profiles are not ready")

        if self.navigate_enabled and self.patrol_enabled:
            patrol_summary = self.run_patrol()
        elif self.navigate_enabled:
            patrol_summary = self.run_single_waypoint()
        else:
            self.reset_patrol_waypoint_state(self.waypoint_name)
            result = self.scan_for_owner()
            action_result = self.current_action_result
            self.patrol_results = [
                {
                    "round": 1,
                    "waypoint_index": 1,
                    "waypoint": self.waypoint_name,
                    "owner_found": result is not None,
                    "action_certain": self.action_result_is_certain(action_result),
                    "action": (action_result or {}).get("action", "unknown"),
                    "place": (action_result or {}).get("place", "unknown"),
                }
            ]
            patrol_summary = {
                "owner_found": result is not None,
                "rooms_completed": 1,
                "interrupted": False,
            }

        exit_reached = False
        if (
            self.navigate_enabled
            and self.return_to_exit_when_complete
            and not rospy.is_shutdown()
        ):
            self.publish_workflow_state(
                "exit_navigation_started",
                exit_waypoint=self.exit_waypoint_name,
            )
            exit_reached = self.navigate_to_waypoint(self.exit_waypoint_name)

        self.set_workflow_state(
            WorkflowState.COMPLETED,
            reason="exit_reached" if exit_reached else "patrol_finished",
            owner_found=patrol_summary["owner_found"],
            rooms_completed=patrol_summary["rooms_completed"],
            exit_waypoint=self.exit_waypoint_name,
            exit_reached=exit_reached,
        )


def main():
    rospy.init_node("owner_voice_reid_optimized")
    node = OptimizedOwnerVoiceReid()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("Optimized owner voice Re-ID failed: %s", exc)
        node.set_workflow_state(WorkflowState.FAILED, reason=str(exc))
        node.publish_status("error", message=str(exc))
        raise


if __name__ == "__main__":
    main()
