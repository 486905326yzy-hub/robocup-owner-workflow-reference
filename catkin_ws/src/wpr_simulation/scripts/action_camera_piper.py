#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera action recognizer with Piper TTS output.

The script opens a local camera, speaks a start prompt, samples a short video
window, classifies the configured care actions, and speaks the final result.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Can't initialize NVML")

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_VOICE_ROOT = PROJECT_ROOT.parent / "offline_voice_bridge"
DEFAULT_PIPER_BIN = OFFLINE_VOICE_ROOT / "tools" / "piper" / "piper"
DEFAULT_PIPER_MODEL = OFFLINE_VOICE_ROOT / "models" / "piper" / "zh_CN-huayan-medium.onnx"
DEFAULT_PIPER_CONFIG = OFFLINE_VOICE_ROOT / "models" / "piper" / "zh_CN-huayan-medium.onnx.json"
DEFAULT_DET_MODEL = PROJECT_ROOT / "models" / "vision" / "yolo11n.pt"
DEFAULT_POSE_MODEL = PROJECT_ROOT / "models" / "vision" / "yolo11n-pose.pt"

FURNITURE_EN_TO_ZH = {
    "bed": "床",
    "chair": "椅子",
    "sofa": "沙发",
}
YOLO_FURNITURE_NAMES = {
    "bed": "bed",
    "chair": "chair",
    "couch": "sofa",
    "sofa": "sofa",
}

# COCO human pose keypoint indexes used by YOLO pose models.
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

SKELETON = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (RIGHT_HIP, RIGHT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]

Box = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class FurnitureBox:
    label: str
    conf: float
    box: Box


@dataclass
class FrameAnalysis:
    t: float
    frame_size: Tuple[int, int]
    person_box: Optional[Box]
    keypoints: Optional[np.ndarray]
    furniture_label: Optional[str]
    furniture_score: float
    is_lying: bool
    is_sitting: bool
    raised_hand: bool
    wrists: Dict[str, Point]


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersect_area(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def expand_box(box: Box, ratio: float, frame_w: int, frame_h: int) -> Box:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * ratio
    pad_y = (y2 - y1) * ratio
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(frame_w - 1), x2 + pad_x),
        min(float(frame_h - 1), y2 + pad_y),
    )


def point_in_box(point: Point, box: Box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def keypoint(kpts: Optional[np.ndarray], idx: int, min_conf: float = 0.25) -> Optional[Point]:
    if kpts is None or idx >= len(kpts):
        return None
    x, y, conf = kpts[idx]
    if conf < min_conf:
        return None
    return float(x), float(y)


def midpoint(a: Optional[Point], b: Optional[Point]) -> Optional[Point]:
    if a is None or b is None:
        return None
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def angle_degrees(a: Point, b: Point, c: Point) -> float:
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 180.0
    cos_value = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_value)))


def load_model(path_or_name: str) -> YOLO:
    local_path = PROJECT_ROOT / path_or_name
    return YOLO(str(local_path if local_path.exists() else path_or_name))


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
    except Exception:
        pass
    return "cpu"


def speak_with_piper(text: str, args) -> bool:
    piper_bin = Path(args.piper_bin).expanduser()
    model_path = Path(args.piper_model).expanduser()
    config_path = Path(args.piper_config).expanduser() if args.piper_config else None
    if not piper_bin.exists() or not model_path.exists():
        return False

    piper_dir = piper_bin.parent
    env = os.environ.copy()
    old_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(piper_dir) + ((":" + old_ld_path) if old_ld_path else "")
    env.setdefault("ESPEAK_DATA_PATH", str(piper_dir / "espeak-ng-data"))

    fd, wav_path = tempfile.mkstemp(prefix="action_tts_", suffix=".wav", dir="/tmp")
    os.close(fd)
    cmd = [
        str(piper_bin),
        "--model",
        str(model_path),
        "--output_file",
        wav_path,
        "--length_scale",
        str(args.length_scale),
    ]
    if config_path and config_path.exists():
        cmd.extend(["--config", str(config_path)])

    try:
        subprocess.run(
            cmd,
            input=text,
            text=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        player = shutil.which("aplay") or shutil.which("paplay")
        if player:
            subprocess.run([player, "-q", wav_path] if Path(player).name == "aplay" else [player, wav_path])
        else:
            print("[TTS wav]", wav_path)
        return True
    except Exception as exc:
        print("Piper TTS failed:", exc)
        return False
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def speak(text: str, args) -> None:
    print("[播报]", text)
    if not args.no_tts and speak_with_piper(text, args):
        return
    fallback = shutil.which("spd-say")
    if not args.no_tts and fallback:
        subprocess.run([fallback, text])


def extract_furniture(result, conf_threshold: float) -> List[FurnitureBox]:
    furniture: List[FurnitureBox] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return furniture
    names = result.names
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    for raw_box, class_id, score in zip(xyxy, cls, conf):
        name = names.get(int(class_id), str(class_id))
        label = YOLO_FURNITURE_NAMES.get(name)
        if not label or score < conf_threshold:
            continue
        furniture.append(FurnitureBox(label=label, conf=float(score), box=tuple(map(float, raw_box))))
    return furniture


def extract_person_pose(result) -> Tuple[Optional[Box], Optional[np.ndarray]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    scores = [box_area(tuple(map(float, box))) * float(c) for box, c in zip(xyxy, conf)]
    idx = int(np.argmax(scores))
    person_box = tuple(map(float, xyxy[idx]))

    kpts = None
    if result.keypoints is not None and result.keypoints.data is not None:
        data = result.keypoints.data.cpu().numpy()
        if len(data) > idx:
            kpts = data[idx]
    return person_box, kpts


def match_furniture(
    person_box: Optional[Box], furniture: List[FurnitureBox], frame_size: Tuple[int, int], kpts: Optional[np.ndarray]
) -> Tuple[Optional[str], float]:
    if person_box is None or not furniture:
        return None, 0.0

    frame_w, frame_h = frame_size
    px1, py1, px2, py2 = person_box
    person_area = max(1.0, box_area(person_box))
    lower_body_point = ((px1 + px2) / 2.0, py1 + 0.72 * (py2 - py1))
    seat_point = ((px1 + px2) / 2.0, py1 + 0.58 * (py2 - py1))
    hip_mid = midpoint(keypoint(kpts, LEFT_HIP), keypoint(kpts, RIGHT_HIP))
    knee_mid = midpoint(keypoint(kpts, LEFT_KNEE), keypoint(kpts, RIGHT_KNEE))
    support_points = [p for p in (lower_body_point, seat_point, hip_mid, knee_mid) if p is not None]

    best_label = None
    best_score = 0.0
    for item in furniture:
        fbox = item.box
        inter = intersect_area(person_box, fbox)
        overlap_person = inter / person_area
        overlap_furn = inter / max(1.0, box_area(fbox))
        expanded = expand_box(fbox, 0.30, frame_w, frame_h)
        inside_bonus = 0.7 if any(point_in_box(point, expanded) for point in support_points) else 0.0
        score = overlap_person * 2.0 + overlap_furn + inside_bonus + item.conf * 0.05
        if score > best_score:
            best_score = score
            best_label = item.label

    if best_score < 0.10:
        return None, best_score
    return best_label, best_score


def infer_posture(
    person_box: Optional[Box], kpts: Optional[np.ndarray], furniture_label: Optional[str], furniture_score: float
) -> Tuple[bool, bool]:
    if person_box is None:
        return False, False

    x1, y1, x2, y2 = person_box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect = width / height

    left_shoulder = keypoint(kpts, LEFT_SHOULDER)
    right_shoulder = keypoint(kpts, RIGHT_SHOULDER)
    left_hip = keypoint(kpts, LEFT_HIP)
    right_hip = keypoint(kpts, RIGHT_HIP)
    left_knee = keypoint(kpts, LEFT_KNEE)
    right_knee = keypoint(kpts, RIGHT_KNEE)
    left_ankle = keypoint(kpts, LEFT_ANKLE)
    right_ankle = keypoint(kpts, RIGHT_ANKLE)
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    knee_mid = midpoint(left_knee, right_knee)

    torso_horizontal = False
    torso_vertical = False
    if shoulder_mid and hip_mid:
        dx = abs(shoulder_mid[0] - hip_mid[0])
        dy = abs(shoulder_mid[1] - hip_mid[1])
        torso_horizontal = dx > dy * 1.15
        torso_vertical = dy > dx * 1.05 and hip_mid[1] > shoulder_mid[1]

    is_lying = aspect > 1.25 or (torso_horizontal and aspect > 0.85)

    bent_knees = 0
    for hip, knee, ankle in ((left_hip, left_knee, left_ankle), (right_hip, right_knee, right_ankle)):
        if hip and knee and ankle and angle_degrees(hip, knee, ankle) < 145.0:
            bent_knees += 1

    knee_count = sum(1 for knee in (left_knee, right_knee) if knee is not None)
    ankle_count = sum(1 for ankle in (left_ankle, right_ankle) if ankle is not None)
    upright_torso = bool(shoulder_mid and hip_mid and hip_mid[1] > shoulder_mid[1] + height * 0.08)
    lower_body_truncated = False
    if knee_mid:
        lower_body_truncated = (y2 - knee_mid[1]) < height * 0.22 or ankle_count == 0

    compact_body = (height / width) < 2.65
    supported_by_furniture = furniture_label is not None and furniture_score >= 0.12
    seated_pose = (bent_knees > 0) or (upright_torso and knee_count > 0 and lower_body_truncated)
    is_sitting = bool(
        not is_lying
        and (
            (supported_by_furniture and (upright_torso or compact_body or knee_count > 0))
            or seated_pose
        )
    )

    return is_lying, is_sitting


def detect_raised_hand(kpts: Optional[np.ndarray], frame_h: int) -> Tuple[bool, Dict[str, Point]]:
    wrists: Dict[str, Point] = {}
    raised = False
    for side, wrist_idx, shoulder_idx in (
        ("left", LEFT_WRIST, LEFT_SHOULDER),
        ("right", RIGHT_WRIST, RIGHT_SHOULDER),
    ):
        wrist = keypoint(kpts, wrist_idx)
        shoulder = keypoint(kpts, shoulder_idx)
        if wrist is None or shoulder is None:
            continue
        wrists[side] = wrist
        if wrist[1] < shoulder[1] - frame_h * 0.03:
            raised = True
    return raised, wrists


def analyze_frame(det_model, pose_model, frame, t: float, args) -> Tuple[FrameAnalysis, np.ndarray]:
    frame_h, frame_w = frame.shape[:2]
    det_result = det_model.predict(
        frame,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )[0]
    pose_result = pose_model.predict(
        frame,
        imgsz=args.imgsz,
        conf=args.pose_conf,
        device=args.device,
        verbose=False,
    )[0]

    furniture = extract_furniture(det_result, args.conf)
    person_box, kpts = extract_person_pose(pose_result)
    furniture_label, furniture_score = match_furniture(person_box, furniture, (frame_w, frame_h), kpts)
    is_lying, is_sitting = infer_posture(person_box, kpts, furniture_label, furniture_score)
    raised_hand, wrists = detect_raised_hand(kpts, frame_h)

    analysis = FrameAnalysis(
        t=t,
        frame_size=(frame_w, frame_h),
        person_box=person_box,
        keypoints=kpts,
        furniture_label=furniture_label,
        furniture_score=furniture_score,
        is_lying=is_lying,
        is_sitting=is_sitting,
        raised_hand=raised_hand,
        wrists=wrists,
    )
    return analysis, draw_annotation(frame.copy(), analysis, furniture)


def draw_annotation(frame: np.ndarray, analysis: FrameAnalysis, furniture: List[FurnitureBox]) -> np.ndarray:
    for item in furniture:
        x1, y1, x2, y2 = [int(v) for v in item.box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 180, 255), 2)
        label = FURNITURE_EN_TO_ZH[item.label]
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2)

    if analysis.person_box is not None:
        x1, y1, x2, y2 = [int(v) for v in analysis.person_box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)
        posture = "lying" if analysis.is_lying else "sitting" if analysis.is_sitting else "person"
        cv2.putText(frame, posture, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2)

    if analysis.keypoints is not None:
        for a, b in SKELETON:
            pa = keypoint(analysis.keypoints, a)
            pb = keypoint(analysis.keypoints, b)
            if pa and pb:
                cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 100, 80), 2)
        for idx in range(len(analysis.keypoints)):
            p = keypoint(analysis.keypoints, idx)
            if p:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255, 255, 0), -1)
    return frame


def box_center_y(box: Box) -> float:
    return (box[1] + box[3]) / 2.0


def box_height(box: Box) -> float:
    return max(1.0, box[3] - box[1])


def box_width(box: Box) -> float:
    return max(1.0, box[2] - box[0])


def detect_fall(frames: List[FrameAnalysis]) -> bool:
    valid = [f for f in frames if f.person_box is not None]
    if len(valid) < 4:
        return False
    third = max(1, len(valid) // 3)
    early = valid[:third]
    late = valid[-third:]
    frame_h = valid[-1].frame_size[1]

    early_upright = [
        f for f in early if not f.is_lying and box_height(f.person_box) > box_width(f.person_box) * 1.20
    ]
    late_lying = [f for f in late if f.is_lying]
    if len(early_upright) / len(early) < 0.45 or len(late_lying) / len(late) < 0.45:
        return False

    early_center = np.mean([box_center_y(f.person_box) for f in early])
    late_center = np.mean([box_center_y(f.person_box) for f in late])
    early_height = np.mean([box_height(f.person_box) for f in early])
    late_height = np.mean([box_height(f.person_box) for f in late])
    center_drop = late_center - early_center
    height_shrunk = late_height < early_height * 0.78
    return center_drop > frame_h * 0.06 or height_shrunk


def detect_wave(frames: List[FrameAnalysis]) -> bool:
    valid = [f for f in frames if f.person_box is not None and f.raised_hand]
    if len(valid) < 3:
        return False
    frame_w = valid[-1].frame_size[0]

    for side in ("left", "right"):
        points = [(f.t, f.wrists[side][0]) for f in valid if side in f.wrists]
        if len(points) < 3:
            continue
        xs = [x for _, x in points]
        amplitude = max(xs) - min(xs)
        diffs = np.diff(xs)
        signs = [1 if d > 4 else -1 if d < -4 else 0 for d in diffs]
        nonzero = [s for s in signs if s != 0]
        sign_changes = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a != b)
        if amplitude > max(28.0, frame_w * 0.06) and (sign_changes >= 1 or len(points) >= 5):
            return True
    return False


def common_furniture(frames: List[FrameAnalysis]) -> Optional[str]:
    counts = Counter(f.furniture_label for f in frames if f.furniture_label)
    return counts.most_common(1)[0][0] if counts else None


def summarize(frames: List[FrameAnalysis]) -> str:
    valid = [f for f in frames if f.person_box is not None]
    if not valid:
        return "没有检测到人物。"

    if detect_fall(valid):
        return "检测到有人突然摔倒。"
    if detect_wave(valid):
        return "检测到有人挥手示意。"

    lying_ratio = sum(1 for f in valid if f.is_lying) / len(valid)
    sitting_ratio = sum(1 for f in valid if f.is_sitting) / len(valid)
    furn = common_furniture(valid)
    furn_zh = FURNITURE_EN_TO_ZH.get(furn, "")

    if lying_ratio >= 0.38:
        return f"检测到有人躺在{furn_zh}上。" if furn_zh else "检测到有人躺下。"
    if sitting_ratio >= 0.25:
        return f"检测到有人坐在{furn_zh}上。" if furn_zh else "检测到有人坐下。"
    return "没有识别到指定动作。"


def open_capture(source, width: int, height: int):
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2 if isinstance(source, int) else cv2.CAP_ANY)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def capture_and_analyze(det_model, pose_model, args) -> str:
    source = parse_source(args.camera)
    cap = open_capture(source, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera/source: {args.camera}")

    display = not args.no_display and bool(os.environ.get("DISPLAY"))
    if display:
        cv2.namedWindow("action_recognition", cv2.WINDOW_NORMAL)

    frames: List[FrameAnalysis] = []
    last_annotated = None
    start = time.monotonic()
    next_sample = start
    sample_period = 1.0 / max(1.0, args.sample_fps)

    try:
        while time.monotonic() - start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            now = time.monotonic()
            if now >= next_sample:
                t = now - start
                analysis, last_annotated = analyze_frame(det_model, pose_model, frame, t, args)
                frames.append(analysis)
                next_sample = now + sample_period

            if display:
                shown = last_annotated if last_annotated is not None else frame
                remaining = max(0.0, args.seconds - (time.monotonic() - start))
                cv2.putText(
                    shown,
                    f"Capturing: {remaining:0.1f}s",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                try:
                    cv2.imshow("action_recognition", shown)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                except cv2.error:
                    display = False
    finally:
        cap.release()
        if display:
            cv2.destroyWindow("action_recognition")

    return summarize(frames)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize care actions from camera and speak results with Piper TTS.")
    parser.add_argument("--camera", default="0", help="camera index or video file path; default: 0")
    parser.add_argument("--seconds", type=float, default=10.0, help="seconds captured for each recognition pass")
    parser.add_argument("--sample-fps", type=float, default=3.0, help="analysis sampling FPS")
    parser.add_argument("--width", type=int, default=640, help="requested capture width")
    parser.add_argument("--height", type=int, default=480, help="requested capture height")
    parser.add_argument("--det-model", default=str(DEFAULT_DET_MODEL), help="YOLO object detection model")
    parser.add_argument("--pose-model", default=str(DEFAULT_POSE_MODEL), help="YOLO pose model")
    parser.add_argument("--device", default="auto", help="ultralytics device, e.g. auto, cpu, 0; default auto")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.35, help="object confidence threshold")
    parser.add_argument("--pose-conf", type=float, default=0.30, help="pose confidence threshold")
    parser.add_argument("--loop", action="store_true", help="run recognition repeatedly")
    parser.add_argument("--no-display", action="store_true", help="do not show the live camera window")
    parser.add_argument("--no-tts", action="store_true", help="print speech text instead of playing audio")
    parser.add_argument("--piper-bin", default=str(DEFAULT_PIPER_BIN), help="Piper executable path")
    parser.add_argument("--piper-model", default=str(DEFAULT_PIPER_MODEL), help="Piper ONNX voice model")
    parser.add_argument("--piper-config", default=str(DEFAULT_PIPER_CONFIG), help="Piper model json config")
    parser.add_argument("--length-scale", type=float, default=1.0, help="Piper speech speed; larger is slower")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.device = resolve_device(args.device)
    print(f"[设备] YOLO inference device: {args.device}")
    det_model = load_model(args.det_model)
    pose_model = load_model(args.pose_model)

    while True:
        speak(f"摄像头已打开。提示音结束后开始识别，请保持动作{args.seconds:g}秒。", args)
        result = capture_and_analyze(det_model, pose_model, args)
        speak(result, args)
        print("[识别结果]", result)
        if not args.loop:
            break
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
