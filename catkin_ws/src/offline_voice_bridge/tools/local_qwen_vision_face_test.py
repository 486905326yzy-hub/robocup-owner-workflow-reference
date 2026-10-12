#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

import cv2

import local_voice_test


DEFAULT_SPEAKER_DEVICE = "plughw:CARD=Generic_1,DEV=0"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_QWEN_MODEL = "qwen3.5:9b"


def capture_frame(args):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index %s" % args.camera_index)

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Qwen Face Feature Test"
    captured = None
    if args.manual:
        print("Camera opened. Press SPACE to analyze current frame, or Q to quit.")
    else:
        print("Camera opened. Capturing automatically after %.1f seconds..." % args.warmup_seconds)
    start_ticks = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")
            cv2.imshow(window_name, frame)
            elapsed = (cv2.getTickCount() - start_ticks) / tick_freq
            if not args.manual and elapsed >= args.warmup_seconds:
                captured = frame.copy()
                cv2.waitKey(1)
                break
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if args.manual and key == 32:
                captured = frame.copy()
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured is None:
        raise RuntimeError("No frame captured")
    return captured


def frame_to_jpeg_base64(frame, output_path=None):
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG")
    image_bytes = encoded.tobytes()
    if output_path:
        with open(output_path, "wb") as image_file:
            image_file.write(image_bytes)
        print("Captured image: %s" % output_path)
    return base64.b64encode(image_bytes).decode("ascii")


def call_qwen_vision(args, image_b64):
    prompt = (
        "/no_think\n"
        "Look at the provided image and describe only visible, non-sensitive facial appearance features of the person. "
        "Do not identify the person. Do not describe or infer race, ethnicity, nationality, skin color, skin tone, religion, health, or gender identity. "
        "Mention concrete visible details such as face shape, hairstyle or hair length/color if visible, glasses, facial hair, expression, and notable visible accessories. "
        "If no face is clearly visible, say that clearly. "
        "Answer in one or two short English sentences suitable for text-to-speech."
    )
    payload = {
        "model": args.llm_model,
        "stream": False,
        "think": False,
        "keep_alive": args.llm_keep_alive,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "options": {
            "temperature": 0.2,
            "top_p": 0.7,
            "num_predict": args.llm_max_tokens,
            "num_ctx": 2048,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.llm_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.llm_timeout) as response:
            raw = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise RuntimeError("Qwen vision request timed out. First run may load the model; retry or increase --llm-timeout. %s" % exc)
    except urllib.error.URLError as exc:
        raise RuntimeError("Qwen vision request failed: %s" % exc)

    result = json.loads(raw)
    content = result.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Qwen vision returned empty content: %s" % raw)
    return content.replace("\n", " ").strip()


def speak(args, text):
    wav_path = args.output_wav or "/dev/shm/local_qwen_vision_face_reply.wav"
    engine = local_voice_test.synthesize(text, wav_path, args)
    print("TTS engine: %s" % engine)
    print("Reply WAV: %s" % wav_path)
    if not args.no_play:
        local_voice_test.play_wav(wav_path, args.player, args.speaker_device)


def build_parser():
    parser = argparse.ArgumentParser(description="Open webcam, analyze visible facial appearance with local Qwen vision, and speak the result.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--image", help="Use an existing image file instead of opening the camera")
    parser.add_argument("--manual", action="store_true", help="Require SPACE to capture instead of automatic capture")
    parser.add_argument("--warmup-seconds", type=float, default=2.0, help="Camera preview time before automatic capture")
    parser.add_argument("--save-image", default="/tmp/local_qwen_vision_face.jpg")

    parser.add_argument("--llm-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--llm-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--llm-max-tokens", type=int, default=120)
    parser.add_argument("--llm-keep-alive", default="10m")

    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--output-wav")
    parser.add_argument("--speaker-device", default=DEFAULT_SPEAKER_DEVICE)
    parser.add_argument("--engine", choices=["auto", "piper", "flite"], default="auto")
    parser.add_argument("--piper", default=local_voice_test.DEFAULT_PIPER)
    parser.add_argument("--piper-model", default=local_voice_test.DEFAULT_PIPER_MODEL)
    parser.add_argument("--piper-config", default=local_voice_test.DEFAULT_PIPER_CONFIG)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--flite", default="flite")
    parser.add_argument("--flite-voice", default="slt")
    parser.add_argument("--player", default="aplay")
    return parser


def main():
    args = build_parser().parse_args()
    try:
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                raise RuntimeError("Could not read image: %s" % args.image)
            image_b64 = frame_to_jpeg_base64(frame, args.save_image)
        else:
            frame = capture_frame(args)
            image_b64 = frame_to_jpeg_base64(frame, args.save_image)

        description = call_qwen_vision(args, image_b64)
        print("Vision description: %s" % description)
        speak(args, description)
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
