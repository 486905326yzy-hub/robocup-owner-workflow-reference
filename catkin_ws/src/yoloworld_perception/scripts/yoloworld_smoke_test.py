#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import torch
from ultralytics import YOLOWorld


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal YOLO-World GPU smoke test.")
    parser.add_argument("--model", default="yolov8s-world.pt", help="Model path or Ultralytics model name")
    parser.add_argument("--classes", default="person", help="Comma-separated text prompts")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-model", default="", help="Optional path to save the loaded model")
    return parser.parse_args()


def main():
    args = parse_args()
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]

    print(f"torch: {torch.__version__}")
    print(f"torch_cuda_runtime: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not visible to this Python process")

    model_path = os.path.expanduser(os.path.expandvars(args.model))
    if os.path.isdir(model_path):
        raise SystemExit(
            f"Model path is a directory, not a .pt file: {model_path}\n"
            "Use: --model ~/catkin_ws/src/yoloworld_perception/models/yolov8s-world-person.pt"
        )
    if os.path.sep in model_path and not os.path.exists(model_path):
        raise SystemExit(f"Model file does not exist: {model_path}")
    print(f"model: {model_path}")
    print(f"classes: {classes}")
    model = YOLOWorld(model_path)
    model.set_classes(classes)

    image = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    start = time.time()
    results = model.predict(image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
    elapsed_ms = (time.time() - start) * 1000.0
    print(f"device: {args.device}")
    print(f"elapsed_ms: {elapsed_ms:.1f}")
    print(f"detections: {0 if results[0].boxes is None else len(results[0].boxes)}")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        print(f"device_name: {torch.cuda.get_device_name(0)}")
        print(f"max_allocated_mb: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f}")

    if args.save_model:
        save_path = os.path.expanduser(os.path.expandvars(args.save_model))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"saved_model: {save_path}")


if __name__ == "__main__":
    main()
