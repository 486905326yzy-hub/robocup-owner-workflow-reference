#!/usr/bin/env python3
import argparse
import os
import sys
import time

import cv2
import numpy as np


def select_largest_face(faces):
    if not faces:
        return None

    def priority(face):
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) < 4:
            return 0.0
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        det_score = float(getattr(face, "det_score", 1.0) or 1.0)
        return width * height * det_score

    return max(faces, key=priority)


def normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None
    return vector / norm


def main():
    parser = argparse.ArgumentParser(description="Check owner face reference with InsightFace.")
    parser.add_argument("--image", default="/home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/data/owner/owner.jpg")
    parser.add_argument("--model-name", default="buffalo_sc")
    parser.add_argument("--model-root", default="/home/ubuntu20/.insightface")
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--det-thresh", type=float, default=0.45)
    args = parser.parse_args()

    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        print("ERROR: InsightFace is not installed: %s" % exc, file=sys.stderr)
        return 2

    image_path = os.path.expanduser(args.image)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("ERROR: could not read image: %s" % image_path, file=sys.stderr)
        return 2

    app = FaceAnalysis(name=args.model_name, root=os.path.expanduser(args.model_root), providers=["CPUExecutionProvider"])
    start = time.time()
    app.prepare(ctx_id=-1, det_thresh=args.det_thresh, det_size=(args.det_size, args.det_size))
    load_elapsed = time.time() - start

    start = time.time()
    faces = app.get(image)
    infer_elapsed = time.time() - start
    face = select_largest_face(faces)
    if face is None:
        print("ERROR: no face found in %s" % image_path, file=sys.stderr)
        return 1

    embedding = normalize_embedding(face.embedding)
    if embedding is None:
        print("ERROR: invalid embedding", file=sys.stderr)
        return 1

    bbox = [round(float(value), 1) for value in face.bbox]
    print("face_count=%d" % len(faces))
    print("selected_bbox=%s" % bbox)
    print("det_score=%.3f" % float(getattr(face, "det_score", 0.0)))
    print("embedding_dim=%d" % embedding.shape[0])
    print("model_load_sec=%.2f" % load_elapsed)
    print("inference_sec=%.3f" % infer_elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
