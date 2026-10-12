#!/usr/bin/env python3
"""Temporal ReID for a video containing one pre-cropped person.

YOLO is deliberately absent in phase 1: input frames must already be body crops.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque

import cv2

from gallery import PersonGallery
from reid_model import ReIDModel
from similarity import temporal_average
from utils import ensure_parent, load_config, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Run temporal ReID on a single-person crop video.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='outputs/visualization/video_reid.mp4')
    parser.add_argument('--every', type=int, default=3, help='run OSNet every N frames')
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    if args.every < 1:
        raise SystemExit('--every must be at least 1')
    config = load_config(args.config)
    model_cfg, params = config['model'], config['reid']
    model = ReIDModel(model_cfg['name'], resolve_path(model_cfg['weights']), model_cfg['device'],
                      model_cfg['image_height'], model_cfg['image_width'], model_cfg['mask_head'], model_cfg['head_mask_ratio'])
    gallery = PersonGallery.load(resolve_path(config['paths']['gallery']))
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise SystemExit(f'Cannot open {args.input}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = ensure_parent(resolve_path(args.output))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    csv_path = out_path.with_suffix('.csv')
    history, last = deque(maxlen=params['temporal_window']), None
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        rows = csv.DictWriter(handle, fieldnames=['frame', 'identity', 'score', 'margin', 'owner_scores'])
        rows.writeheader()
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.every == 0:
                result = gallery.identify(model.extract_embedding(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                          top_k=params['top_k'], similarity_threshold=params['similarity_threshold'],
                                          margin_threshold=params['margin_threshold'])
                result['similarity_threshold'] = params['similarity_threshold']
                result['margin_threshold'] = params['margin_threshold']
                history.append(result)
                last = temporal_average(list(history))
                rows.writerow({'frame': frame_index, 'identity': last['identity'], 'score': f"{last['score']:.3f}",
                               'margin': f"{last['margin']:.3f}", 'owner_scores': last['owner_scores']})
            if last:
                label = f"{last['identity']}  ReID:{last['score']:.2f}  margin:{last['margin']:.2f}"
                cv2.rectangle(frame, (0, 0), (min(width, 620), 42), (0, 0, 0), -1)
                cv2.putText(frame, label, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
            writer.write(frame)
            frame_index += 1
    capture.release(); writer.release()
    print(f'Wrote {out_path} and {csv_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
