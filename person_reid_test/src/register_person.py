#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gallery import PersonGallery
from reid_model import ReIDModel
from utils import list_images, load_config, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Register one owner from multiple cropped full-body images.')
    parser.add_argument('--name', required=True, help='owner label, for example owner1')
    parser.add_argument('--input', help='directory of body crops (default: data/<name>)')
    parser.add_argument('--replace', action='store_true', help='replace existing gallery features for this name')
    parser.add_argument('--allow-fewer', action='store_true', help='permit fewer than configured registration images')
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    directory = Path(args.input) if args.input else resolve_path(f'data/{args.name}')
    images = list_images(directory)
    minimum = int(config['reid']['min_registration_images'])
    if not images:
        print(f'No images in {directory}', file=sys.stderr)
        return 2
    if len(images) < minimum and not args.allow_fewer:
        print(f'Need at least {minimum} images, found {len(images)}. Use --allow-fewer only for smoke tests.', file=sys.stderr)
        return 2
    model_cfg = config['model']
    model = ReIDModel(model_name=model_cfg['name'], weights=resolve_path(model_cfg['weights']),
                      device=model_cfg['device'], image_height=model_cfg['image_height'],
                      image_width=model_cfg['image_width'], mask_head=model_cfg['mask_head'],
                      head_mask_ratio=model_cfg['head_mask_ratio'])
    embeddings = []
    for index, image in enumerate(images, 1):
        embeddings.append(model.extract_embedding(image))
        print(f'[{index}/{len(images)}] {image.name}')
    gallery_path = resolve_path(config['paths']['gallery'])
    gallery = PersonGallery.load(gallery_path) if gallery_path.exists() else PersonGallery()
    gallery.register_batch(args.name, embeddings, replace=args.replace)
    gallery.save(gallery_path)
    print(f'Saved {len(embeddings)} features for {args.name} to {gallery_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
