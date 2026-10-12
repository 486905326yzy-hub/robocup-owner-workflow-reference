#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gallery import PersonGallery
from reid_model import ReIDModel
from utils import list_images, load_config, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Identify one or more already cropped full-body images.')
    parser.add_argument('--input', required=True, help='image path or directory of body crops')
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    source = Path(args.input)
    images = [source] if source.is_file() else list_images(source)
    if not images:
        raise SystemExit(f'No supported images found in {source}')
    model_cfg = config['model']
    model = ReIDModel(model_cfg['name'], resolve_path(model_cfg['weights']), model_cfg['device'],
                      model_cfg['image_height'], model_cfg['image_width'], model_cfg['mask_head'], model_cfg['head_mask_ratio'])
    gallery = PersonGallery.load(resolve_path(config['paths']['gallery']))
    params = config['reid']
    for image in images:
        result = gallery.identify(model.extract_embedding(image), top_k=params['top_k'],
                                  similarity_threshold=params['similarity_threshold'], margin_threshold=params['margin_threshold'])
        result['image'] = str(image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
