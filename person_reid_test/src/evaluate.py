#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gallery import PersonGallery
from reid_model import ReIDModel
from utils import ensure_parent, list_images, load_config, resolve_path


def query_rows(query_root: Path):
    labels = query_root / 'labels.csv'
    if labels.exists():
        with labels.open(encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                yield query_root / row['path'], row['owner'], row.get('pose', 'unspecified')
        return
    for owner_dir in sorted(path for path in query_root.iterdir() if path.is_dir()):
        for image in list_images(owner_dir):
            yield image, owner_dir.name, 'unspecified'


def main() -> int:
    parser = argparse.ArgumentParser(description='Produce ReID similarity matrix and pose metrics.')
    parser.add_argument('--query-dir', default='data/query')
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    gallery = PersonGallery.load(resolve_path(config['paths']['gallery']))
    owners = sorted(gallery.entries)
    model_cfg, params = config['model'], config['reid']
    model = ReIDModel(model_cfg['name'], resolve_path(model_cfg['weights']), model_cfg['device'],
                      model_cfg['image_height'], model_cfg['image_width'], model_cfg['mask_head'], model_cfg['head_mask_ratio'])
    records = []
    for image, truth, pose in query_rows(resolve_path(args.query_dir)):
        result = gallery.identify(model.extract_embedding(image), top_k=params['top_k'],
                                  similarity_threshold=params['similarity_threshold'], margin_threshold=params['margin_threshold'])
        records.append({'image': str(image), 'truth': truth, 'pose': pose, 'prediction': result['identity'],
                        'score': result['score'], **result['owner_scores']})
    if not records:
        raise SystemExit('No labelled queries. Use data/query/<owner>/*.jpg or data/query/labels.csv')
    frame = pd.DataFrame(records)
    matrix = pd.DataFrame(index=owners, columns=sorted(frame['truth'].unique()), dtype=float)
    for owner in owners:
        for truth in matrix.columns:
            matrix.loc[owner, truth] = frame.loc[frame.truth == truth, owner].mean()
    output_dir = resolve_path(config['paths']['matrix_dir']); output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / 'similarity_matrix.csv', float_format='%.4f')
    plt.figure(figsize=(max(5, len(matrix.columns) * 1.4), max(4, len(matrix.index) * 1.0)))
    sns.heatmap(matrix, annot=True, fmt='.3f', vmin=0, vmax=1, cmap='viridis')
    plt.xlabel('Query identity'); plt.ylabel('Gallery identity'); plt.tight_layout()
    plt.savefig(output_dir / 'similarity_matrix.png', dpi=180); plt.close()
    frame['correct'] = frame.truth == frame.prediction
    frame['unknown'] = frame.prediction == 'Unknown'
    metrics = []
    for label, group in [('overall', frame), *[(f'pose:{pose}', group) for pose, group in frame.groupby('pose')]]:
        known = group[~group.unknown]
        metrics.append({'group': label, 'samples': len(group), 'accuracy': group.correct.mean(),
                        'unknown_rate': group.unknown.mean(),
                        'false_match_rate': (known.prediction != known.truth).mean() if len(known) else 0.0})
    pd.DataFrame(metrics).to_csv(output_dir / 'metrics.csv', index=False, float_format='%.4f')
    frame.to_csv(output_dir / 'per_image_results.csv', index=False, float_format='%.4f')
    print(f'Wrote {output_dir / "similarity_matrix.csv"}, similarity_matrix.png, metrics.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
