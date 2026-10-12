from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


class PersonGallery:
    """Multi-shot gallery: every accepted registration embedding is retained."""

    def __init__(self, entries: dict[str, list[np.ndarray]] | None = None):
        self.entries = entries or {}

    def register(self, name: str, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError('embedding must not be zero')
        self.entries.setdefault(name, []).append(vector / norm)

    def register_batch(self, name: str, embeddings: Iterable[np.ndarray], replace: bool = False) -> None:
        if replace:
            self.entries[name] = []
        for embedding in embeddings:
            self.register(name, embedding)

    def identify(self, query_embedding: np.ndarray, top_k: int = 3,
                 similarity_threshold: float = 0.65, margin_threshold: float = 0.08) -> dict:
        if not self.entries:
            raise RuntimeError('gallery is empty')
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores: dict[str, float] = {}
        per_image: dict[str, list[float]] = {}
        for name, vectors in self.entries.items():
            if not vectors:
                continue
            sims = np.asarray(vectors, dtype=np.float32) @ query
            best = np.sort(sims)[-min(top_k, len(sims)):]
            scores[name] = float(best.mean())
            per_image[name] = [float(value) for value in sims]
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else -1.0
        margin = best_score - second_score
        accepted = best_score >= similarity_threshold and margin >= margin_threshold
        return {
            'identity': best_name if accepted else 'Unknown',
            'best_candidate': best_name,
            'score': best_score,
            'second_score': second_score,
            'margin': margin,
            'accepted': accepted,
            'owner_scores': scores,
            'per_image_similarity': per_image,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = sorted(self.entries)
        payload = {f'embeddings_{index}': np.asarray(self.entries[name], dtype=np.float32)
                   for index, name in enumerate(names)}
        payload['names_json'] = np.asarray(json.dumps(names))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> 'PersonGallery':
        archive = np.load(Path(path), allow_pickle=False)
        names = json.loads(str(archive['names_json'].item()))
        entries = {name: [row for row in archive[f'embeddings_{index}']]
                   for index, name in enumerate(names)}
        return cls(entries)
