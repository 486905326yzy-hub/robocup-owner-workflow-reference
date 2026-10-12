from __future__ import annotations

from collections import defaultdict


def temporal_average(frame_results: list[dict]) -> dict:
    """Average each owner score across frames, then apply the same acceptance rule."""
    if not frame_results:
        raise ValueError('no frame results')
    all_names = {name for result in frame_results for name in result['owner_scores']}
    scores = {name: sum(result['owner_scores'].get(name, 0.0) for result in frame_results) /
                   len(frame_results) for name in all_names}
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else -1.0
    first = frame_results[0]
    margin = best_score - second_score
    accepted = best_score >= first['similarity_threshold'] and margin >= first['margin_threshold']
    return {'identity': best_name if accepted else 'Unknown', 'best_candidate': best_name,
            'score': best_score, 'second_score': second_score, 'margin': margin,
            'accepted': accepted, 'owner_scores': scores, 'frames': len(frame_results)}
