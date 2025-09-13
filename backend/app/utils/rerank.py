from typing import List
from collections import defaultdict


def rrf(results: List[dict], top_k: int, k: int = 60):
    if len(results) == 0:
        return []
    elif len(results) == 1:
        return results[0]
    else:
        rrf_scores = defaultdict(float)
        for result in results:
            for rank, keyframe in enumerate(result, start=1):
                rrf_scores[keyframe] += 1 / (rank + k)
        reranked_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [result[0] for result in reranked_results]
