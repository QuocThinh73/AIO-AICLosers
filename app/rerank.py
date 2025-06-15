from collections import defaultdict

def rrf(list_paths, k_rrf=60):
    rrf_scores = defaultdict(float)

    for paths in list_paths.values():
        for rank, p in enumerate(paths, start=1):
            rrf_scores[p] += 1.0 / (k_rrf + rank)

    # sắp xếp theo RRF giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    fused_paths = [p for p, _ in sorted_items]
    fused_scores = [s for _, s in sorted_items]
    return fused_paths, fused_scores