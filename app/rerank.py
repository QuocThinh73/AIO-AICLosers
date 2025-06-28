from collections import defaultdict

def rrf(list_paths, k_rrf=60):
    """
    Implement Reciprocal Rank Fusion to combine search results from multiple models.
    
    Args:
        list_paths (dict): Dictionary mapping model names to their ranked result paths
        k_rrf (int): RRF parameter for fusion calculation (default: 60)
    
    Returns:
        tuple: (fused_paths, fused_scores) - Combined and re-ranked results
    """
    rrf_scores = defaultdict(float)

    # Calculate RRF scores for each path across all models
    for paths in list_paths.values():
        for rank, p in enumerate(paths, start=1):
            rrf_scores[p] += 1.0 / (k_rrf + rank)

    # Sort by RRF score in descending order
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    fused_paths = [p for p, _ in sorted_items]
    fused_scores = [s for _, s in sorted_items]
    return fused_paths, fused_scores