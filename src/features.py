# src/features.py

import networkx as nx

def get_pairwise_features(G: nx.DiGraph, edge_A: tuple, edge_B: tuple, pr_A: float, pr_B: float) -> dict:
    """
    Extracts structural and scoring features for a pair of candidate edges.
    
    IMPORTANT: To ensure consistency for the Machine Learning model, 
    edge_A should always be the edge with the higher PageRank score (pr_A >= pr_B).
    """
    u, v = edge_A
    x, y = edge_B
    
    # N is used as a penalty value for unreachable paths
    N = G.number_of_nodes()

    # 1. Shared Nodes (0, 1, or 2)
    shared_nodes = len(set(edge_A).intersection(set(edge_B)))

    # 2. Distance A -> B (from A's target to B's source)
    try:
        dist_A_to_B = nx.shortest_path_length(G, source=v, target=x)
    except nx.NetworkXNoPath:
        dist_A_to_B = N 

    # 3. Distance B -> A (from B's target to A's source)
    try:
        dist_B_to_A = nx.shortest_path_length(G, source=y, target=u)
    except nx.NetworkXNoPath:
        dist_B_to_A = N

    # 4. Common Successors
    successors_A = set(G.successors(v))
    successors_B = set(G.successors(y))
    common_successors = len(successors_A.intersection(successors_B))

    # 5. Common Predecessors
    predecessors_A = set(G.predecessors(u))
    predecessors_B = set(G.predecessors(x))
    common_predecessors = len(predecessors_A.intersection(predecessors_B))

    # 6. PageRank Score Difference
    pr_diff = abs(pr_A - pr_B)

    # 7. PageRank Score Ratio (avoids division by zero)
    pr_ratio = min(pr_A, pr_B) / max(pr_A, pr_B) if max(pr_A, pr_B) > 0 else 0.0

    return {
        "shared_nodes": shared_nodes,
        "dist_A_to_B": dist_A_to_B,
        "dist_B_to_A": dist_B_to_A,
        "common_successors": common_successors,
        "common_predecessors": common_predecessors,
        "pr_diff": pr_diff,
        "pr_ratio": pr_ratio
    }