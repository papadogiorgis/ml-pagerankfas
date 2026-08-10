# src/pagerankfas/batching.py

import os
import time
import itertools
import pandas as pd
import networkx as nx
import joblib
from networkx.algorithms.approximation import maximum_independent_set

from .utils.scc_handler import get_scc
from .utils.line_graph import build_line_digraph
from .ml_enhanced.features import get_pairwise_features

def get_top_k_edges(G: nx.DiGraph, k: int = 10) -> list:
    """
    Computes PageRank across all SCCs and returns the top K edges globally.
    """
    SCCs = get_scc(G)
    edge_scores = {}

    for component in SCCs:
        LG = build_line_digraph(component)
        N = len(LG.nodes)
        if N == 0:
            continue

        out_degrees = dict(LG.out_degree())
        successors = {n: list(LG.successors(n)) for n in LG.nodes}
        
        # 5 iterations of PageRank (No Damping Factor)
        pagerank_scores = {node: 1.0 / N for node in LG.nodes}
        for _ in range(5):
            next_scores = {node: 0.0 for node in LG.nodes}
            for node in LG.nodes:
                deg = out_degrees[node]
                if deg > 0:
                    share = pagerank_scores[node] / deg
                    for target in successors[node]:
                        next_scores[target] += share
                else:
                    share = pagerank_scores[node] / N
                    for target in LG.nodes:
                        next_scores[target] += share
            pagerank_scores = next_scores

        for edge, score in pagerank_scores.items():
            edge_scores[edge] = score

    # Sort edges descending by score and return the top K
    sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_edges[:k]


def pagerankfas_batching(
    graph: nx.DiGraph, 
    model_path: str = None, 
    threshold: float = 0.30, 
    batch_size: int = 10,
    show_prints: bool = False
) -> list:
    """
    Computes a Feedback Arc Set (FAS) using ML-Predicted Batching.
    """
    # 1. Initialization and Setup
    G = graph.copy()
    fas = []
    
    # Resolve model path dynamically if not provided
    if model_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        model_path = os.path.join(base_dir, 'src', 'pagerankfas', 'ml_enhanced', 'rf_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    
    rf_model = joblib.load(model_path)
    
    # Remove self loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        fas.extend(self_loops)
        G.remove_edges_from(self_loops)
        if show_prints:
            print(f"[Init] Stripped {len(self_loops)} self-loops.")

    iteration_num = 0
    start_time = time.time()
    last_print_time = start_time
    
    # 2. Main Algorithm Loop
    while not nx.is_directed_acyclic_graph(G):
        iteration_num += 1
        
        # Extract the Top-K candidate edges
        top_k_data = get_top_k_edges(G, k=batch_size)
        
        # If no edges returned or only 1 edge, fallback to standard removal
        if not top_k_data:
            break
        if len(top_k_data) == 1:
            edge = top_k_data[0][0]
            fas.append(edge)
            G.remove_edge(*edge)
            continue

        # 3. Feature Extraction for all unique pairs
        pairs = list(itertools.combinations(top_k_data, 2))
        feature_list = []
        pair_mappings = [] # To keep track of which pair belongs to which row

        for (edge_1, pr_1), (edge_2, pr_2) in pairs:
            # Enforce edge_A having the higher PageRank
            if pr_1 >= pr_2:
                edge_A, pr_A = edge_1, pr_1
                edge_B, pr_B = edge_2, pr_2
            else:
                edge_A, pr_A = edge_2, pr_2
                edge_B, pr_B = edge_1, pr_1
                
            features = get_pairwise_features(G, edge_A, edge_B, pr_A, pr_B)
            feature_list.append(features)
            pair_mappings.append((edge_A, edge_B))

        # 4. ML Conflict Prediction
        # Convert to DataFrame to ensure feature names match training data
        features_df = pd.DataFrame(feature_list)
        
        # Extract the probability of Class 1 (Conflict)
        conflict_probs = rf_model.predict_proba(features_df)[:, 1]

        # 5. Build the Conflict Graph
        conflict_graph = nx.Graph()
        
        # Add all Top-K edges as nodes in the conflict graph
        for edge_data, _ in top_k_data:
            conflict_graph.add_node(edge_data)

        # Add edges between nodes that the model predicts will conflict
        for idx, prob in enumerate(conflict_probs):
            if prob >= threshold:
                edge_A, edge_B = pair_mappings[idx]
                conflict_graph.add_edge(edge_A, edge_B)

        # 6. Find the Maximum Independent Set (MIS)
        # The MIS gives us the largest set of edges that DO NOT conflict
        safe_batch = maximum_independent_set(conflict_graph)
        
        # Fallback: if the approximation returns empty, remove just the absolute top edge
        if not safe_batch:
            safe_batch = [top_k_data[0][0]]

        # 7. Batch Removal
        for edge in safe_batch:
            fas.append(edge)
            G.remove_edge(*edge)
        
        # Heartbeat print
        if show_prints:
            current_time = time.time()
            total_time = current_time - start_time
            time_since_last_print = current_time - last_print_time
            last_print_time = current_time
            
            print(f"[Batching Heartbeat] Loop: {iteration_num} | "
                  f"Batch Size: {len(safe_batch)} | "
                  f"FAS Size: {len(fas)} | "
                  f"Edges left: {G.number_of_edges()} | "
                  f"Total Time: {total_time:.2f}s")

    return fas