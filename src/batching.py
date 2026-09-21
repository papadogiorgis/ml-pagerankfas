# src/pagerankfas/batching.py

import os
import time
import itertools
import pandas as pd
import networkx as nx
import joblib
from networkx.algorithms.approximation import maximum_independent_set

from utils import build_line_digraph, get_scc
from features import get_pairwise_features

def pagerankfas_batching(
    graph: nx.DiGraph, 
    model_path: str = None, 
    threshold: float = 0.30, 
    batch_size: int = 20,
    show_prints: bool = False
) -> list:
    """
    Computes a Feedback Arc Set (FAS) using ML-Predicted Batching.
    Evaluates and removes batches of edges PER Strongly Connected Component (SCC).
    """
    # 1. Initialization and Setup
    G = graph.copy()
    fas = []
    
    # Resolve model path dynamically if not provided
    if model_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(base_dir, 'src', 'rf_model.pkl')
    
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
    
    # 2. Main Algorithm Loop (Runs until the whole graph is a DAG)
    while not nx.is_directed_acyclic_graph(G):
        iteration_num += 1
        
        # Track how many edges we remove across all SCCs in this one massive loop
        edges_removed_this_loop = 0 
        
        # Get all SCCs
        SCCs = get_scc(G)
        
        # 3. Process each SCC independently
        for component in SCCs:
            
            LG = build_line_digraph(component)
            N = len(LG.nodes)
            if N == 0:
                continue

            out_degrees = dict(LG.out_degree())
            successors = {n: list(LG.successors(n)) for n in LG.nodes}
            
            # 3a. Run PageRank for this specific SCC
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

            # 3b. Extract the Top-K candidate edges for THIS SCC
            sorted_edges = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_data = sorted_edges[:batch_size]
            
            # If no edges, skip
            if not top_k_data:
                continue
            
            # If only 1 edge in this SCC, fallback to standard removal immediately
            if len(top_k_data) == 1:
                edge = top_k_data[0][0]
                fas.append(edge)
                G.remove_edge(*edge)
                edges_removed_this_loop += 1
                continue

            # 3c. Feature Extraction for all unique pairs IN THIS SCC
            pairs = list(itertools.combinations(top_k_data, 2))
            feature_list = []
            pair_mappings = [] 

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

            # 3d. ML Conflict Prediction
            features_df = pd.DataFrame(feature_list)
            conflict_probs = rf_model.predict_proba(features_df)[:, 1]

            # 3e. Build the Conflict Graph for this SCC
            conflict_graph = nx.Graph()
            for edge_data, _ in top_k_data:
                conflict_graph.add_node(edge_data)

            for idx, prob in enumerate(conflict_probs):
                if prob >= threshold:
                    edge_A, edge_B = pair_mappings[idx]
                    conflict_graph.add_edge(edge_A, edge_B)

            # 3f. Find the Maximum Independent Set (MIS)
            safe_batch = maximum_independent_set(conflict_graph)
            
            if not safe_batch:
                safe_batch = [top_k_data[0][0]]

            # 3g. Batch Removal for this SCC
            for edge in safe_batch:
                fas.append(edge)
                G.remove_edge(*edge)
                edges_removed_this_loop += 1
        
        # 4. Heartbeat print (Triggers once per full graph pass)
        if show_prints:
            current_time = time.time()
            total_time = current_time - start_time
            time_since_last_print = current_time - last_print_time
            last_print_time = current_time
            
            print(f"[Batching Heartbeat] Loop: {iteration_num} | "
                  f"Edges Removed: {edges_removed_this_loop} | "
                  f"Total FAS Size: {len(fas)} | "
                  f"Edges left in G: {G.number_of_edges()} | "
                  f"Total Time: {total_time:.2f}s "
                  f"(loop took: {time_since_last_print:.2f}s)")

    return fas