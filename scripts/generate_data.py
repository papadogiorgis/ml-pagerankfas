# scripts/generate_data.py

import os
import sys
import random
import time
import multiprocessing
import networkx as nx
import pandas as pd
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure the src directory is in the Python path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pagerankfas.classic import pagerankfas
from pagerankfas.utils.scc_handler import get_scc
from pagerankfas.utils.line_graph import build_line_digraph
from pagerankfas.ml_enhanced.features import get_pairwise_features

def get_top_k_edges(G: nx.DiGraph, k: int = 10) -> list:
    """
    Runs the first iteration of the PageRankFAS scoring logic to extract
    the Top-K edges across all Strongly Connected Components (SCCs).
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

        # Store scores mapping line-graph nodes (which are edges in G) to their scores
        for edge, score in pagerank_scores.items():
            edge_scores[edge] = score

    # Sort edges descending by score and return the top K
    sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_edges[:k]

def process_single_graph(i, num_graphs, global_start_time):
    """
    Worker function that handles the generation and processing of a single graph.
    """
    random.seed()
    graph_start_time = time.time()
    
    # Extract the core/process number from the multiprocessing process name
    process_name = multiprocessing.current_process().name
    core_id = ''.join(filter(str.isdigit, process_name))
    if not core_id:
        core_id = str(os.getpid())

    # Print start timestamp
    current_t = time.time() - global_start_time
    print(f"{current_t:.2f}: [Core {core_id} starts processing Graph {i+1}]", flush=True)
    
    n = random.randint(30, 100)
    p = random.uniform(0.05, 0.15)
    G = nx.gnp_random_graph(n=n, p=p, directed=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    
    # Handle skipped graphs early, returning empty datasets
    if nx.is_directed_acyclic_graph(G):
        return (i, [], 0, core_id, time.time() - graph_start_time)

    baseline_fas = pagerankfas(G, show_prints=False)
    baseline_size = len(baseline_fas)
    
    top_10 = get_top_k_edges(G, k=10)
    
    if len(top_10) < 2:
        return (i, [], 0, core_id, time.time() - graph_start_time)
    
    pairs = list(combinations(top_10, 2))
    safe = 0
    graph_dataset = []
    
    for pair_idx, ((edge_1, pr_1), (edge_2, pr_2)) in enumerate(pairs, 1):
        if pr_1 >= pr_2:
            edge_A, pr_A = edge_1, pr_1
            edge_B, pr_B = edge_2, pr_2
        else:
            edge_A, pr_A = edge_2, pr_2
            edge_B, pr_B = edge_1, pr_1

        features = get_pairwise_features(G, edge_A, edge_B, pr_A, pr_B)

        G_test = G.copy()
        G_test.remove_edge(*edge_A)
        G_test.remove_edge(*edge_B)
        
        test_fas_size = 2 + len(pagerankfas(G_test, show_prints=False))
        
        label = 1 if test_fas_size > baseline_size else 0
        features["label"] = label
        if label == 0:
            safe += 1
            
        graph_dataset.append(features)

    graph_duration = time.time() - graph_start_time
    
    # Return all the metadata so the main process can handle the final print safely
    return (i, graph_dataset, safe, core_id, graph_duration)


def generate_dataset(num_graphs: int = 3000, output_csv: str = "training_data.csv"):
    dataset = []
    global_start_time = time.time()
    
    print(f"0.00: Starting parallel data generation for {num_graphs} graphs...")
    completed = 0
    total_safe = 0

    with ProcessPoolExecutor() as executor:
        # Pass global_start_time into each worker
        futures = {
            executor.submit(process_single_graph, i, num_graphs, global_start_time): i 
            for i in range(num_graphs)
        }
        
        # as_completed yields tasks exactly as they finish
        for future in as_completed(futures):
            try:
                i, graph_dataset, safe, core_id, graph_duration = future.result()
                
                if graph_dataset:
                    dataset.extend(graph_dataset)
                
                completed += 1
                total_safe += safe
                t = time.time() - global_start_time
                left = ((num_graphs * t) / completed ) - t
                
                # Print the formatted completion string
                print(f"{t:.2f}: CORE {core_id} COMPLETED GRAPH {i+1} IN {graph_duration:.2f}s. [SAFE = {safe}/45, COMPLETED = {completed}/{num_graphs}]", flush=True, end=" ")
                print(f"[SAFE = {((safe*100)/45):.2f}% | TOTAL SAFE = {((total_safe*100)/(45*completed)):.2f}%]", end=" ")
                print(f"{left:.1f} seconds left ({(left / 3600):.1f} hours)")
                
            except Exception as e:
                graph_num = futures[future] + 1
                print(f"Graph {graph_num} processing failed: {e}")

    df = pd.DataFrame(dataset)
    df.to_csv(output_csv, index=False)
    print(f"Data generation complete. Saved {len(df)} pairs to {output_csv}.")
    
    print("Label Distribution:")
    print(df["label"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

if __name__ == "__main__":
    generate_dataset(num_graphs=3000, output_csv="tests/datasets/training_data.csv")