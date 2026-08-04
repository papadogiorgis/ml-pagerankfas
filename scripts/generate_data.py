# scripts/generate_data.py

import os
import sys
import random
import time
import networkx as nx
import pandas as pd
from itertools import combinations

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

def generate_dataset(num_graphs: int = 3000, output_csv: str = "training_data.csv"):
    """
    Generates random directed graphs, extracts the Top-10 edges, tests the 45 pairs,
    extracts their features, and writes the results to a CSV.
    """
    dataset = []
    
    print(f"Starting data generation for {num_graphs} graphs...")

    for i in range(num_graphs):
        graph_start_time = time.time()
        # Randomize size (e.g., 50 to 150 nodes) and density (e.g., 5% to 20%)
        n = random.randint(50, 150)
        p = random.uniform(0.05, 0.2)
        G = nx.gnp_random_graph(n=n, p=p, directed=True)
        
        # Strip self-loops immediately to match classic.py initialization
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        
        if nx.is_directed_acyclic_graph(G):
            continue

        # 1. Establish the Baseline FAS size
        print(f"\n--- Graph {i+1}/{num_graphs} (Nodes: {n}, Edges: {G.number_of_edges()}, Density: {(p*100):.2f}%) ---")
        print("Calculating baseline FAS...")
        baseline_fas = pagerankfas(G, show_prints=False)
        baseline_size = len(baseline_fas)
        print(f"Baseline FAS size: {baseline_size}")

        # 2. Get the Top-10 candidate edges and their scores
        top_10 = get_top_k_edges(G, k=10)
        
        # If the graph doesn't have at least 2 edges in cycles, skip
        if len(top_10) < 2:
            print("Not enough cycle edges, skipping.")
            continue
        
        # 3. Generate the 45 unique pairs from the Top-10 edges
        pairs = list(combinations(top_10, 2))
        print(f"Testing {len(pairs)} pairs...")
        safe = 0
        
        for pair_idx, ((edge_1, pr_1), (edge_2, pr_2)) in enumerate(pairs, 1):
            # Ensure edge_A is always the one with the higher PageRank score
            if pr_1 >= pr_2:
                edge_A, pr_A = edge_1, pr_1
                edge_B, pr_B = edge_2, pr_2
            else:
                edge_A, pr_A = edge_2, pr_2
                edge_B, pr_B = edge_1, pr_1

            # 4. Extract ML Features
            features = get_pairwise_features(G, edge_A, edge_B, pr_A, pr_B)

            # 5. Determine Ground Truth Label
            # Create a test graph, remove both edges, and run to completion
            G_test = G.copy()
            G_test.remove_edge(*edge_A)
            G_test.remove_edge(*edge_B)
            
            # The total FAS size for this test is the 2 edges we just manually removed, 
            # plus whatever the classic algorithm needs to remove to make the rest a DAG.
            test_fas_size = 2 + len(pagerankfas(G_test, show_prints=False))
            
            # Label = 1 (Conflict) if the FAS size inflated. Label = 0 (Safe) otherwise.
            label = 1 if test_fas_size > baseline_size else 0
            features["label"] = label
            if label==0:
                safe+=1
            
            dataset.append(features)
            
            # Print progress every 15 pairs to prove it isn't stuck
            if pair_idx % 15 == 0 or pair_idx == len(pairs):
                print(f"  Processed {pair_idx}/{len(pairs)} pairs...")

        graph_duration = time.time() - graph_start_time
        print(f"Graph {i+1} completed in {graph_duration:.2f} seconds. [SAFE = {safe}/45 ({((safe/45)*100):.2f}%)]")

    # Save to CSV
    df = pd.DataFrame(dataset)
    df.to_csv(output_csv, index=False)
    print(f"Data generation complete. Saved {len(df)} pairs to {output_csv}.")
    
    # Print a quick distribution of labels
    print("Label Distribution:")
    print(df["label"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

if __name__ == "__main__":
    generate_dataset(num_graphs=3000, output_csv="tests/datasets/training_data.csv")