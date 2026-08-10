# scripts/benchmark.py

import os
import sys
import time
import random
import networkx as nx

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pagerankfas.classic import pagerankfas
from pagerankfas.batching import pagerankfas_batching

def run_benchmarks(num_graphs: int = 10, threshold: float = 0.30, batch_size: int = 10):
    """
    Evaluates the Classic PageRankFAS against the ML-Enhanced Batching version
    across random graphs to measure time speedups and FAS size differences.
    """
    print("=" * 75)
    print(f"  PHASE 4 BENCHMARKING: CLASSIC vs. ML-BATCHING (Threshold: {threshold})")
    print("=" * 75)
    
    # We use a fixed seed so your benchmark results are reproducible for your report
    random.seed(42)
    
    total_speedup = 0.0
    total_fas_ratio = 0.0
    
    # Table Header
    print(f"{'Graph Size':<15} | {'Classic Time':<12} | {'Batch Time':<12} | {'Speedup':<8} | {'Classic FAS':<12} | {'Batch FAS':<10}")
    print("-" * 75)

    for i in range(num_graphs):
        # Generate random graphs that are slightly larger to see the time difference clearly
        n = random.randint(3000, 10000)
        p = random.uniform(0.05, 0.15)
        G = nx.gnp_random_graph(n=n, p=p, directed=True)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        
        # Skip DAGs
        if nx.is_directed_acyclic_graph(G):
            continue
            
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        graph_label = f"{nodes}n, {edges}e"

        # -----------------------------------------
        # 1. Run Classic Algorithm
        # -----------------------------------------
        G_classic = G.copy()
        start_classic = time.time()
        fas_classic = pagerankfas(G_classic, show_prints=False)
        time_classic = time.time() - start_classic
        
        # -----------------------------------------
        # 2. Run ML-Batching Algorithm
        # -----------------------------------------
        G_batching = G.copy()
        start_batching = time.time()
        fas_batching = pagerankfas_batching(
            G_batching, 
            threshold=threshold, 
            batch_size=batch_size, 
            show_prints=False
        )
        time_batching = time.time() - start_batching

        # -----------------------------------------
        # 3. Calculate Metrics
        # -----------------------------------------
        speedup = time_classic / time_batching if time_batching > 0 else 0
        fas_diff = len(fas_batching) - len(fas_classic)
        fas_ratio = len(fas_batching) / len(fas_classic) if len(fas_classic) > 0 else 0
        
        # Format the difference string (e.g., "+2" if it got worse, "0" if identical)
        fas_diff_str = f"(+{fas_diff} | {fas_ratio:.2f}x)" if fas_diff > 0 else f"({fas_diff} | {fas_ratio:.2f}x)"
        if fas_diff == 0:
            fas_diff_str = "(Exact)"
            
        total_speedup += speedup
        total_fas_ratio += fas_ratio

        print(f"{graph_label:<15} | {time_classic:>10.2f}s | {time_batching:>9.2f}s | {speedup:>7.1f}x | {len(fas_classic):>11} | {len(fas_batching):>4} {fas_diff_str}")

    # -----------------------------------------
    # 4. Final Aggregations
    # -----------------------------------------
    print("-" * 75)
    overall_speedup = total_speedup / num_graphs if num_graphs > 0 else 0
    overall_fas_ratio = total_fas_ratio / num_graphs if num_graphs > 0 else 0
    print(f"OVERALL SUMMARY:")
    print(f"Average Speedup: {overall_speedup:.2f}x")
    print(f"Average FAS Ratio: {overall_fas_ratio:.2f}x")
    print("=" * 75)

if __name__ == "__main__":
    # You can tweak these parameters to see how the trade-offs behave
    run_benchmarks(num_graphs=10, threshold=0.40, batch_size=10)