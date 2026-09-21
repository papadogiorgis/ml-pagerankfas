import os
import sys
import time
from pathlib import Path
import networkx as nx

# Add the src folder to the path so modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from classic import pagerankfas
    from batching import pagerankfas_batching
except ImportError:
    print("Error: Could not import pagerankfas modules. Ensure this script is in the 'scripts' folder.")
    sys.exit(1)

def calc_mult(base, target):
    """Calculates ratio avoiding division by zero."""
    return base / target if target > 0 else 0.0

def main():
    print("--- Single Graph Evaluation ---")
    
    # Define dataset path
    # dataset_path = Path(__file__).resolve().parent / "datasets" / "web-Google.txt.gz"
    # dataset_path = Path(__file__).resolve().parent / "datasets" / "web-NotreDame.txt.gz"
    dataset_path = Path(__file__).resolve().parent / "datasets" / "p2p-Gnutella30.txt.gz"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Load the graph
    print(f"\nLoading graph from {dataset_path}...")
    load_start = time.time()
    # NetworkX automatically handles .gz files. Comments typically start with '#' in SNAP datasets
    G = nx.read_edgelist(dataset_path, create_using=nx.DiGraph, nodetype=int, comments='#')
    print(f"Graph loaded in {time.time() - load_start:.2f} seconds.")
    print(f"Graph Stats: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

    # Run ML-Batching PageRankFAS first
    threshold = 0.7
    print(f"\n[1/2] Running ML-Batching PageRankFAS (threshold={threshold})...")
    start_t = time.time()
    batching_fas = pagerankfas_batching(G.copy(), threshold=threshold, show_prints=True)
    batching_time = time.time() - start_t
    batching_size = len(batching_fas)

    print(f"  -> Batching FAS Size: {batching_size:,}")
    print(f"  -> Batching Time:     {batching_time:.4f} seconds")

    # Run Classic PageRankFAS second
    print("\n[2/2] Running Classic PageRankFAS...")
    start_t = time.time()
    classic_fas = pagerankfas(G.copy(), show_prints=True)
    classic_time = time.time() - start_t
    classic_size = len(classic_fas)
    
    print(f"  -> Classic FAS Size: {classic_size:,}")
    print(f"  -> Classic Time:     {classic_time:.4f} seconds")

    # Calculate Multipliers
    size_mult = calc_mult(batching_size, classic_size)
    time_mult = calc_mult(batching_time, classic_time)

    # Print Final Summary
    print("\n--- Final Results & Multipliers ---")
    print(f"Classic Size:  {classic_size:,}")
    print(f"Batching Size: {batching_size:,}")
    print(f"Size Multiplier (Batching / Classic): {size_mult:.4f}x")
    print("-" * 40)
    print(f"Classic Time:  {classic_time:.4f}s")
    print(f"Batching Time: {batching_time:.4f}s")
    print(f"Time Multiplier (Batching / Classic): {time_mult:.4f}x")
    
if __name__ == '__main__':
    main()