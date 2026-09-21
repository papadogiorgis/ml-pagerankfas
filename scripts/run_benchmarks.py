import os
import sys
import time
import csv
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx

# Add the src folder to the path so modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from classic import pagerankfas
    from batching import pagerankfas_batching
except ImportError:
    print("Warning: Could not import pagerankfas modules. Ensure this script is at the root of the project.")

# Define target thresholds
THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

def calc_mult(base, target):
    """Calculates ratio avoiding division by zero."""
    return base / target if target > 0 else 0.0

def process_graph(graph_data):
    """
    Worker function to process a single graph.
    graph_data: tuple of (graph_name, edges_file_path, mfes_file_path)
    """
    name, edges_file, mfes_file = graph_data
    
    # 1. Exact Size
    with open(mfes_file, 'r') as f:
        exact_size = sum(1 for line in f if line.strip())
        
    # Build graph
    G = nx.read_edgelist(edges_file, create_using=nx.DiGraph, nodetype=int)
    
    # 2. Classic PageRankFAS
    start_t = time.time()
    classic_fas = pagerankfas(G, show_prints=False)
    classic_time = time.time() - start_t
    classic_size = len(classic_fas)
    
    row = [name, exact_size, classic_size, classic_time]
    
    # 3. Batching PageRankFAS across all thresholds
    for th in THRESHOLDS:
        start_t = time.time()
        b_fas = pagerankfas_batching(G, threshold=th, show_prints=False)
        b_time = time.time() - start_t
        b_size = len(b_fas)
        
        m_s = calc_mult(b_size, classic_size)
        m_t = calc_mult(b_time, classic_time)
        
        row.extend([b_size, m_s, b_time, m_t])
        
    return row

def main():
    print("--- PageRankFAS Benchmarking Tool ---")
    dataset_path = Path(__file__).resolve().parent / "datasets/baharev/tournament"
    if not dataset_path.exists() or not dataset_path.is_dir():
        print("Error: Directory not found.")
        return

    print("Scanning for graphs (*.edges and *.mfes pairs)...")
    graph_files = []
    for edges_file in dataset_path.rglob("*.edges"):
        mfes_file = edges_file.with_suffix('.mfes')
        if mfes_file.exists():
            graph_files.append((edges_file.stem, str(edges_file), str(mfes_file)))
    
    total_found = len(graph_files)
    print(f"Found {total_found} valid graph pairs.")
    if total_found == 0:
        return

    # Step 1: Prompt for number of graphs
    while True:
        try:
            num_runs = int(input(f"How many graphs do you want to run? (1-{total_found}): "))
            if 1 <= num_runs <= total_found:
                break
            print(f"Please enter a number between 1 and {total_found}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    graph_files.sort(key=lambda x: x[0])
    graphs_to_run = graph_files[:num_runs]

    output_dir = Path(__file__).resolve().parent / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "benchmark_results_tournament_0p0_to_1p0.csv"
    
    # Generate dynamic headers
    headers = ["Graph_Name", "Exact_FAS_Size", "Classic_Size", "Classic_Time_s"]
    for th in THRESHOLDS:
        headers.extend([
            f"Batch{th}_Size", f"Mult_Size_{th}",
            f"Batch{th}_Time_s", f"Mult_Time_{th}"
        ])

    # Indices of the multiplier columns (Mult_Size and Mult_Time for each threshold)
    multiplier_indices = []
    base_idx = 4
    for _ in THRESHOLDS:
        multiplier_indices.extend([base_idx + 1, base_idx + 3])
        base_idx += 4

    # Initialize CSV with headers
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    print(f"\n[0.00s] Starting benchmark on {num_runs} graphs...")
    global_start = time.time()
    completed = 0
    
    # Running sums for multiplier columns
    sums = {idx: 0.0 for idx in multiplier_indices}

    # Use all available CPU cores
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_graph, g): g for g in graphs_to_run}
        
        for future in as_completed(futures):
            try:
                # Step 2: Receive results
                result = future.result()
                
                # Write row immediately
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)
                
                # Accumulate multiplier columns
                for idx in multiplier_indices:
                    sums[idx] += result[idx]
                
                completed += 1
                
                # Step 3: ETA & Terminal Printing
                t_elapsed = time.time() - global_start
                mean_time_per_graph = t_elapsed / completed
                graphs_left = num_runs - completed
                eta = mean_time_per_graph * graphs_left
                pct = (completed / num_runs) * 100
                
                print(f"[{t_elapsed:.2f}s] Completed: {completed}/{num_runs} ({pct:.1f}%) | ETA: {eta:.1f}s")
                
            except Exception as e:
                print(f"Error processing a graph: {e}")

    # Step 4: Final 'RESULTS' Row
    final_time = time.time() - global_start
    print(f"\n[{final_time:.2f}s] All requested graphs completed. Writing final calculations...")
    
    final_row = [""] * len(headers)
    final_row[0] = "RESULTS"
    if completed > 0:
        for idx in multiplier_indices:
            final_row[idx] = sums[idx] / completed

    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(final_row)
        
    print(f"Benchmark finished successfully. Data saved to '{output_csv}'.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()