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
    from pagerankfas.classic import pagerankfas
    from pagerankfas.batching import pagerankfas_batching
except ImportError:
    print("Warning: Could not import pagerankfas modules. Ensure this script is at the root of the project.")

def process_graph(graph_data):
    """
    Worker function to process a single graph.
    graph_data: tuple of (graph_name, edges_file_path, mfes_file_path)
    """
    name, edges_file, mfes_file = graph_data
    
    # 1. Exact Size
    # The .mfes file contains the exact feedback edge set, one edge per line.
    exact_size = 0
    with open(mfes_file, 'r') as f:
        exact_size = sum(1 for line in f if line.strip())
        
    # Build graph
    # Baharev .edges are standard edge lists
    G = nx.read_edgelist(edges_file, create_using=nx.DiGraph, nodetype=int)
    
    # 2. Classic PageRankFAS
    start_t = time.time()
    classic_fas = pagerankfas(G, show_prints=False)
    classic_time = time.time() - start_t
    classic_size = len(classic_fas)
    
    # 3. Batching PageRankFAS - Threshold 0.2
    start_t = time.time()
    b2_fas = pagerankfas_batching(G, threshold=0.2, show_prints=False)
    b2_time = time.time() - start_t
    b2_size = len(b2_fas)
    
    # 4. Batching PageRankFAS - Threshold 0.3
    start_t = time.time()
    b3_fas = pagerankfas_batching(G, threshold=0.3, show_prints=False)
    b3_time = time.time() - start_t
    b3_size = len(b3_fas)
    
    # 5. Batching PageRankFAS - Threshold 0.4
    start_t = time.time()
    b4_fas = pagerankfas_batching(G, threshold=0.4, show_prints=False)
    b4_time = time.time() - start_t
    b4_size = len(b4_fas)
    
    # Calculate multipliers (avoiding division by zero)
    def calc_mult(base, target):
        return base / target if target > 0 else 0.0
        
    m_s_2 = calc_mult(classic_size, b2_size)
    m_t_2 = calc_mult(classic_time, b2_time)
    
    m_s_3 = calc_mult(classic_size, b3_size)
    m_t_3 = calc_mult(classic_time, b3_time)
    
    m_s_4 = calc_mult(classic_size, b4_size)
    m_t_4 = calc_mult(classic_time, b4_time)
    
    return [
        name, exact_size, classic_size, classic_time,
        b2_size, m_s_2, b2_time, m_t_2,
        b3_size, m_s_3, b3_time, m_t_3,
        b4_size, m_s_4, b4_time, m_t_4
    ]

def main():
    print("--- PageRankFAS Benchmarking Tool ---")
    data_dir = input("Enter the path to the dataset folder containing the Baharev graphs (e.g., data/benchmark_mfes): ").strip()
    
    dataset_path = Path(data_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        print("Error: Directory not found.")
        return

    print("Scanning for graphs (*.edges and *.mfes pairs)...")
    graph_files = []
    # Recursively find all edge files and match them with mfes files
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

    # Sort files to ensure consistent runs (e.g., by name) and slice the requested amount
    graph_files.sort(key=lambda x: x[0])
    graphs_to_run = graph_files[:num_runs]

    output_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'benchmark_results.csv'))
    headers = [
        "Graph_Name", "Exact_FAS_Size", "Classic_Size", "Classic_Time_s",
        "Batch0.2_Size", "Mult_Size_0.2", "Batch0.2_Time_s", "Mult_Time_0.2",
        "Batch0.3_Size", "Mult_Size_0.3", "Batch0.3_Time_s", "Mult_Time_0.3",
        "Batch0.4_Size", "Mult_Size_0.4", "Batch0.4_Time_s", "Mult_Time_0.4"
    ]

    # Initialize CSV with headers
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    print(f"\n[0.00s] Starting benchmark on {num_runs} graphs...")
    global_start = time.time()
    completed = 0
    
    # To keep a running sum of the multiplier columns for the final averages
    sums = {5: 0.0, 7: 0.0, 9: 0.0, 11: 0.0, 13: 0.0, 15: 0.0}

    # Use all available CPU cores
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_graph, g): g for g in graphs_to_run}
        
        for future in as_completed(futures):
            try:
                # Step 2: Receive results
                result = future.result()
                
                # Write row immediately so data isn't lost if the script crashes
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)
                
                # Accumulate multiplier columns
                sums[5] += result[5]
                sums[7] += result[7]
                sums[9] += result[9]
                sums[11] += result[11]
                sums[13] += result[13]
                sums[15] += result[15]
                
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
    
    final_row = [""] * 16
    final_row[0] = "RESULTS"
    if completed > 0:
        final_row[5] = sums[5] / completed
        final_row[7] = sums[7] / completed
        final_row[9] = sums[9] / completed
        final_row[11] = sums[11] / completed
        final_row[13] = sums[13] / completed
        final_row[15] = sums[15] / completed

    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(final_row)
        
    print(f"Benchmark finished successfully. Data saved to '{output_csv}'.")

if __name__ == '__main__':
    # Fix for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
