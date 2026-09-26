# scripts/benchmark_random_graphs.py

import os
import sys
import time
import csv
import statistics
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src folder to the path so modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from classic import pagerankfas
    from batching import pagerankfas_batching
except ImportError:
    print("Error: Could not import pagerankfas modules. Ensure this script is in the 'scripts' folder.")
    sys.exit(1)

def calc_mult(classic_val, batching_val):
    """Calculates multiplier (Classic / ML-Batching) avoiding division by zero."""
    return batching_val / classic_val if classic_val > 0 else 0.0

def generate_plots(csv_path, output_dir):
    """Generates the requested plots using the mean values from the CSV."""
    print(f"\nGenerating plots from {csv_path}...")
    plot_dir = output_dir
    plot_dir.mkdir(exist_ok=True)
    
    # Load data using Pandas
    df = pd.read_csv(csv_path)
    
    # Filter for mean values only
    df_mean = df[df['is_it_mean'] == 1].copy()
    
    # Calculate FAS percentages: (FAS size / total edges) * 100
    df_mean['fas_perc_ml'] = (df_mean['fas_size_ml'] / df_mean['num_edges']) * 100
    df_mean['fas_perc_classic'] = (df_mean['fas_size_classic'] / df_mean['num_edges']) * 100

    nodes = df_mean['graph_size'].unique()
    densities = df_mean['density'].unique()

    # 1. Plots grouped by number of nodes (X-axis = Density)
    for n in nodes:
        subset = df_mean[df_mean['graph_size'] == n].sort_values(by='density')
        
        # Plot 1a: Density vs FAS Percentage
        plt.figure(figsize=(8, 5))
        plt.plot(subset['density'], subset['fas_perc_ml'], marker='o', label='ML-Batching')
        plt.plot(subset['density'], subset['fas_perc_classic'], marker='s', label='Classic')
        plt.title(f'FAS Percentage vs Density (Nodes: {n})')
        plt.xlabel('Density')
        plt.ylabel('FAS Percentage (%)')
        plt.ylim(0, 100) # Percentage 0-100
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plot_dir / f'nodes_{n}_fas_perc.png')
        plt.close()
        
        # Plot 1b: Density vs Multipliers
        plt.figure(figsize=(8, 5))
        plt.plot(subset['density'], subset['size_multiplier'], marker='o', label='Size Multiplier (C/ML)')
        plt.plot(subset['density'], subset['time_multiplier'], marker='s', label='Time Multiplier (C/ML)')
        plt.title(f'Multipliers vs Density (Nodes: {n})')
        plt.xlabel('Density')
        plt.ylabel('Multiplier (x.0)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plot_dir / f'nodes_{n}_multipliers.png')
        plt.close()

    # 2. Plots grouped by density (X-axis = Number of Nodes)
    for p in densities:
        subset = df_mean[df_mean['density'] == p].sort_values(by='graph_size')
        
        # Plot 2a: Nodes vs FAS Percentage
        plt.figure(figsize=(8, 5))
        plt.plot(subset['graph_size'], subset['fas_perc_ml'], marker='o', label='ML-Batching')
        plt.plot(subset['graph_size'], subset['fas_perc_classic'], marker='s', label='Classic')
        plt.title(f'FAS Percentage vs Number of Nodes (Density: {p})')
        plt.xlabel('Number of Nodes')
        plt.ylabel('FAS Percentage (%)')
        plt.ylim(0, 100) # Percentage 0-100
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plot_dir / f'density_{p}_fas_perc.png')
        plt.close()
        
        # Plot 2b: Nodes vs Multipliers
        plt.figure(figsize=(8, 5))
        plt.plot(subset['graph_size'], subset['size_multiplier'], marker='o', label='Size Multiplier (C/ML)')
        plt.plot(subset['graph_size'], subset['time_multiplier'], marker='s', label='Time Multiplier (C/ML)')
        plt.title(f'Multipliers vs Number of Nodes (Density: {p})')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Multiplier (x.0)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plot_dir / f'density_{p}_multipliers.png')
        plt.close()
        
    print(f"Plots saved successfully in: {plot_dir}")


def main():
    print("--- Automated Random Graph Benchmark ---")
    
    node_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000] 
    densities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25] 
    
    runs_per_config = 3
    threshold = 0.7
    output_dir = Path(__file__).resolve().parent / "benchmarks/random_graphs"
    output_dir.mkdir(exist_ok=True)
    output_csv = output_dir / "random_benchmark_results.csv"
    
    file_exists = output_csv.exists()
    
    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Added num_edges to calculate the exact percentage of edges placed in the FAS later
        if not file_exists:
            writer.writerow([
                "graph_size", "density", "num_edges", "fas_size_ml", "fas_size_classic", 
                "size_multiplier", "time_ml", "time_classic", "time_multiplier", "is_it_mean"
            ])
            
        for n in node_sizes:
            for p in densities:
                print(f"\nBenchmarking configuration: {n} nodes | {p} density")
                
                # Temporary lists for mean calculation
                edges_counts = []
                b_sizes, c_sizes, s_mults = [], [], []
                b_times, c_times, t_mults = [], [], []
                
                for run in range(runs_per_config):
                    G = nx.fast_gnp_random_graph(n, p, directed=True)
                    m = G.number_of_edges()
                    
                    start_t = time.time()
                    batching_fas = pagerankfas_batching(G.copy(), threshold=threshold, show_prints=False)
                    b_time = time.time() - start_t
                    b_size = len(batching_fas)
                    
                    start_t = time.time()
                    classic_fas = pagerankfas(G.copy(), show_prints=False)
                    c_time = time.time() - start_t
                    c_size = len(classic_fas)
                    
                    s_mult = calc_mult(c_size, b_size)
                    t_mult = calc_mult(c_time, b_time)
                    
                    # Store metrics for aggregation
                    edges_counts.append(m)
                    b_sizes.append(b_size)
                    c_sizes.append(c_size)
                    s_mults.append(s_mult)
                    b_times.append(b_time)
                    c_times.append(c_time)
                    t_mults.append(t_mult)
                    
                    writer.writerow([
                        n, p, m, b_size, c_size, s_mult, b_time, c_time, t_mult, 0
                    ])
                    print(f"  Run {run+1}/3 complete. ML Time: {b_time:.4f}s | Classic Time: {c_time:.4f}s")
                
                # Write aggregated means row (is_it_mean = 1)
                writer.writerow([
                    n, p,
                    statistics.mean(edges_counts),
                    statistics.mean(b_sizes),
                    statistics.mean(c_sizes),
                    statistics.mean(s_mults),
                    statistics.mean(b_times),
                    statistics.mean(c_times),
                    statistics.mean(t_mults),
                    1
                ])
                f.flush()
                
    print(f"\nAll benchmarks complete. Results appended to: {output_csv}")
    
    # Generate the requested plots using Pandas and Matplotlib
    generate_plots(output_csv, output_dir)

if __name__ == "__main__":
    main()