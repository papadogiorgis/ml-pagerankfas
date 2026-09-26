# scripts/generate_dummy_graphs.py

import networkx as nx
import os

def main():
    print("--- Dummy Graph Dataset Generator ---")
    
    # 1. Ask for the number of nodes
    while True:
        try:
            n = int(input("Enter the number of nodes: "))
            if n > 0: 
                break
            print("Number of nodes must be greater than 0.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # 2. Ask for the density/probability
    while True:
        try:
            p = float(input("Enter the density/probability (0.0 to 1.0): "))
            if 0.0 <= p <= 1.0: 
                break
            print("Probability must be between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a float (e.g., 0.05).")

    # 3. Ask for the graph type
    while True:
        g_type = input("Enter graph type ('erdos' for Erdos-Renyi, 'random' for Random): ").strip().lower()
        if g_type in ['erdos', 'random']:
            break
        print("Invalid input. Please enter 'erdos' or 'random'.")

    print(f"\nGenerating {g_type} directed graph with {n} nodes and p={p}...")
    
    # Generate the directed graph based on user choice.
    # Note: In NetworkX, erdos_renyi_graph and gnp_random_graph are mathematically 
    # identical G_{n,p} models, but we use fast_gnp for the "random" choice as it is 
    # highly optimized for sparse graphs.
    if g_type == 'erdos':
        G = nx.erdos_renyi_graph(n, p, directed=True)
    else:
        G = nx.fast_gnp_random_graph(n, p, directed=True)

    print(f"Graph generated: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")

    # 4. Save to .txt.gz
    # Ensure a datasets folder exists in the current directory
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"dummy_{g_type}_{n}.txt.gz"
    filepath = os.path.join(output_dir, filename)

    print(f"Compressing and saving dataset to {filepath}...")
    
    # nx.write_edgelist automatically compresses the file if it ends in .gz
    # data=False ensures we only write "source target" pairs without weight dictionaries,
    # which perfectly matches standard SNAP dataset formats.
    nx.write_edgelist(G, filepath, data=False)
    
    print("Done! You can now test your PageRankFAS script.")
    print(f"Update your target script to use: dataset_path = Path(__file__).resolve().parent / 'datasets' / '{filename}'")

if __name__ == "__main__":
    main()