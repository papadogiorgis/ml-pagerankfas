# ML-Batching-PageRankFAS: Machine Learning-Accelerated Feedback Arc Set

ML-Batching-PageRankFAS is a graph algorithm framework designed to solve the **Feedback Arc Set (FAS)** problem on directed graphs. It builds upon the heuristic **PageRankFAS** algorithm—which iteratively identifies and removes cycle-causing edges via PageRank over directed line graphs—and accelerates it using **Machine Learning-guided Batch Edge Removal**.

By training a cost-sensitive Random Forest classifier to predict structural conflicts between candidate edges within Strongly Connected Components (SCCs), ML-Batching-PageRankFAS constructs dynamic conflict graphs and identifies safe batches via Maximum Independent Set (MIS) approximations, achieving substantial runtime speedups while maintaining near-classic FAS solution quality.

---

## Table of Contents
- [Architecture & Methodology](#architecture--methodology)
  - [Classic PageRankFAS](#classic-pagerankfas)
  - [ML-Batching Mechanism](#ml-batching-mechanism)
  - [Pairwise Topological Features](#pairwise-topological-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart & Python API](#quickstart--python-api)
- [Training Pipeline](#training-pipeline)
  - [1. Dataset Generation](#1-dataset-generation)
  - [2. Model Training & Evaluation](#2-model-training--evaluation)
- [Benchmarking & Evaluation Tools](#benchmarking--evaluation-tools)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [License](#license)

---

## Architecture & Methodology

Given a directed graph $G = (V, E)$, a Feedback Arc Set is a subset of edges $E' \subseteq E$ whose removal renders $G$ a Directed Acyclic Graph (DAG). Finding the minimum FAS is NP-hard.

```text
       Input Directed Graph G
                 │
                 ▼
     Decompose into Non-Trivial SCCs
                 │
                 ▼
      Construct Line Digraph L(G)
                 │
                 ▼
   5-Iteration PageRank (Uniform Sinks)
                 │
                 ▼
       Extract Top-K Candidate Edges
                 │
                 ▼
  Pairwise Feature Extraction & ML Conflict Scoring
                 │
                 ▼
    Form Conflict Graph & Solve Approx MIS
                 │
                 ▼
   Batch Remove Independent Edges & Repeat until DAG
```

### Classic PageRankFAS
1. Decomposes the graph into non-trivial Strongly Connected Components ($\vert{}V_{\text{scc}}\vert{} > 1$).
2. Constructs the directed line graph $L(G)$, where each vertex represents an edge $(u, v) \in E$.
3. Runs 5 iterations of PageRank without a damping factor ($\alpha = 1.0$), distributing scores uniformly across sinks.
4. Identifies the edge with the maximum PageRank score and removes it.
5. Repeats this process iteratively until the graph contains no cycles.

### ML-Batching Mechanism
Removing edges one-by-one introduces high computational overhead due to repeated line graph reconstructions and PageRank iterations. ML-Batching accelerates this:
1. **Candidate Pool Selection:** Extracts the top-$K$ scoring edges in each SCC (default $K = 20$).
2. **Conflict Prediction:** Generates all $\binom{K}{2}$ pairs and evaluates them using a Random Forest classifier to estimate the probability $P(\text{conflict})$ that removing both edges simultaneously yields a suboptimal FAS size.
3. **Conflict Graph Construction:** Builds an undirected graph where nodes are candidate edges and edges represent predicted conflicts ($P(\text{conflict}) \ge \tau$).
4. **Maximum Independent Set (MIS):** Solves an approximation of the MIS on the conflict graph to identify mutually non-conflicting edges that can be safely removed in a single batch pass.

### Pairwise Topological Features
For any candidate pair $(e_A, e_B)$ ordered such that $PageRank(e_A) \ge PageRank(e_B)$, the classifier evaluates seven structural features:
- `shared_nodes`: Number of shared endpoints between $e_A$ and $e_B$ ($0$, $1$, or $2$).
- `dist_A_to_B`: Directed shortest path distance from $\text{target}(e_A)$ to $\text{source}(e_B)$ (penalized by $\vert{}V\vert{}$ if unreachable).
- `dist_B_to_A`: Directed shortest path distance from $\text{target}(e_B)$ to $\text{source}(e_A)$ (penalized by $\vert{}V\vert{}$ if unreachable).
- `common_successors`: Count of shared outgoing neighbors ($\vert{}\text{Succ}(\text{target}(e_A)) \cap \text{Succ}(\text{target}(e_B))\vert{}$).
- `common_predecessors`: Count of shared incoming neighbors ($\vert{}\text{Pred}(\text{source}(e_A)) \cap \text{Pred}(\text{source}(e_B))\vert{}$).
- `pr_diff`: Absolute difference in PageRank scores $\vert{}pr_A - pr_B\vert{}$.
- `pr_ratio`: Ratio of PageRank scores $\min(pr_A, pr_B) / \max(pr_A, pr_B)$.

---

## Repository Structure

```text
ml-pagerankfas/
├── .gitignore
├── LICENSE
├── README.md
├── report.pdf
├── requirements.txt
├── src/
│   ├── batching.py              # ML-accelerated batching PageRankFAS implementation
│   ├── classic.py               # Standard iterative PageRankFAS implementation
│   ├── features.py              # Pairwise feature extraction routines
│   ├── utils.py                 # Line graph conversion and SCC extraction helpers
│   └── rf_model.pkl             # Trained Random Forest conflict classifier
└── scripts/
    ├── benchmark_random_graphs.py  # Benchmark suite across Erdős-Rényi parameter grids
    ├── generate_dummy_graphs.py    # CLI tool to generate synthetic graph datasets
    ├── generate_training_data.py   # Multi-core training pair generation and labeling
    ├── plot_benchmark_results.py   # Visualizer for threshold performance tradeoffs
    ├── run_benchmarks_baharev.py   # Benchmark runner for Baharev tournament datasets
    ├── test_web_graphs.py          # Single-graph evaluation script for SNAP/large graphs
    ├── train_model.py              # Model training script with threshold analysis
    └── benchmarks/                 # Output directory for CSV logs and plots
        ├── *.csv
        ├── *.png
        └── notes.txt
```

---

## Installation

### Prerequisites
- Python 3.9+
- Recommended: Virtual environment (`venv` or `conda`)

```bash
# Clone the repository
git clone [https://github.com/your-username/ml-pagerankfas.git](https://github.com/your-username/ml-pagerankfas.git)
cd ml-pagerankfas

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Quickstart & Python API

Both the classic and ML-batching implementations can be executed directly on NetworkX directed graphs:

```python
import networkx as nx
from src.classic import pagerankfas
from src.batching import pagerankfas_batching

# Create or load a directed graph
G = nx.erdos_renyi_graph(n=500, p=0.01, directed=True)

# 1. Compute FAS using Classic PageRankFAS
classic_fas = pagerankfas(G, show_prints=False)
print(f"Classic FAS Size: {len(classic_fas)}")

# 2. Compute FAS using ML-Batching PageRankFAS
# threshold: conflict probability cutoff (lower = conservative/accurate, higher = faster)
# batch_size: top-K candidate edges evaluated per SCC (default = 20)
batch_fas = pagerankfas_batching(
    G, 
    threshold=0.30, 
    batch_size=20, 
    show_prints=False
)
print(f"ML-Batching FAS Size: {len(batch_fas)}")

# Verify the resulting graphs are Directed Acyclic Graphs (DAGs)
G_classic_dag = G.copy()
G_classic_dag.remove_edges_from(classic_fas)
assert nx.is_directed_acyclic_graph(G_classic_dag)

G_batch_dag = G.copy()
G_batch_dag.remove_edges_from(batch_fas)
assert nx.is_directed_acyclic_graph(G_batch_dag)
```

---

## Training Pipeline

### 1. Dataset Generation
Generates synthetic random graphs ($G(n, p)$ with $n \in [30, 100]$ and $p \in [0.05, 0.15]$), computes classic FAS baselines, extracts candidate edge pairs from top PageRank scores, and labels pairs based on whether simultaneous removal degrades total FAS size:

```bash
python scripts/generate_training_data.py
```
- Uses `multiprocessing` to process graphs concurrently across available CPU cores.
- Saves generated pair features and conflict labels to `scripts/datasets/training_data.csv`.

### 2. Model Training & Evaluation
Trains a balanced `RandomForestClassifier` on the pairwise dataset, performs decision threshold analysis, and serializes the model to `src/rf_model.pkl`:

```bash
python scripts/train_model.py
```

---

## Benchmarking & Evaluation Tools

- **Benchmark Random Graphs:**
  Evaluates algorithms across ranges of node sizes ($100$ to $50,000$) and edge densities ($0.001$ to $0.25$), exporting CSV logs and Matplotlib plots:
  ```bash
  python scripts/benchmark_random_graphs.py
  ```

- **Evaluate on Baharev Tournament Graphs:**
  Runs benchmarks across tournament graph collections matching `.edges` with `.mfes` ground-truth minimum feedback sets across thresholds $\tau \in [0.0, 1.0]$ using multi-core execution:
  ```bash
  python scripts/run_benchmarks_baharev.py
  ```

- **Inspect Real-World SNAP Networks:**
  Runs single-instance comparisons on compressed web and P2P networks (`.txt.gz` format):
  ```bash
  python scripts/test_web_graphs.py
  ```

- **Plot Threshold Tradeoffs:**
  Parses benchmark CSV output and generates multiplier performance tradeoff curves:
  ```bash
  python scripts/plot_benchmark_results.py
  ```

- **Generate Synthetic Datasets:**
  Interactive generator for compressed test graphs:
  ```bash
  python scripts/generate_dummy_graphs.py
  ```

---

## Configuration & Hyperparameters

When running `pagerankfas_batching`, key parameters can be configured:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.70` | Conflict decision boundary. |
| `batch_size` | `int` | `20` | Maximum candidate edges extracted from the line graph PageRank scores per SCC. |
| `model_path` | `str` | `None` | Custom path to a serialized Random Forest `.pkl` model. Defaults to `src/rf_model.pkl`. |
| `show_prints` | `bool` | `False` | Enables real-time heartbeat progress logs reporting edges removed, remaining edges, and loop execution timings. |

---

## Thesis & Citation

This project is part of a Bachelor's Thesis at the University of Crete. For the full document, see the [Bachelor's Thesis Report](report.pdf).


---

## License

```text
Copyright (c) 2026 Georgios Papadakis, Dr. Panagiotis Lionakis. All rights reserved.

This software and its documentation are the exclusive property of the authors.
No part of this code may be copied, reproduced, distributed, or modified 
without explicit written permission from the copyright holders.
```
