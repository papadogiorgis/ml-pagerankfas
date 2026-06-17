# src/pagerankfas/utils/line_graph.py

import networkx as nx

def build_line_digraph(scc_graph: nx.DiGraph) -> nx.DiGraph:
    """
    It constructs the directed line graph L(G) from an SCC subgraph.
    Each vertex in L(G) is a tuple representing an edge (u, v) from the
    original graph.
    """
    # NetworkX native line_graph() computes the L(G) for directed graphs
    linegraph = nx.line_graph(scc_graph)
    return linegraph