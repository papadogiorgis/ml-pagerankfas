# src/utils.py

import networkx as nx
from typing import List

def build_line_digraph(scc_graph: nx.DiGraph) -> nx.DiGraph:
    """
    It constructs the directed line graph L(G) from an SCC subgraph.
    Each vertex in L(G) is a tuple representing an edge (u, v) from the
    original graph.
    """
    # NetworkX native line_graph() computes the L(G) for directed graphs
    linegraph = nx.line_graph(scc_graph)
    return linegraph

def get_scc(graph: nx.DiGraph) -> List[nx.DiGraph]:
    """
    Decomposes a directed graph into its Strongly Connected Components (SCCs).
    It also filters out single-node components that cannot contain any cycles.
    """
    SCCs = []
    
    # extract SCCs
    for component in nx.strongly_connected_components(graph):
        # a component must have at least 2 nodes to form a cycle in a graph
        if len(component)>1:
            # create a subgraph (this SCC) containing the edges between
            # these nodes
            subgraph = graph.subgraph(component).copy()
            SCCs.append(subgraph)
    
    # return the SCCs' list
    return SCCs