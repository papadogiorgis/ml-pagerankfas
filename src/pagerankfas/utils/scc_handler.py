# src/pagerankfas/utils/scc_handler.py

import networkx as nx
from typing import List

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