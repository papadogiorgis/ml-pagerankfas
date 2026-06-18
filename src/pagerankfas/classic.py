# src/pagerankfas/classic.py

import networkx as nx
from .utils import build_line_digraph
from .utils import get_scc

def pagerankfas(graph: nx.DiGraph) -> list:
    """
    Computes a Feedback Arc Set (FAS) for a directed graph using the (classic)
    PageRankFAS algorithm.
    """
    #i copy the original graph to avoid mutating it
    G = graph.copy()
    fas = [] # initialize the fas list
    
    #the loop runs until the graph has no more cycles
    while not nx.is_directed_acyclic_graph(G):
        #first, get all SCCs
        SCCs = get_scc(G)
        
        #process each scc
        for component in SCCs:
            #build the line graph of the SCC
            LG = build_line_digraph(component)
            #check if the line graph is empty, if yes skip this scc
            if len(LG.nodes)==0:
                continue
            
            #run the pagerank algorithm for 5 iterations with
            #no damping factor (alpha=1.0)
            pagerank_scores = nx.pagerank(LG, alpha=1.0, max_iter=5)
            
            #find the node in LG (which is an edge in G) that has the
            #highest pagerank score
            max_pagerank_node = max(pagerank_scores, key=pagerank_scores.get)
            
            #add the chosen edge to the fas and remove it from the graph
            fas.append(max_pagerank_node)
            G.remove_edge(*max_pagerank_node)
    
    #after the while loop is finished, so the graph is dag, return the fas list
    return fas