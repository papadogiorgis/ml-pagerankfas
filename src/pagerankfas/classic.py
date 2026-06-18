# src/pagerankfas/classic.py

import networkx as nx
from .utils import build_line_digraph, get_scc

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
            N = len(LG.nodes)
            if N == 0:
                continue
            
            #i precalculate out-degrees and successors to avoid
            #doing this inside the 5-iteration loop
            out_degrees = dict(LG.out_degree())
            successors = {n: list(LG.successors(n)) for n in LG.nodes}
            
            #run the pagerank algorithm for 5 iterations with
            #no damping factor (alpha=1.0)
            pagerank_scores = {node: 1.0 / N for node in LG.nodes}
            for _ in range(5):
                next_scores = {node: 0.0 for node in LG.nodes}
                for node in LG.nodes:
                    deg = out_degrees[node]
                    
                    if deg>0:
                        #distribute the current score equally
                        #to all out-neighbors
                        share = pagerank_scores[node] / deg
                        for target in successors[node]:
                            next_scores[target] += share
                    else:
                        #handling sinks
                        #distribute their score uniformly across the graph
                        share = pagerank_scores[node] / N
                        for target in LG.nodes:
                            next_scores[target] += share
                pagerank_scores = next_scores
            
            #find the node in LG (which is an edge in G) that has the
            #highest pagerank score
            max_pagerank_node = max(pagerank_scores, key=pagerank_scores.get)
            
            #add the chosen edge to the fas and remove it from the graph
            fas.append(max_pagerank_node)
            G.remove_edge(*max_pagerank_node)
    
    #after the while loop is finished, so the graph is dag, return the fas list
    return fas