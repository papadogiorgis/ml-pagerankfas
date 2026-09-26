# src/classic.py

import networkx as nx
import time
from utils import build_line_digraph, get_scc

def pagerankfas(graph: nx.DiGraph, show_prints: bool = False) -> list:
    """
    Computes a Feedback Arc Set (FAS) for a directed graph using the (classic)
    PageRankFAS algorithm.
    """
    #i copy the original graph to avoid mutating it
    G = graph.copy()
    fas = [] # initialize the fas list
    
    #remove self loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        fas.extend(self_loops)
        G.remove_edges_from(self_loops)
        if show_prints:
            print(f"[Init] Stripped {len(self_loops)} self-loops",
                  f"from the graph.")
    
    #initialize print variables
    iteration_num = 0
    #initialize time variables
    start_time = time.time()
    last_print_time = start_time
    
    #the loop runs until the graph has no more cycles
    while not nx.is_directed_acyclic_graph(G):
        iteration_num += 1
        
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
        
        #heartbeat print every 10 iterations with timing
        #if show_prints and (iteration_num % 10 == 0):
        if show_prints:
            current_time = time.time()
            total_time = current_time - start_time
            time_since_last_print = current_time - last_print_time
            last_print_time = current_time
            print(f"[PageRankFAS Heartbeat] Loop: {iteration_num} |",
                  f"FAS Size: {len(fas)} |",
                  f"Edges left in G: {G.number_of_edges()} |",
                  f"Total Time: {total_time:.2f}s",
                  f"(from previous print: {time_since_last_print:.2f}s)")
    
    #after the while loop is finished, so the graph is dag, return the fas list
    return fas