# src/pagerankfas/tests/test_classic_pagerankfas.py

import pagerankfas as pfas
import unittest
import networkx as nx
import os

class TestClassicPageRankFAS(unittest.TestCase):
    # the first 2 test take about 0.002 seconds to run
    # python -m unittest -k test_simple_cycle -k test_already_dag
    def test_simple_cycle(self):
        """
        Test that the algorithm breaks a simple 3 node cycle.
        """
        G = nx.DiGraph()
        G.add_edges_from([(1,2), (2,3), (3,1)])
        
        #first, i make sure that the Graph has indeed cycles
        self.assertFalse(nx.is_directed_acyclic_graph(G))
        
        #then i run the algorithm and take the fas list
        fas = pfas.pagerankfas(G)
        
        #the fas list should have exactly 1 edge to break the cycle
        self.assertEqual(len(fas), 1)
        
        #remove the identified fas from the graph
        G.remove_edges_from(fas)
        #and after the removal make sure that the graph is dag now
        self.assertTrue(nx.is_directed_acyclic_graph(G))
        
    def test_already_dag(self):
        """
        Test that the algorithm returns an empty list for DAGs.
        """
        G = nx.DiGraph()
        G.add_edges_from([(1,2), (2,3), (1,3)])
        
        #i run the algorithm and take the fas list
        fas = pfas.pagerankfas(G)
        
        #it should return an empty list since there are no cycles
        self.assertEqual(len(fas), 0)
    
    def test_random_graph(self):
        """
        Test on a small random directed graph.
        """
        #create a random directed graph with 200 nodes
        #it takes about 100 seconds
        # python -m unittest -k test_random_graph
        G = nx.gnp_random_graph(200, 0.1, directed=True, seed=42)
        if not nx.is_directed_acyclic_graph(G):
            fas = pfas.pagerankfas(G)
            G.remove_edges_from(fas)
            self.assertTrue(nx.is_directed_acyclic_graph(G))
        
    def test_cit_hepph(self):
        #THIS TAKES FOREVER TO RUN, HAVENT COMPLETED IT YET
        """
        Loads the cit-HepPh dataset, verifies it contains cycles, runs
        pagerankfas, and it validates that the remaining graph is dag.
        """
        #get the directory where this file lives
        currdir = os.path.dirname(os.path.abspath(__file__))
        #find the file path
        file_path = os.path.normpath(os.path.join(currdir, "datasets", 
                                                  "cit-HepPh.txt.gz"))
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Could not find {file_path}"
            )
        
        print("Starting extensive test...\n1. Loading the graph...")
        #snap uses # for comments
        #create_using=nx.DiGraph ensures its directed
        G = nx.DiGraph()
        G = nx.read_edgelist(file_path, comments='#', create_using=nx.DiGraph, 
                             nodetype=int)
        print(f"--> Loaded graph with {G.number_of_nodes()} nodes and ",
              f"{G.number_of_edges()} edges.")
        
        print("2. Checking if the initial graph is a DAG...")
        is_it_dag = nx.is_directed_acyclic_graph(G)
        print(f"---> Is initial graph a DAG? {is_it_dag}")
        if is_it_dag:
            print("The graph is already a DAG! No need to run PageRankFAS.")
            return
        
        print("3. Running PageRankFAS algorithm...")
        #enable the heartbeat prints for this massive graph
        fas = pfas.pagerankfas(G, show_prints=True)
        print(f"--> Algorithm identified {len(fas)} edges to remove!")
        
        print("4. Removing Feedback Arc Set edges from the graph...")
        G.remove_edges_from(fas)
        
        print("5. Verifying if the modified graph is now a DAG...")
        is_final_dag = nx.is_directed_acyclic_graph(G)
        print(f"---> Is modified graph a DAG? {is_final_dag}")
        
        if is_final_dag:
            print("\nSUCCESS: The graph has been completely broken out of its cycles.")
        else:
            print("\nFAILURE: Cycles still remain in the graph.")
        
if __name__ == '__main__':
    unittest.main()