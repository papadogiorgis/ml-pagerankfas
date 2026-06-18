# src/pagerankfas/tests/test_classic_pagerankfas.py

import pagerankfas as pfas
import unittest
import networkx as nx

class TestClassicPageRankFAS(unittest.TestCase):
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
        
if __name__ == '__main__':
    unittest.main()