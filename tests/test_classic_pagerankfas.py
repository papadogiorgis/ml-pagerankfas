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
        fas = pfas.pagerankfas(G, show_prints=True)
        
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
        fas = pfas.pagerankfas(G, show_prints=True)
        
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
            fas = pfas.pagerankfas(G, show_prints=True)
            G.remove_edges_from(fas)
            self.assertTrue(nx.is_directed_acyclic_graph(G))
        
    def test_cit_hepph(self):
        #takes about 6 minutes to run
        # python -m unittest -k test_cit_hepph
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
    
    def test_benchmark_edges_file(self):
        """
        Loads a download benchmark .edges file, ensures it contains cycles,
        runs pagerankfas, and validates that the remaining graph is a DAG.
        """
        #get the directory where this file lives
        currdir = os.path.dirname(os.path.abspath(__file__))
        #find the file path
        filenames = ["erdos_renyi_n_100_c_7_seed_94.edges", "erdos_renyi_n_95_c_7_seed_68.edges", "erdos_renyi_n_100_c_7_seed_3.edges", "erdos_renyi_n_100_c_7_seed_57.edges",
                     "erdos_renyi_n_95_c_7_seed_87.edges", "erdos_renyi_n_95_c_7_seed_82.edges", "erdos_renyi_n_100_c_7_seed_96.edges", "erdos_renyi_n_100_c_7_seed_17.edges",
                     "erdos_renyi_n_95_c_7_seed_98.edges", "erdos_renyi_n_100_c_7_seed_87.edges", "erdos_renyi_n_100_c_7_seed_32.edges", "erdos_renyi_n_75_c_9_seed_53.edges",
                     "erdos_renyi_n_75_c_9_seed_3.edges", "erdos_renyi_n_100_c_7_seed_44.edges", "erdos_renyi_n_90_c_7_seed_73.edges", "erdos_renyi_n_100_c_7_seed_98.edges",
                     "erdos_renyi_n_100_c_7_seed_43.edges", "erdos_renyi_n_75_c_9_seed_63.edges", "erdos_renyi_n_95_c_7_seed_40.edges", "erdos_renyi_n_75_c_9_seed_57.edges",
                     "tournament_n_34_seed_78.edges", "tournament_n_34_seed_75.edges", "tournament_n_34_seed_48.edges", "tournament_n_34_seed_94.edges",
                     "tournament_n_33_seed_86.edges", "tournament_n_33_seed_39.edges", "tournament_n_34_seed_67.edges", "tournament_n_32_seed_18.edges",
                     "tournament_n_34_seed_58.edges", "tournament_n_34_seed_31.edges", "tournament_n_34_seed_69.edges", "tournament_n_34_seed_24.edges",
                     "tournament_n_34_seed_43.edges", "tournament_n_34_seed_87.edges", "tournament_n_34_seed_83.edges", "tournament_n_34_seed_11.edges",
                     "tournament_n_33_seed_95.edges", "tournament_n_33_seed_64.edges", "tournament_n_33_seed_73.edges", "tournament_n_34_seed_68.edges"]

        correct_pagerankfas = [167, 180, 189, 201, 189, 188, 204, 185, 182, 199, 184, 208, 217, 191, 181, 186, 201, 211, 188, 221,
                               211, 198, 204, 201, 189, 187, 200, 175, 194, 196, 197, 197, 194, 206, 192, 201, 182, 178, 184, 201]
        
        for i in range(40):
            file_path= os.path.normpath(os.path.join(currdir,"..","temp","baharev",filenames[i]))
            #create_using=nx.DiGraph ensures its directed
            G = nx.DiGraph()
        
            #open .edges file
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    #skip empty lines or comment lines
                    if not line or line.startswith("#"):
                        continue
                    #split line by whitespace
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        G.add_edge(u,v)
                        
            fas = pfas.pagerankfas(G)
            if len(fas) == correct_pagerankfas[i]:
                dif = "(CORRECT)"
            else:
                if len(fas)>correct_pagerankfas[i]:
                    dif = f"(+{len(fas)-correct_pagerankfas[i]})"
                else:
                    dif = f"({len(fas)-correct_pagerankfas[i]})"
            print(f"{i+1}. {filenames[i]}: {len(fas)} {dif}")
            G.remove_edges_from(fas)
            is_final_dag = nx.is_directed_acyclic_graph(G)
            
            if not is_final_dag:
                print("\nFAILURE: Cycles still remain in the graph.")
        
if __name__ == '__main__':
    unittest.main()