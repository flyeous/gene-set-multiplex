import abc
import copy
import igraph
from igraph import *
import itertools
import leidenalg as la
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd  
from sklearn import metrics
from tqdm import tqdm
from utils import func

class multiplex_tissue_network:    
    def __init__(self, layer:list, layer_ns:list, gene_set_collection_name, method = "kmeans", self_loop = False):
        self.layer_ns = copy.deepcopy(layer_ns)
        self.layer = copy.deepcopy(layer)
        self.current_method = method
        self.gene_set_collection_name = gene_set_collection_name
        if method != "modified_Jaccard":
            assert not func.filter_check(self.layer, self.gene_set_collection_name, self.current_method), "Different filtered gene sets exist for the tissues!"
        self.adj_lst = self._load_adj(gene_set_collection_name, method = method, self_loop = self_loop)
        self.adj_filtered_lst = None
        self.adj_Jaccard = None
        self.adj_Jaccard = None
        self.graph_lst = None
        self.community_detection = {}
        self.adj_Jaccard = None
        self.multiplex_property = {}
        
    
    def _load_adj(self,  gene_set_collection_name, method, self_loop):
        assert self.layer is not None, "Tissue list is None!"
        assert method == "BGM" or method == "leiden" or method == "modified_Jaccard" or method == "kmeans", "Unsupported method!"
        adj_lst = [layer.gene_set[self.gene_set_collection_name].sim_mat[method] for layer in self.layer]
        ### Exclude self-loops
        if not self_loop:
            adj_lst = func.self_loop_exclude(adj_lst)
        return adj_lst
    
    def _adj_filter_tr(self, adj_mat, q, tr = None):
        adj_mat = func.adj_filter_tr(adj_mat, q = q, tr = tr)
        return adj_mat
    
    def call_adj_filter_tr(self, q, tr = None):
        self.adj_filtered_lst = [self._adj_filter_tr(adj, q, tr) for adj in self.adj_lst]
        self.tr = tr
        self.q = q
        

    def load_Jaccard(self,self_loop = False, num_core = 30):
        assert self.layer is not None, "Tissue list is None!"
        self.layer[0].call_Jaccard(self.gene_set_collection_name, num_core = num_core)
        adj_Jaccard = self.layer[0].gene_set[self.gene_set_collection_name].sim_mat['Jaccard']
        ### Exclude self-loops
        if not self_loop:
            N = adj_Jaccard.shape[0]
            iden_mat = np.diag(np.repeat(1, N))
            adj_Jaccard = adj_Jaccard - iden_mat 
        if self.current_method == "modified_Jaccard":
            self.adj_Jaccard = adj_Jaccard
        else:
            filter_index = self.layer[0].gene_set[self.gene_set_collection_name].filter_index[self.current_method]
            self.adj_Jaccard = adj_Jaccard[filter_index,][:,filter_index]
        
        
        
    def community_detection_MVP(self, min_cluster, MVP_para = {'weights': "weight", 'n_iterations':-1, 'seed':123}):
        self.graph_lst = [Graph.Weighted_Adjacency(adj_tr, mode="undirected") for adj_tr in self.adj_filtered_lst]
        membership, improv = la.find_partition_multiplex(
                       self.graph_lst, la.ModularityVertexPartition, **MVP_para)
        membership = np.array(membership)
        index, num = np.unique(membership, return_counts = True)
        membership[membership >= index[np.where(num <= min_cluster)[0][0]]] = index[np.where(num <= min_cluster)[0][0]]
        self.community_detection['MVP'] = (membership, improv)
        
    def community_detection_MVP_per_layer(self, Jaccard = False, MVP_para = {'weights': "weight", 'n_iterations':-1, 'seed':123}):
        self.graph_lst = [Graph.Weighted_Adjacency(adj_tr, mode="undirected") for adj_tr in self.adj_filtered_lst]
        if Jaccard is True:
            Jaccard_layer = copy.deepcopy(self.adj_Jaccard)
            partition_Jaccard = la.find_partition(Graph.Weighted_Adjacency(Jaccard_layer, mode="directed"), la.ModularityVertexPartition, **MVP_para)
            self.community_detection['MVP_Jaccard'] = partition_Jaccard
        partitions = []
        for graph in tqdm(self.graph_lst):
            partitions.append(la.find_partition(graph, la.ModularityVertexPartition, **MVP_para))
        self.community_detection['MVP_per_layer'] = partitions
        

        
    def NMI_layers(self, ):
        assert self.community_detection['MVP_per_layer'] is not None and self.community_detection['MVP_Jaccard'] is not None , "community_detection is not complete!"
        partition_lst = self.community_detection['MVP_per_layer']
        partition_lst = [item.membership for item in partition_lst]
        ### add Jaccard membership
        partition_lst.insert(0,  self.community_detection['MVP_Jaccard'].membership)
        NMI = np.array([metrics.normalized_mutual_info_score(mem_A, mem_B) \
               for mem_A in partition_lst for mem_B in partition_lst]).reshape(len(partition_lst),len(partition_lst))
        self.multiplex_property['inter-layer similarity'] = NMI
        
          
    def multiplex_participation_coefficient(self, self_loop = False, replace = True):
        assert self.adj_filtered_lst is not None, "adj_filtered_lst is None!"
        adj_lst = copy.deepcopy(self.adj_filtered_lst)
        N = adj_lst[0].shape[0]
        M = len(adj_lst)
        ### exclude self-loops
        iden_mat = np.diag(np.repeat(1, N))
        K_lst = list()
        for alpha in adj_lst:
            if not self_loop:
                K = np.sum(alpha-iden_mat, axis = 1)
            else:
                K = np.sum(alpha, axis = 1)
            K_lst.append(K)
        O = np.sum(K_lst, axis = 0)
        P = M/(M-1)*(1- np.sum((np.array(K_lst)/O)**2, axis = 0))
        if replace:
            P  = func.replace_min(P, label = "multiplex participation coefficient")
            
        self.multiplex_property['O'] = O
        self.multiplex_property['P'] = P
        
        
    def multiplex_page_rank(self, alpha = 0.85, max_iteration = 100, beta = 1, gamma = 1, threshold = 1e-7, replace = True):
        assert self.adj_filtered_lst is not None, "adj_filtered_lst is None!"
        
        adj_lst_norm = [func.column_normalization(mat) for mat in copy.deepcopy(self.adj_filtered_lst)]
        
        length = len(adj_lst_norm)
        
        mulitiplex_page_rank = func.multiplex_page_rank(adj_lst_norm, alpha_lst = np.repeat(alpha, length).tolist(), \
                         iteration_lst = np.repeat(max_iteration, length).tolist(), \
                         beta = beta, gamma = gamma, threshold = threshold)
        
        self.multiplex_property['multiplex_page_rank'] = mulitiplex_page_rank 
        
        print("Finish computing multiplex PageRank coefficients!")
        
    
    def Jaccard_page_rank(self, alpha = 0.85, max_iteration = 1000, threshold = 1e-7, replace = True):
        assert self.adj_Jaccard is not None, "adj_Jaccard is None!"
        
        Jaccard_norm = func.column_normalization(copy.deepcopy(self.adj_Jaccard))
        
        pg_rank = func.page_rank(Jaccard_norm, alpha = alpha, iteration = max_iteration, threshold = threshold )
        
        if replace:
            pg_rank  = func.replace_min(pg_rank , label = "monoplex PageRank")
        
        self.multiplex_property['Jaccard_page_rank'] = pg_rank 
        
        print("Finish computing PageRank coefficients for Jaccard!")
        
        
        
    def centrality_per_layer(self, method = "hub", alpha = 0.85, max_iteration = 1000, threshold = 1e-7, Jaccard = True):
        assert self.adj_filtered_lst is not None, "adj_filtered_lst is None!"
        if Jaccard:
            assert self.adj_Jaccard is not None, "adj_Jaccard is None!"
            
        if method == "hub":
            if Jaccard:
                Jaccard_hub, _ = func.skn_hits(self.adj_Jaccard)
                self.multiplex_property['Jaccard_hub'] = Jaccard_hub
                print("Finish computing hub scores for the Jaccard layer!")
            self.multiplex_property['per_layer_hub'] = [func.skn_hits(adj_mat)[0] for adj_mat in self.adj_filtered_lst]
            print("Finish computing hub scores for each layer!")
            
        elif method == "pagerank":
            if Jaccard:
                Jaccard_norm = func.column_normalization(copy.deepcopy(self.adj_Jaccard))
                Jaccard_pg = func.page_rank(Jaccard_norm, alpha = alpha, iteration = max_iteration, threshold = threshold)
                self.multiplex_property['Jaccard_page_rank'] = Jaccard_pg 
                print("Finish computing PageRank coefficients for the Jaccard layer!") 
                
            self.multiplex_property['per_layer_page_rank'] = [func.page_rank(func.column_normalization(mat), alpha = alpha, iteration = max_iteration, threshold = threshold)  for mat in copy.deepcopy(self.adj_filtered_lst)] 
            print("Finish computing PageRank coefficients for each layer!")
        else:
            print('"method" can be "hub" or "pagerank"...')
            return
    
    def C_1(self, replace = True):
        assert self.adj_filtered_lst is not None, 'adj_filtered_lst is None!'
        C_1 = func.C_1(self.adj_filtered_lst, replace = replace)       
        self.multiplex_property['C1'] = C_1  
    
    def C_2(self, replace = True, num_core = 30):
        assert self.adj_filtered_lst is not None, 'adj_filtered_lst is None!'
        C_2 = func.C_2(self.adj_filtered_lst, replace = replace, num_core = num_core)
        self.multiplex_property['C2'] = C_2
        
    

        
        
