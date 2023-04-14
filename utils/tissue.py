import anndata
import copy
import contextlib
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import pandas as pd
import scanpy as sc
import time
from utils import func

# import warnings
# warnings.filterwarnings('ignore')
# ### warnings.filterwarnings(action='once')
# ### Disable warnings: reference https://stackoverflow.com/questions/8391411
    
class tissue:
    
    _Jaccard_proj = {}
    def __init__(self, name, scRNAseq, processed, gene_set_collection_name, gene_set_gene_symbols, gene_set_names):
        self.name = name
        self.scRNAseq = scRNAseq
        self.processed = processed
        self.gene_set = {gene_set_collection_name: geneSet(raw_gs = gene_set_gene_symbols, raw_gs_names = gene_set_names)}
        self.proj = {}
        self.sampled_data = None
        
    def scRNAseq_release(self):
        self.scRNAseq = None
        self.sampled_data = None
    
    def cell_sampling(self, size, seed):
        assert self.scRNAseq is not None, "Load scRNAseq data first!"
        assert size <= self.scRNAseq.X.shape[0], f'sample size should be less than {self.scRNAseq.X.shape[0]}!'

        np.random.seed(seed)
        sampling = np.random.choice(self.scRNAseq.X.shape[0], size = size, replace = False)
        self.sampled_data = (seed, size, self.scRNAseq[sampling,])
        
    
    def add_scRNAseq(self, scRNAseq):
        self.scRNAseq = scRNAseq
        
    def add_gene_set(self,gene_set_collection_name, gene_set_gene_symbols, gene_set_names):
        self.gene_set[gene_set_collection_name] = geneSet(raw_gs = gene_set_gene_symbols, raw_gs_names = gene_set_names)
    
    def call_leiden(self, gene_set_collection_name, clustering_metric =  "rand_adj", para_leiden = {'pca_para' : {'random_state':222},  'pheno_para' : {'seed':333, 'k' : 15, 'resolution_parameter' : 2}, 'num_core' : 30}):

        assert self.sampled_data is not None, 'Call cell_sampling first!'
        
        self.gene_set[gene_set_collection_name]._leiden(sampled_data = self.sampled_data[2], clustering_metric = clustering_metric, **para_leiden)
        
    def call_Jaccard(self, gene_set_collection_name, num_core = 30):
        self.gene_set[gene_set_collection_name]._Jaccard(num_core)
        
    def call_modified_Jaccard(self, gene_set_collection_name, num_core = 30):
        assert self.sampled_data is not None, 'Call cell_sampling first!'
        self.gene_set[gene_set_collection_name]._modified_Jaccard(sampled_data = self.sampled_data[2], num_core = num_core)
    
    def call_tSNE_dirichlet(self, gene_set_collection_name,  clustering_metric = "rand_adj",\
    para_tSNE_dirichlet = {'para_tSNE' : {'n_components':2, 'perplexity':30.0, 'n_iter':1000, 'random_state':123}, \
    'para_dirichlet' : {'n_components':20, 'covariance_type':"full", 'max_iter':1000}, 'num_core': 30}):
        assert self.sampled_data is not None, 'Call cell_sampling first!'
        
        self.gene_set[gene_set_collection_name]._tSNE_dirichlet(sampled_data = self.sampled_data[2],clustering_metric = clustering_metric, **para_tSNE_dirichlet)
        
    def call_kmeans(self, gene_set_collection_name,  clustering_metric = "rand_adj",\
        para_Kmeans = {'num_core':30, 'para_kmeans':{'n_clusters':10, 'random_state':0, 'n_init':10}}):
        assert self.sampled_data is not None, 'Call cell_sampling first!'
        
        self.gene_set[gene_set_collection_name]._kmeans(sampled_data = self.sampled_data[2], clustering_metric = clustering_metric, **para_Kmeans)
        
    def call_proj(self, gene_set_collection_name_1, gene_set_collection_name_2, method, clustering_metric = "AMI", num_cores = 30 ):
        assert (self.gene_set[gene_set_collection_name_1] is not None) and (self.gene_set[gene_set_collection_name_2] is not None), "Make sure two gene sets exist!"
        assert method in ["BGM", "leiden", "Jaccard", "kmeans"], "Unsupported method!"   
        
        if method != "Jaccard":
            assert (self.gene_set[gene_set_collection_name_1].cluster_labels[method]) is not None and (self.gene_set[gene_set_collection_name_2].cluster_labels[method]) is not None, "!"
            
            self.proj[f'proj_{method}_{gene_set_collection_name_1}_{gene_set_collection_name_2}'] = func.projection(num_core = num_cores, cluster_label_1 = self.gene_set[gene_set_collection_name_1].cluster_labels[method], cluster_label_2 = self.gene_set[gene_set_collection_name_2].cluster_labels[method], clustering_metric = clustering_metric)
        else:
            gene_set_list_1 = self.gene_set[gene_set_collection_name_1].gene_set
            gene_set_list_2 = self.gene_set[gene_set_collection_name_2].gene_set
            len_1 = len(gene_set_list_1)
            len_2 = len(gene_set_list_2)
          
            if __name__ == "utils.tissue":
                with Pool(num_cores) as proc:           
                    res = proc.starmap(func.JC_proj, [(i,j, gene_set_list_1, gene_set_list_2) for i in range(len_1) for j in range(len_2)]) 
                    res = np.reshape(np.array(res), (len_1, len_2))   
            print(f"Finish building the Jaccard coefficient similarity matrix! ")
            self._Jaccard_proj[f'proj_{method}_{gene_set_collection_name_1}_{gene_set_collection_name_2}'] = res        
        
class geneSet():
    
    methods = ['Jaccard', 'kmeans', 'leiden', 'BayesianGaussianMixture']

    def __init__(self, raw_gs:list, raw_gs_names:list):
        self.gene_set:list = raw_gs
        self.gs_names:list = raw_gs_names
        self.cluster_labels:dict = {'leiden': None, 'BGM':None }
        self.filter_index:dict = {'leiden': None, 'BGM': None}
        self.filter_overlap:dict = {'leiden': None, 'BGM': None}
        self.miscellaneous:dict = {'leiden_modularity': None, 'BGM_cluster_num': None}
        self.sim_mat: dict = {'Jaccard': None, 'leiden': None, 'BGM':None, }
        
    def _leiden(self, sampled_data, clustering_metric, num_core, pca_para = {}, pheno_para = {}) :

        assert self.gene_set is not None, 'self.gene_set is None.'
        
        data = sampled_data
        gex_mat, overlap_prop, filtered_index = func.gene_set_overlap_check(gene_set = self.gene_set, gene_exp = data)
        res = func.hyperdimensional_similarity_cluster_structure(gene_set_mat_lst = gex_mat, clustering_method = "leiden", clustering_metric = clustering_metric, num_core = num_core,  pca_para = pca_para, pheno_para = pheno_para)   
        
        self.filter_overlap['leiden'] = overlap_prop
        self.sim_mat['leiden'] = res[0]
        self.cluster_labels['leiden'] = res[1]
        self.filter_index['leiden'] = filtered_index
        self.miscellaneous['leiden_modularity'] = res[2]

    def _Jaccard(self, num_core):

        length = len(self.gene_set)
           
        if __name__ == "utils.tissue":
            with Pool(num_core) as proc:           
                res = proc.starmap(func.JC, [(i,j, self.gene_set) for i in range(length) for j in range(length)]) 
        res = np.reshape(np.array(res), (length, length))   
        
        geneSet._Jaccard_mat = res
        self.sim_mat['Jaccard'] = geneSet._Jaccard_mat
        
        print("Finish building the Jaccard coefficient similarity matrix!")
    
    def _modified_Jaccard(self, sampled_data, num_core):
        assert self.gene_set is not None, 'self.gene_set is None.'
        data = sampled_data
        res = func.modified_JC(self.gene_set, data, num_core = num_core)  
        
        self.sim_mat['modified_Jaccard'] = res
        
        print("Finish building the modified Jaccard coefficient similarity matrix!")
      
    def _tSNE_dirichlet(self, sampled_data, num_core, clustering_metric, para_tSNE = {}, para_dirichlet = {}):
        
        assert self.gene_set is not None, 'gene_set is None.'
        
        data = sampled_data
        gex_mat, overlap_prop, filtered_index = func.gene_set_overlap_check(gene_set = self.gene_set, gene_exp = data)
        res = func.hyperdimensional_similarity_cluster_structure(gene_set_mat_lst = gex_mat, clustering_method = "BGM", clustering_metric = clustering_metric, num_core = num_core, para_tSNE = para_tSNE, para_dirichlet = para_dirichlet)   
        self.sim_mat['BGM'] = res[0]
        self.cluster_labels['BGM'] = res[1]
        self.filter_index['BGM'] = filtered_index
        self.filter_overlap['BGM'] = overlap_prop
        self.miscellaneous['BGM_cluster_num'] = res[2]
        
    def _kmeans(self, sampled_data, num_core, clustering_metric, para_kmeans = {}):
        
        assert self.gene_set is not None, 'gene_set is None.'
        data = sampled_data
        gex_mat, overlap_prop, filtered_index = func.gene_set_overlap_check(gene_set = self.gene_set, gene_exp = data)
        res = func.hyperdimensional_similarity_cluster_structure(gene_set_mat_lst = gex_mat, clustering_method = "kmeans", clustering_metric = clustering_metric, num_core = num_core, para_kmeans = para_kmeans)   
        self.sim_mat['kmeans'] = res[0]
        self.cluster_labels['kmeans'] = res[1]
        self.filter_index['kmeans'] = filtered_index
        self.filter_overlap['kmeans'] = overlap_prop
        self.miscellaneous['k'] = para_kmeans['n_clusters']
    
    
        
    
        
        
        
             
        