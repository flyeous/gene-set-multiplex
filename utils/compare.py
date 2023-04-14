import anndata
import contextlib
import hoggorm
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import scanpy as sc
import scipy.spatial
import skbio.stats
from sklearn.metrics import pairwise_distances
import time
from utils import func
from utils import visualization

def hyperparameter_worker_kmeans(gs_lst:list, sampled_data:anndata.AnnData, para_kmeans = {'n_clusters':10, 'random_state':0, 'n_init':10},clustering_metric = "rand_adj", num_core = 30):
    data = sampled_data
    gex_mat, overlap_prop, _ = func.gene_set_overlap_check(gene_set = gs_lst, gene_exp = data)
    begin_time = time.time()
    sim_mat,_, _ = func.hyperdimensional_similarity_cluster_structure(gene_set_mat_lst = gex_mat, clustering_method = "kmeans", clustering_metric = clustering_metric,  para_kmeans = para_kmeans,  num_core = num_core )
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'Kmeans takes {end_time - begin_time:.2f} s')
    return sim_mat, time_cost

def hyperparameter_worker_BGM(gs_lst:list, sampled_data:anndata.AnnData, clustering_metric = "rand_adj", para_dirichlet:dict = {'n_components':20, 'covariance_type':"full", 'max_iter':1000}, num_core = 30):
    data = sampled_data
    gex_mat, overlap_prop, _ = func.gene_set_overlap_check(gene_set = gs_lst, gene_exp = data)
    
    begin_time = time.time()
    sim_mat,_, _, _ = func.hyperdimensional_similarity_cluster_structure(gene_set_mat_lst = gex_mat, clustering_method = "BGM", clustering_metric = clustering_metric, para_dirichlet = para_dirichlet, num_core = num_core)
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'BGM takes {end_time - begin_time:.2f} s')
    
    return sim_mat, time_cost

def hyperparameter_worker_leiden(gs_lst:list, sampled_data:anndata.AnnData, clustering_metric = "rand_adj", pca_para = {}, pheno_para = {'k':30, 'resolution_parameter':2, 'seed':111, 'min_cluster_size': 50}, num_core = 30):
    data = sampled_data
    gex_mat, overlap_prop, _ = func.gene_set_overlap_check(gene_set = gs_lst, gene_exp = data)
    
    begin_time = time.time()
    sim_mat,_, _, _ = func.hyperdimensional_similarity_cluster_structure(gene_set_mat_lst = gex_mat, clustering_method = "leiden", clustering_metric = clustering_metric, pca_para = pca_para,pheno_para = pheno_para, num_core = num_core )
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'Leiden takes {end_time - begin_time:.2f} s')
    return sim_mat, time_cost


def pairwise_compute(mat_i, mat_j, func, func_parameters):
    return func(mat_i, mat_j, **func_parameters)


def mantel_worker(sampled_gs:list, sampled_data:anndata.AnnData)-> (float, np.ndarray):
    length = len(sampled_gs)
    gex_mat, overlap_prop, _ = func.gene_set_overlap_check(gene_set = sampled_gs, gene_exp = sampled_data)
    
    begin_time = time.time()
    dist_mat_lst = [scipy.spatial.distance_matrix(gex_mat[i], gex_mat[i]) for i in range(length)]
    res = [pairwise_compute(dist_i, dist_j, skbio.stats.distance.mantel, {"permutations":0})[0] for dist_i in dist_mat_lst for dist_j in dist_mat_lst]
    res = np.array(res).reshape(length, length)
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'mantel takes {end_time - begin_time:.2f} s')
    
    return (res, time_cost)
    
def RV2_worker(sampled_gs:list, sampled_data:anndata.AnnData)-> (float, np.ndarray):
    length = len(sampled_gs)
    gex_mat, overlap_prop, _ = func.gene_set_overlap_check(gene_set = sampled_gs, gene_exp = sampled_data)  
    begin_time = time.time()
    res_RV2 = hoggorm.mat_corr_coeff.RV2coeff([gex_mat[i] - np.mean(gex_mat[i], axis = 0) for i in range(length)])
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'RV2coeff takes {end_time - begin_time:.2f} s')
    return (res_RV2, time_cost)

def modified_Jaccard_worker(sampled_gs, sampled_data:anndata.AnnData, num_core = 30):
    begin_time = time.time()
    res = func.modified_JC(gene_set_lst = sampled_gs, gene_exp = sampled_data, num_core = num_core)
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'Modified Jaccard takes {end_time - begin_time:.2f} s')
    return (res, time_cost)


### Hyperparameters
def run_duration_comparison_hyperparameter(sampled_gs:list, sampled_data:anndata.AnnData, clustering_metric, worker, hyper_lst, num_core ):
    time_lst = []
    res_lst = []
    
    ### reference:
    res_RV2, _ = RV2_worker(sampled_gs, sampled_data)
    res_mantel, _ = mantel_worker(sampled_gs, sampled_data)
    
    for hp in hyper_lst:
        res_temp, time_temp = worker(sampled_gs, sampled_data, hp, clustering_metric = clustering_metric)
        time_lst.append(time_temp)
        res_lst.append(res_temp)
        
    
    return res_lst, time_lst, res_RV2, res_mantel

### full version 
def run_duration_comparison_full(sampled_gs:list, sampled_data:anndata.AnnData, clustering_metric, num_core):
    time_list = []
    
    res_RV2, time_cost = RV2_worker(sampled_gs, sampled_data)
    time_list.append(('RV2coeff',time_cost))

    res_leiden, time_cost = hyperparameter_worker_leiden(sampled_gs, sampled_data, clustering_metric = clustering_metric, num_core = num_core)
    time_list.append(('leiden',time_cost))
    
    res_BGM, time_cost = hyperparameter_worker_BGM(sampled_gs, sampled_data, clustering_metric = clustering_metric, num_core = num_core)
    time_list.append(('BGM',time_cost))
    
    res_kmeans, time_cost = hyperparameter_worker_kmeans(sampled_gs, sampled_data, clustering_metric = clustering_metric, num_core = num_core)
    time_list.append(('kmeans',time_cost))
    
    res_modified_Jaccard, time_cost = modified_Jaccard_worker(sampled_gs, sampled_data, num_core = num_core)
    time_list.append(('modified_Jaccard',time_cost))
    
    res_mantel, time_cost = mantel_worker(sampled_gs, sampled_data)
    time_list.append(('mantel',time_cost))

    res_list = [res_RV2, res_leiden, res_BGM, res_kmeans, res_modified_Jaccard, res_mantel]
    return time_list, res_list

def gene_set_similarity_revealed_by_clustering_structure_UMAP_illustration(E_loc, F_loc, gs_names, gs_genes, \
                         filter_index, scRNAseq_sample, Jaccard_similarity_matrix, clustering_metric = "AMI", title_size = 16,\
                     para_UMAP = {'random_state':111, 'n_neighbors':15}, \
             para_plot = {'figsize':(5,5), 'dpi':100, 'cmap':"Spectral"},\
            para_kmeans = {'n_clusters':10,'random_state':0, 'n_init':10, 'max_iter': 300, \
                           'tol':1e-4, 'algorithm':'lloyd'}, num_core = 60):
    
    index_E = filter_index[E_loc]
    gene_set_E_ns = gs_names[index_E]
    gene_set_E = gs_genes[index_E]
    index_F = filter_index[F_loc]
    gene_set_F_ns = gs_names[index_F]
    gene_set_F = gs_genes[index_F]
    
    U_E, _, _ = func.gene_set_overlap_check([gene_set_E], scRNAseq_sample)
    U_F, _, _ = func.gene_set_overlap_check([gene_set_F], scRNAseq_sample)
    
    rand_mat, cluster_label = func.cluster_measure_pipe(U_lst = [U_E[0], U_F[0]], clustering_method = "kmeans", clustering_metric = clustering_metric,\
                min_feature = 4, num_core = num_core, para_kmeans = para_kmeans)
    
    _, UMAP_cor_E = visualization.vis_UMAP(1-pairwise_distances(U_E[0]),\
                                      feature = cluster_label[0], title = gene_set_E_ns, title_size = title_size, \
             UMAP_cor = None, 
             para_UMAP = para_UMAP, \
             para_plot = para_plot, colorbar = False)
    _, UMAP_cor_F = visualization.vis_UMAP(1-pairwise_distances(U_F[0]),\
                                      feature = cluster_label[1], title = gene_set_F_ns, title_size = title_size, \
             UMAP_cor = None, 
             para_UMAP = para_UMAP, \
             para_plot = para_plot, colorbar = False)

    H = rand_mat[0,1]
    Jaccard_coef = Jaccard_similarity_matrix[filter_index, :][:, filter_index][E_loc, F_loc]
    RV_coef = RV2_worker([gene_set_E, gene_set_F], scRNAseq_sample)[0][0,1]
    mantel_coef = mantel_worker([gene_set_E, gene_set_F], scRNAseq_sample)[0][0,1]
    
    print(f'The similarity between {gs_names[E_loc]} and {gs_names[F_loc]}\n'+\
          f'is {Jaccard_coef:.3f} (Jaccard), {H:.3f} (S), {RV_coef:.3f} (RV_coef), and {mantel_coef:.3f} (mantel_coef).')
    
    return UMAP_cor_E, UMAP_cor_F, H, RV_coef, mantel_coef, Jaccard_coef, U_E, U_F

