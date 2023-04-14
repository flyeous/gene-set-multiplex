import anndata
import contextlib
import copy
import gseapy as gp
import itertools
import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
import scanpy as sc
import scipy
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from sklearn import mixture
import sknetwork as skn
from tqdm import tqdm
import umap
import warnings
warnings.filterwarnings('ignore')
### warnings.filterwarnings(action='once')


### monoplex/multiplex PageRank centrality
def column_normalization(A):
    A = A / np.sum(A, axis = 0)
    return A

def page_rank(A, alpha, iteration, threshold = 1e-5):
    N = A.shape[0]
    current_rank = np.repeat(1/N, N).reshape(-1, 1)
    A_power = (alpha)*A + (1-alpha)/N
    converge = False
    for i in range(iteration):
        next_rank = A_power@current_rank
        difference = current_rank - next_rank
        current_rank = next_rank
        if np.linalg.norm(difference) < threshold:
            converge = True
            break
    if not converge:
        print(f'The PageRank does not converge with {iteration} iteration times and convergence threshold {threshold}')
        return
    return current_rank.ravel()

def skn_page_rank(A, alpha = 0.85, iteration = 1000, threshold=1e-5):
    model = skn.ranking.PageRank(damping_factor = 0.85, n_iter= 1000, tol = 1e-5)
    model.fit(A)
    return model.scores_

def skn_hits(A):
    model = skn.ranking.HITS()
    model.fit(A)
    hub, authority = model.scores_row_, model.scores_col_
    return hub, authority

def multiplex_page_rank(A_lst, alpha_lst, iteration_lst, beta, gamma, threshold = 1e-7):
    N = A_lst[0].shape[0]
    init_rank = np.repeat(1/N, N).reshape(-1, 1)
    for i in range(len(A_lst)):
        if i == 0:
            current_rank = page_rank(column_normalization(A_lst[i]), alpha = alpha_lst[i], iteration = iteration_lst[0], threshold = threshold)
        else:  
            temp_adj = np.diag(current_rank.ravel()**(beta))@(A_lst[i])
            col_norm = np.sum(temp_adj, axis = 0)
            temp_adj = temp_adj/col_norm
            current_rank_iter = copy.deepcopy(init_rank)
            for j in range(iteration_lst[i]):
                next_rank_iter = alpha_lst[i]*temp_adj@current_rank_iter + (1-alpha_lst[i])*(current_rank**(gamma)/np.sum(current_rank**(gamma))).reshape(-1,1) 
                difference = next_rank_iter - current_rank_iter
                current_rank_iter = next_rank_iter
            if np.linalg.norm(difference) >= threshold:
                print(f'The multiplex PageRank coefficients do not converge in layer {i}')
                return
            else:
                current_rank = copy.deepcopy(current_rank_iter)
                
    return current_rank.ravel()

def multiplex_participation_coefficient(adj_filtered_lst, self_loop = False, replace = True):
    assert adj_filtered_lst is not None, "adj_filtered_lst is None!"
    adj_lst = copy.deepcopy(adj_filtered_lst)
    N = adj_lst[0].shape[0]
    M = len(adj_lst)
    ### remove self-loops
    iden_mat = np.diag(np.repeat(1, N))
    res_lst = list()
    for alpha in adj_lst:
        if not self_loop:
            C_i = np.sum(alpha-iden_mat, axis = 1)
        else:
            C_i = np.sum(alpha, axis = 1)
        res_lst.append(C_i)
    O_i = np.sum(res_lst, axis = 0)
    P_i = M/(M-1)*(1- np.sum((np.array(res_lst)/O_i)**2, axis = 0))
    if replace:
        P_i  = replace_min(P_i , label = "multiplex participation coefficient")
    return O_i, P_i


def C_1(adj_lst, replace = True):
    adj_lst = copy.deepcopy(adj_lst)
    N = adj_lst[0].shape[0]
    M = len(adj_lst)
    ### remove self-loops
    iden_mat = np.diag(np.repeat(1, N))
    adj_lst = [(mat - iden_mat) for mat in adj_lst]
    numerator = 0
    for i,j in itertools.permutations(range(M),2): 
        numerator += np.diag((adj_lst[i]+adj_lst[i].transpose())@(adj_lst[j]+adj_lst[j].transpose())@(adj_lst[i]+adj_lst[i].transpose()))  
    denominator = 0
    for i in range(M):
        denominator += np.sum(adj_lst[i]+adj_lst[i].transpose(), axis = 1)**2 - np.sum((adj_lst[i]+adj_lst[i].transpose())**2, axis = 1)
    denominator *= 2*(M-1)
    C_1 = numerator/denominator
    if replace:
        C_1 = replace_min(C_1, label = "C1") 
    return C_1 

def C_2(adj_lst, replace = True, num_core = 30):
    adj_lst = copy.deepcopy(adj_lst)
    N = adj_lst[0].shape[0]
    M = len(adj_lst)
    ### remove self-loops
    iden_mat = np.diag(np.repeat(1, N))
    adj_lst = [(mat - iden_mat) for mat in adj_lst]
    numerator = 0
    for i,j,k in itertools.permutations(range(M),3):
        numerator += np.diag((adj_lst[i]+adj_lst[i].transpose())@(adj_lst[j]+adj_lst[j].transpose())@(adj_lst[k]+adj_lst[k].transpose()))
    denominator = 0
    for i,j in itertools.permutations(range(M),2): 
        denominator += np.sum(adj_lst[i]+adj_lst[i].transpose(), axis = 1)*np.sum(adj_lst[j]+adj_lst[j].transpose(), axis = 1) - np.sum((adj_lst[i]+adj_lst[i].transpose())*(adj_lst[j]+adj_lst[j].transpose()), axis = 1)
    
    denominator *= 2*(M-2)
    C_2 = numerator/denominator
    if replace:
         C_2 = replace_min(C_2, label = "C2")
    return C_2



def cluster_measure_pipe(U_lst, clustering_method = "leiden", clustering_metric = "rand_adj", min_feature = 4, num_core = 10, 
                         pca_para = {}, \
    pheno_para = {'k':30, 'resolution_parameter':2, 'seed':111, 'min_cluster_size': 50},\
                         para_dirichlet = {'n_components':20, 'covariance_type':"full", 'max_iter':1000},\
                         para_kmeans = {'n_clusters':10, 'random_state':0, 'n_init':10, 'max_iter': 300, 'tol':1e-4, 'algorithm':'lloyd'}):
                                        
    cluster_label = []
    
    print(f'{clustering_method} clustering ...')
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        if clustering_method == "leiden":
            cluster_quality = []
    
            for i in tqdm(range(len(U_lst))):
                U = U_lst[i]
                if U.shape[1] < min_feature:
                    print(f'gene set {i} has less than {min_feature} features!')
                    return
                U = anndata.AnnData(U, dtype = "float64")
                sc.pp.pca(U, **pca_para)     
                sc.external.tl.phenograph(U, clustering_algo = 'leiden', **pheno_para)     
                ### Append the clustering labels
                labels = U.obs['pheno_leiden'].to_list()
                cluster_label.append(labels)
                ### Modularity score
                Q_score = U.uns['pheno_jaccard_q']
                cluster_quality.append(Q_score)
        
        elif clustering_method == "BGM": 
            cluster_num = []
        
            for i in tqdm(range(len(U_lst))):
                U = U_lst[i].toarray()
                if U.shape[1] < min_feature:
                    print(f'gene set {i} has less than {min_feature} features!')
                    return
                model = mixture.BayesianGaussianMixture(**para_dirichlet).fit(U)
                cluster_label.append(model.predict(U))
                cluster_num.append(len(np.unique(model.predict(U))))
            
        elif clustering_method == "kmeans":
            for i in tqdm(range(len(U_lst))):
                U = U_lst[i]
                if U.shape[1] < min_feature:
                    print(f'gene set {i} has less than {min_feature} features!')
                    return
                kmeans = KMeans(**para_kmeans).fit(U)
                cluster_label.append(kmeans.labels_)
                
        else:
            print("Unsupported clustering method!")
        
    length = len(cluster_label)
    print("Clustering finishes! Computing similarity matrix ...")
        
    if clustering_metric in ["rand_adj", "AMI", "NMI"]:
        if __name__ == "utils.func":
            with Pool(num_core) as proc: 
                res_mat = proc.starmap(eval(f'_{clustering_metric}'), [(i,j, cluster_label) for i in range(length) for j in range(length)])
            res_mat = np.reshape(np.array(res_mat), (length, length))   
        
        if clustering_method == "leiden":
            return (res_mat, cluster_label, cluster_quality)
        elif clustering_method == "BGM":
            return (res_mat, cluster_label, cluster_num)
        else:
            return (res_mat, cluster_label)
    else:
        print("Unsupported clustering metric!")
        return
    
    
    
def _rand_adj(i:int,j:int,cluster_label:list) -> float:

    return metrics.adjusted_rand_score(cluster_label[i], cluster_label[j])


def _NMI(i, j, cluster_label):
    return metrics.normalized_mutual_info_score(cluster_label[i], cluster_label[j])


def _AMI(i,j, cluster_label):
    return metrics.adjusted_mutual_info_score(cluster_label[i], cluster_label[j])




### for tissue objects
def JC(i:int, j:int, gs_list:list) -> float:
    
    return len(set(gs_list[i]).intersection(set(gs_list[j])))/len(set(gs_list[i]).union(set(gs_list[j])))

def JC_proj(i, j, gs_list_1, gs_list_2):
    
    return len(set(gs_list_1[i]).intersection(set(gs_list_2[j])))/len(set(gs_list_1[i]).union(set(gs_list_2[j])))


def _modified_JC_worker(gs_i, gs_j, gene_weight_dict):
    intersection = set(gs_i).intersection(set(gs_j))
    union = set(gs_i).union(set(gs_j))
   
    numerator = np.sum([gene_weight_dict[gene] for gene in intersection])
    denominator = np.sum([gene_weight_dict[gene] for gene in union])
    if denominator == 0:
        print("Two gene sets have zero gene expression! Missing value is replaced by 0.")
        return 0
    else:
        return numerator/denominator


def modified_JC(gene_set_lst, gene_exp:anndata.AnnData, num_core = 30):
    gene_weight = np.array(np.sum(gene_exp.X, axis = 0)/(gene_exp.X.shape[0])).ravel()
    gene_in_exp = gene_exp.var.index.to_numpy()
    gene_weight_dict = dict(zip(gene_in_exp, gene_weight))
    filtered_gs_lst = [list(set(gs).intersection(gene_in_exp)) for gs in gene_set_lst]
    
    
    length = len(gene_set_lst)
    if __name__ == "utils.func":
        with Pool(num_core) as proc:           
            mat = proc.starmap(_modified_JC_worker, [(gs_i,gs_j, gene_weight_dict) \
                                                     for gs_i in filtered_gs_lst for gs_j in filtered_gs_lst]) 
        mat = np.reshape(np.array(mat), (length, length))
    return mat
    
def gene_set_overlap_check(gene_set:list, gene_exp:anndata.AnnData, min_feature = 4, min_prop = 0.9):
    gex_mat = []
    overlap_prop = []
    filtered_index = []
    filtered = 0
    for i in tqdm(range(len(gene_set))):
        temp_gene_list = list(set(gene_set[i]).intersection(gene_exp.var.index))
        proportion = len(temp_gene_list)/len(gene_set[i])
        ### remove gene sets that contain less than 3 detected genes 
        ### or whose proportion of detected genes is less than 90% 
        if len(temp_gene_list) < min_feature or proportion < min_prop:
            filtered += 1
            continue
        ### gene expression matrix: cells * genes (in the fitered gene set)
        temp_gene_mat = gene_exp[:,temp_gene_list]
        gex_mat.append(temp_gene_mat.X)
        overlap_prop.append(proportion)
        filtered_index.append(i)
    print(f'{len(gene_set) - filtered} over {len(gene_set)} pass the first filteration! The overlap with sequenced genes is {np.round(np.mean(overlap_prop), 3)} +/- {np.round(np.std(overlap_prop), 3)}.')
    return gex_mat, overlap_prop, filtered_index


def hyperdimensional_similarity_cluster_structure(gene_set_mat_lst, clustering_method = "leiden", clustering_metric = "rand_adj", min_feature = 4, num_core = 10, pca_para = {}, \
    pheno_para = {'k':30, 'resolution_parameter':2, 'seed':111, 'min_cluster_size': 50},\
                         para_dirichlet = {'n_components':20, 'covariance_type':"full", 'max_iter':1000},\
                                                 para_kmeans = {'n_clusters':10, 'random_state':0, 'n_init':10}):
    
    return cluster_measure_pipe(gene_set_mat_lst, clustering_method = clustering_method, clustering_metric = clustering_metric, min_feature = min_feature, num_core = num_core, pca_para = pca_para, pheno_para = pheno_para, para_dirichlet = para_dirichlet, para_kmeans = para_kmeans)


### for network objects
def self_loop_exclude(sim_mat_lst):
    mat_lst = copy.deepcopy(sim_mat_lst)
    N = mat_lst[0].shape[0]
    iden_mat = np.diag(np.repeat(1, N))
    mat_lst = [(mat - iden_mat) for mat in mat_lst]
    return mat_lst
    
def adj_filter_tr(adj_mat,q = None, tr = 0):
    adj_mat = copy.deepcopy(adj_mat)
    ### negative edges are removed
    adj_mat[adj_mat < 0] = 0 
    ### weights lower than a threshold or quantile truncated to zero
    if tr is None:
        tr = np.quantile(adj_mat, q)
        print(f'The {q*100} quantile value is {tr}.')
    adj_mat[adj_mat < tr] = 0 
    ### min-max normalization
    (adj_mat - np.min(adj_mat))/(np.max(adj_mat) - np.min(adj_mat))
    return adj_mat


def _projection_worker(i:int,j:int,cluster_label_1:list, cluster_label_2:list, clustering_metric) -> float:
    if clustering_metric == "rand_adj":
        return metrics.adjusted_rand_score(cluster_label_1[i], cluster_label_2[j])
    elif clustering_metric ==  "NMI":
        return metrics.normalized_mutual_info_score(cluster_label_1[i], cluster_label_2[j])
    elif clustering_metric == "AMI":
        return metrics.adjusted_mutual_info_score(cluster_label_1[i], cluster_label_2[j])
    else:
        print("Unsupported clustering metric!")
        return 

### Projection 
def projection(num_core = 30, cluster_label_1 = None, cluster_label_2 = None, clustering_metric = "AMI"):

    assert cluster_label_1 is not None and cluster_label_2 is not None, 'Two list of cluster labels required.'
    
    length_1 = len(cluster_label_1)
    length_2 = len(cluster_label_2)
    
    print("Clustering had better be performed on the same samples!")
    print("Computing the projection matrix ... ")
    if __name__ == "utils.func":
        with Pool(num_core) as proc:           
            proj_mat = proc.starmap(_projection_worker, [(i,j, cluster_label_1, cluster_label_2, clustering_metric) for i in range(length_1) for j in range(length_2)]) 
        proj_mat = np.reshape(np.array(proj_mat), (length_1, length_2))   
        
        return proj_mat 
    
def filter_check(layer_lst, gene_set_collection_name, method):
 
    length = len(layer_lst)
    filter_check = np.array([layer_i.gene_set[gene_set_collection_name].filter_index[method] \
                         == layer_j.gene_set[gene_set_collection_name].filter_index[method] \
                         for layer_i in layer_lst for layer_j in layer_lst ]).reshape(length, length)
    return np.sum(filter_check != True)
        
def combine_sim(sim_1, proj_1_2, sim_2):
    
    temp_1 = np.concatenate([sim_1, proj_1_2.transpose()], axis = 0)
    temp_2 = np.concatenate([proj_1_2, sim_2], axis = 0)
    return np.concatenate([temp_1, temp_2], axis = 1)

### Replace missing values by the non-missing minimal value
def replace_min(x, label = None ):
    vec = copy.deepcopy(x)
    missing_index = np.where(np.isnan(vec) == True)[0]
    print(f'The raw {label} coefficients contain {len(missing_index)} missing values.')
    if len(missing_index) != 0:
        min_val = np.min(np.delete(vec, missing_index))
        vec[missing_index] = min_val
        print(f"The missing values have been replaced by the minimal non-missing value:{min_val:.3f}.")
    return vec
    
    
### For gene set enrichment analysis
def gsea_tissue(tissue_exp, tissue_con, exp_n, con_n, \
                gene_set_path = "c5.go.cc.v2023.1.Hs.symbols.gmt", \
                para_gsea = {'permutation_type': 'phenotype', 'permutation_num':2000, 'outdir':None,\
                            'method':'signal_to_noise', 'threads':30, 'seed':30}):
    
    assert np.sum(tissue_exp.var.index != tissue_con.var.index) == 0, "genes are different!"
    
    sample_ns = np.concatenate([np.repeat(exp_n, tissue_exp.X.shape[0]), \
                                np.repeat(con_n, tissue_con.X.shape[0])], axis = 0).tolist()
    gene_ns = pd.DataFrame({'Gene' : tissue_exp.var.index, 'NAME' : tissue_exp.var.index})
    
    exp_df = pd.DataFrame(tissue_exp.X.toarray().transpose(),columns = [exp_n for i in range(tissue_exp.X.shape[0])])
    
    con_df = pd.DataFrame(tissue_con.X.toarray().transpose(),columns = [con_n for i in range(tissue_con.X.shape[0])])
    
    res_df = pd.concat([gene_ns, exp_df, con_df], axis = 1)
    
    gs_res = gp.gsea(data = res_df, 
                 gene_sets = gene_set_path,
                 cls= sample_ns, 
                 **para_gsea)
    
    return gs_res, res_df
    
    