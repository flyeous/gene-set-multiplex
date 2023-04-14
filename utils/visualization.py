import copy
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy
import scipy.stats as stats
import umap


def vis_hyper_kmeans(mat, K_num, labels, ref_name, figsize=(12,12), anno_font_size = 16, title_size = 18):
    fig, ax = plt.subplots(figsize= figsize, dpi = 300)
    for i in range(mat.shape[0]):
        ax.plot(np.array(K_num), mat[i,:], label = labels[i])
    ax.set_xlabel('number of K', fontsize=anno_font_size)
    ax.set_ylabel('Pearson correlation coefficient', fontsize=anno_font_size)
    ax.tick_params(axis='both', which='major', labelsize=anno_font_size-2)
    ax.tick_params(axis='both', which='minor', labelsize=anno_font_size-4)
    ax.legend(frameon=False,bbox_to_anchor= (1.0, 1.0), prop={'size': anno_font_size})
    ax.set_title(f'{ref_name}', fontsize = title_size)
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    return fig
    
def vis_heatmap(mat, labels, figsize=(12,12), title = "ImmuneSig",  legend_title = "NMI", x_rotation = 0, y_rotation = 0, ticks = np.arange(0.5,10.5,1), title_size = 16, anno_font_size = 14, cbar_font = 14, para_heatmap = {'annot':True, 'vmin':0, 'vmax':1, 'cmap':'magma', 'square':True, 'cbar_kws': {"shrink": .25, "anchor" :(0.0, 0.85)}} ):
    fig, ax = plt.subplots(figsize=figsize) 
    ax_sns = sns.heatmap(np.round(mat,2), ax = ax, mask = np.triu(mat) , **para_heatmap, annot_kws = {"fontsize":anno_font_size})
    plt.xticks(ticks=ticks, labels=labels, rotation= x_rotation, fontsize= anno_font_size)
    plt.yticks(ticks=ticks, labels=labels,  rotation= y_rotation, fontsize= anno_font_size)
    ax.legend(title=legend_title, frameon=False, title_fontsize = anno_font_size)
    ax.set_title('{}'.format(title),
             fontsize = title_size)
    ### font size of the colorbar
    color_bar = ax_sns.collections[0].colorbar
    color_bar.ax.tick_params(labelsize=cbar_font)
    fig.tight_layout()
    return fig


def vis_UMAP(adj_tr, feature, title, title_size = 16, \
             UMAP_coord = None, 
             para_UMAP = {'random_state':111, 'n_neighbors':50}, \
             para_plot = {'figsize':(5,5), 'dpi':100, 'cmap':"Spectral"}, colorbar = True):
    if UMAP_coord is None:
        model = umap.UMAP(**para_UMAP, metric='precomputed')
        res = model.fit_transform(1 - adj_tr)
        vis = pd.DataFrame(res, columns = ['umap1','umap2'])
    else:
        vis = UMAP_coord.iloc[:,:3]
    vis['feature'] = feature

    fig = plt.figure(figsize=para_plot['figsize'], dpi=para_plot['dpi'])
    plt.scatter(x = vis['umap1'], y = vis['umap2'], c = vis['feature'], cmap = para_plot['cmap'])
    plt.box(on=None)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(title, fontsize=title_size)
    if colorbar:
        plt.colorbar( anchor = (0.0, 1.0), fraction = 0.020 )
    fig.tight_layout()
    plt.show()
    return fig, vis


def vis_multiplex_network(multiplex_nw, UMAP_coord = None, display_feature = 'hub', title_size = 16,\
                           para_vis_UMAP ={ 'para_UMAP': {'random_state':111, 'n_neighbors':50}, \
             'para_plot' : {'figsize':(5,5), 'dpi':100, 'cmap':None}, 'colorbar' : True}):
    
    assert len(multiplex_nw.multiplex_property) != 0, print('multiplex_nw.multiplex_property is empty!')
    assert len(multiplex_nw.community_detection)!= 0, print('multiplex_nw.community_detection is empty!')
    
    fig_lst = [] 
    vis_lst = [] 
    membership = multiplex_nw.community_detection['MVP'][0] 
    gs_length = len(membership)
    tissue_ns = copy.deepcopy(multiplex_nw.layer_ns)
    tissue_ns.insert(0, 'Jaccard')
    for i in range(len(tissue_ns)):
        if i == 0:
            if UMAP_coord is not None:
                cor = UMAP_coord[i]
                adj_tr = None
            else:
                cor = None
                adj_tr = multiplex_nw.adj_Jaccard
                
            if display_feature == 'hub':
                ### ratio
                feature = multiplex_nw.multiplex_property['Jaccard_hub']
            elif display_feature == "pagerank":
                feature = multiplex_nw.multiplex_property['Jaccard_page_rank']/(1/gs_length)
            elif display_feature == 'membership':
                feature = membership
            else:
                print('The display_feature is not supported currently!')
                return
        else:
            if UMAP_coord is not None:
                cor = UMAP_coord[i]
                adj_tr = None
            else:
                cor = None
                adj_tr =  multiplex_nw.adj_filtered_lst[i-1]
            ### ratio
            if display_feature == 'hub':
                feature =  multiplex_nw.multiplex_property['per_layer_hub'][i-1]
            elif display_feature == 'pagerank':
                feature =  multiplex_nw.multiplex_property['per_layer_page_rank'][i-1]/(1/gs_length)
            elif display_feature == 'membership':
                feature = membership
            else:
                print('The display_feature is not supported currently!')
                return
            
        if display_feature == 'membership':
            cmap = 'tab10'
        else:
            cmap = 'Spectral'
        para_vis_UMAP['para_plot']['cmap'] = cmap
        
        fig_temp, vis_temp = vis_UMAP(adj_tr, UMAP_coord = cor, title = tissue_ns[i] , feature = feature,\
                                                    title_size =  title_size, **para_vis_UMAP)
        fig_lst.append(fig_temp)
        vis_lst.append(vis_temp)
        
    return fig_lst, vis_lst


def vis_scatter(para_1, para_2, label_1, label_2, para_jointplot = {'kind':'hex', 'space':0.7, 'marginal_kws':dict(bins=30)}, xtick_integer = False, ytick_integer = False,\
               anno_font_size = 16, height = 8, ratio = 10):
    df = pd.DataFrame(data = {label_1: para_1, label_2 : para_2})
    with sns.axes_style('white'):
        sns_plot = sns.jointplot(x=label_1, y= label_2, data=df, 
                     **para_jointplot, height=height, ratio=ratio)
        sns_plot.set_axis_labels(label_1, label_2, fontsize= anno_font_size)
        if not xtick_integer:
            sns_plot.ax_joint.set_xticklabels([str(np.round(i,2)) for i in sns_plot.ax_joint.get_xticks()], fontsize = anno_font_size)
        else:
            sns_plot.ax_joint.set_xticklabels([str(np.int(i)) for i in sns_plot.ax_joint.get_xticks()], fontsize = anno_font_size)
        if not ytick_integer:
            sns_plot.ax_joint.set_yticklabels([str(np.round(i,2)) for i in sns_plot.ax_joint.get_yticks()], fontsize = anno_font_size)
        else:
            sns_plot.ax_joint.set_yticklabels([str(np.int(i)) for i in sns_plot.ax_joint.get_yticks()], fontsize = anno_font_size)
        sns_plot.figure.tight_layout()
    return sns_plot

def vis_gsea(gsea_df, rank_label, rank_dict, gs_ns, cross_effect = False, top = 10, reverse = False, colorbar = 'Spectral_r', x_label = None, ax_title = "GO-CC", fig_size = (6,6), title_size = 20, anno_font_size = 16):

    enrich_df = copy.deepcopy(gsea_df)
    term = list(enrich_df['Term'])
    ### remove gene sets not in the filtered list
    for item in term:
        if item not in gs_ns:
            print(f'{item} is not in the list of filtered gene set.')
            term.remove(item)
    enrich_df = enrich_df[enrich_df["Term"].isin(term)]
     
    if rank_dict is not None:
        enrich_df[rank_label] = [rank_dict[item] for item in term]
    if reverse:
        enrich_df = enrich_df.iloc[(enrich_df[rank_label].abs()).argsort()]
    else:
        enrich_df = enrich_df.iloc[(-enrich_df[rank_label].abs()).argsort()]

    cm = mcp.gen_color_normalized(cmap=colorbar,data_arr= enrich_df[rank_label])
    enrich_df[rank_label] = enrich_df[rank_label].abs()
    
    plt.figure(figsize = fig_size)
    ax = sns.barplot(data = enrich_df[:top], x = rank_label, y = 'Term', palette = cm)
    ax.set_yticklabels(enrich_df[:top]['Term'], fontsize= anno_font_size)
    ax.set_ylabel("")
    if x_label == None:
        x_label = rank_label
    ax.set_xlabel(x_label, fontsize= anno_font_size)
    ax.set_title(ax_title, fontsize= title_size)
    ax.figure.tight_layout()
    
    return enrich_df, ax
    
def vis_surface(sample_size, gs_size, figsize = (12, 10), unit = 'second',\
                fontsize = 16, rotation = 60, dpi=300, alpha=.7, legend_pos = (1.2, 1.0), elevation_rotation = [20, -60]):
    assert np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True) is not None, "The data path to res files is not right!"
    
    X, Y = np.meshgrid(sample_size, gs_size)
    
    def fetch_time(x,y,i, unit = unit):
        temp = np.load(f"res_{x}_{y}.npy", allow_pickle = True)
        time = temp[0][i][1]
        
        if len(label) == 0:
            label.append(temp[0][i][0])
        if unit == 'minite':
            time = np.round(time/60, 3)
        elif unit == 'hour':
            time = np.round(time/(60*60), 3)
        return time

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= figsize, dpi=dpi)
    ### number of methods
    length = len(np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[1])
    for i in range(length):
        label = []
        ### time
        Z = np.array([fetch_time(x,y,i) for x in sample_size for y in gs_size]).reshape(len(sample_size),-1).transpose()
        surf = ax.plot_surface(X, Y, Z, label = label[0], alpha = alpha)
        surf._facecolors2d=surf._facecolor3d
        surf._edgecolors2d=surf._edgecolor3d

    fig.suptitle("Running time comparison", fontsize = fontsize)
    ax.set_xlabel("sample size", fontsize=fontsize, rotation=rotation)
    ax.set_ylabel("gene set size", fontsize=fontsize, rotation=rotation)
    ax.set_zlabel(f'running time in {unit}s', fontsize=fontsize, rotation=rotation)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.legend(bbox_to_anchor= legend_pos)
    ax.view_init(*elevation_rotation)
    fig.tight_layout()
    return fig

def vis_surface_correlation(sample_size, gs_size, reference = "modified RV", cor_mat_lst = None, figsize = (12, 10),\
                fontsize = 16, rotation = 60, dpi=300, alpha=.7, legend_pos = (1.2, 1.0), elevation_rotation = [20, -60]):
    assert np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True) is not None, "The data path to res files is not right!"
    
    X, Y = np.meshgrid(sample_size, gs_size)
    num_method = len(np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[0])
    
    if cor_mat_lst is None:
        cor_mat_lst = []
        for ss in sample_size:
            for gs in gs_size:
                res = np.load(f"res_{ss}_{gs}.npy", allow_pickle = True)
                ref = np.load(f"res_{sample_size[-1]}_{gs}.npy", allow_pickle = True)
                cor_mat = np.array([(lambda i,j: scipy.stats.pearsonr(res[1][i][np.triu_indices_from(res[1][i], k = 1)],\
                                            ref[1][j][np.triu_indices_from(ref[1][j], k = 1)])[0])(i,j) for i in range(num_method)\
                  for j in range(num_method)]).reshape(num_method, num_method)
                cor_mat_lst.append(cor_mat)
    
    def fetch_value(x,y,i):
        temp = cor_mat_lst[x*len(gs_size) + y]
        if reference == "modified RV":
            return temp[i,0]
        elif reference == "Mantel":
            return temp[i,num_method-1]
        else:
            raise Exception("Reference is not correct!")
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= figsize, dpi=dpi)
    ### number of methods
    length = len(np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[1])
    label = [np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[0][i][0] for i in range(num_method)]
    for i in range(length):
        Z = np.array([fetch_value(x,y,i) for x in range(len(sample_size)) for y in range(len(gs_size))]).reshape(len(sample_size),-1).transpose()
        surf = ax.plot_surface(X, Y, Z, label = label[i], alpha = alpha)
        surf._facecolors2d=surf._facecolor3d
        surf._edgecolors2d=surf._edgecolor3d

    fig.suptitle(f"Compare with the {reference} coefficient", fontsize = fontsize)
    ax.set_xlabel("sample size", fontsize=fontsize, rotation=rotation)
    ax.set_ylabel("gene set size", fontsize=fontsize, rotation=rotation)
    ax.set_zlabel(f'Pearson correlation coefficient', fontsize=fontsize, rotation=rotation)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.legend(bbox_to_anchor= legend_pos)
    ax.view_init(*elevation_rotation)
    fig.tight_layout()
    return fig, cor_mat_lst
    
    
def vis_3D_umap(umap_gs_E, umap_gs_F, floor_label,  roof_label, title, font_size = 18, title_size = 18, line_alpha = 0.02, colorbar = "tab10", floor = 1, roof = 5, z_tick_pad = 100, view_para = {'elev':30., 'azim':-60}, figsize = (6,6), \
                floor_theta = 0, roof_theta = 0):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= figsize)
    ### feature colors
    cmap = plt.colormaps[colorbar]
    ### rescale
    mat_E = umap_gs_E.to_numpy()[:,:2]
    mat_F = umap_gs_F.to_numpy()[:,:2]
    mat_E = (mat_E-np.min(mat_E, axis = 0))/(np.max(mat_E, axis = 0) - np.min(mat_E, axis = 0))
    mat_F = (mat_F-np.min(mat_F, axis = 0))/(np.max(mat_F, axis = 0) - np.min(mat_F, axis = 0))
    ### rotations
    rotation_floor = np.array([[np.cos(floor_theta), -np.sin(floor_theta)],
         [np.sin(floor_theta), np.cos(floor_theta)]])
    rotation_roof = np.array([[np.cos(roof_theta), -np.sin(roof_theta)],
         [np.sin(roof_theta), np.cos(roof_theta)]])
    mat_F = mat_F@rotation_floor.transpose()
    mat_E = mat_E@rotation_roof.transpose()
    ### centering
    center_F = np.mean(mat_F, axis = 0)
    center_E = np.mean(mat_E, axis = 0)
    mat_F -= center_F
    mat_E -= center_E
    ### two layers
    ax.scatter(mat_F[:,0], mat_F[:,1], zs = floor, zdir='z', label='UMAP', c = cmap(umap_gs_F['feature']), alpha = 0.2,zorder = 1)
    ax.scatter(mat_E[:,0], mat_E[:,1], zs = roof, zdir='z', label='UMAP', c = cmap(umap_gs_E['feature']), alpha = 0.8, zorder = 10)
    ### line segments
    for i in range(mat_E.shape[0]):
        ax.plot([mat_E[:,0][i], mat_F[:,0][i]], [mat_E[:,1][i],mat_F[:,1][i]],zs=[roof,floor], alpha = line_alpha, zorder = 10)
    # on the plane y=0
    ax.view_init(**view_para)
    ax.grid(False)
    # Hide the x,y 'rulers'
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([floor,roof],[floor_label,roof_label], minor = False)
    ax.set_xlabel('UMAP1', loc = 'center', fontsize=font_size)
    ax.set_ylabel('UMAP2', loc = 'center', fontsize=font_size)
    ax.set_title(title, fontsize = title_size)
    ax.tick_params(axis='z',pad = z_tick_pad,labelsize = font_size)
    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    fig.tight_layout()
    return fig

    
    
    
    