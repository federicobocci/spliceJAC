'''
plotting resources
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

import networkx as nx
from scipy.sparse.csgraph import connected_components
from pyvis.network import Network

from . import plotting_util


def plot_setup(adata):
    '''Assign a color to each cluster
    Results are stored in adata.uns['colors']

    Parameters
    ----------
    adata: anndata object of gene counts

    Returns
    -------
    None

    '''
    clusters = list(adata.obs['clusters'])
    types = sorted(list(set(clusters)))
    colors = list(plt.cm.Set2.colors)[0:len(types)]
    adata.uns['colors'] = {types[i]: colors[i] for i in range(len(types))}


def umap_scatter(adata,
                 ax=None,
                 order=None,
                 axis=False,
                 fontsize=10,
                 alpha=0.5,
                 show_cluster_center=True,
                 s=2,
                 legens_pos=(0.5, 1.2),
                 legend_loc='upper center',
                 ncol=4,
                 figsize=(4, 4),
                 showfig=False,
                 savefig=True,
                 figname='umap_scatter.pdf',
                 format='pdf'):
    '''2D UMAP plot of the data

    Parameters
    ----------
    adata: anndata object of gene counts
    ax: pyplot axis, if False generate a new figure (default=False)
    order: order of cluster labels in the figure legend, if None the order is random (default=None)
    axis: if true, draw axes, otherwise do not show axes (default=False)
    fontsize: fontsize of axes and legend labels
    alpha: shading of inidividual cells
    show_cluster_center: if True, plot the center of each cluster
    s: size of individual cells
    legens_pos: position of figure legend by axis coordinates
    legend_loc: position of figure legend
    ncol: number of columns in the figure legend
    figsize: size of figure
    showfig: if True, show the figure
    savefig: if True, save the figure
    figname: name of saved figure (any path for figure saving should be added gere)
    format: figure format

    Returns
    -------
    None

    '''

    if 'colors' not in adata.uns.keys():
        plot_setup(adata)

    if order!=None:
        types = order
    else:
        clusters = list(adata.obs['clusters'])
        types = sorted(list(set(clusters)))

    plot_figure=True
    if ax!=None:
        plot_figure=False

    if ax==None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)

    x_umap, y_umap = np.zeros(len(types)), np.zeros(len(types))
    for i in range(len(types)):
        sel_data = adata[adata.obs['clusters'] == types[i]]
        umap_sel = np.asarray(sel_data.obsm['X_umap'])
        ax.scatter(umap_sel[:, 0], umap_sel[:, 1], c=[adata.uns['colors'][types[i]] for j in range(umap_sel[:, 0].size)], s=s, alpha=alpha)
        x_umap[i], y_umap[i] = np.mean(sel_data.obsm['X_umap'][:,0]), np.mean(sel_data.obsm['X_umap'][:,1])

    if show_cluster_center:
        plt.scatter(x_umap, y_umap, c=list(adata.uns['colors'].values()), s=50, edgecolors='k', linewidths=0.5)
    patches = [mpatches.Patch(color=adata.uns['colors'][types[i]], label=types[i]) for i in range(len(types))]
    plt.legend(bbox_to_anchor=legens_pos, handles=patches, loc=legend_loc, ncol=ncol, fontsize=fontsize)

    if axis:
        plt.xlabel('$X_{UMAP}$', fontsize=fontsize)
        plt.ylabel('$Y_{UMAP}$', fontsize=fontsize)
    else:
        plt.axis('off')


    if plot_figure:
        plt.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(figname, format=format, dpi=300)



def plot_grn(G, node_size=200, edge_width=1, font_size=8, adata=None, node_color='b', cluster_name=None, pos_style='spring',
             base_node_size=300, diff_node_size=600, pos_edge_color='b', neg_edge_color='r', arrowsize=10,
             arrow_alpha=0.75, conn_style='straight', colorbar=True):
    '''Plot the GRN with positive and negative interactions

    Parameters
    ----------
    G: networkx graph object
    node_size: size of nodes. If node_size='expression', the node size is scaled based on gene expression. Otherwise, node_size can be an integer (default=200)
    edge_width: width of connection arrows
    font_size: font size for node labels
    adata: gene count anndata object, must be provided if node_size == 'expression'
    node_color: color of nodes. If node_color='centrality', the color is based on the node centrality. Otherwise,  a matplotlib color name can be provided (default='b')
    cluster_name: name of considered cluster (must be provided if node_size='expression')
    pos_style: position of nodes. The options are 'spring' and 'circle'. 'spring' uses the networkx spring position function, whereas 'circle' arranges the nodes in a circle.
    base_node_size: Minimum node size (used if node_size='expression')
    diff_node_size: difference betwene minimum and maximal node size (used if node_size='expression')
    pos_edge_color: color for positive regulation arrow
    neg_edge_color: color for negative regulation arrow
    arrowsize: size of interaction arrows
    arrow_alpha: shading of interaction arrows
    conn_style: style of interaction arrows
    colorbar: if True, show colorbar (required if node_size='expression')

    Returns
    -------
    None

    '''
    assert pos_style=='spring' or pos_style=='circle', "Please choose between pos_style=='spring' or pos_style=='circle'"

    if pos_style == 'spring':
        pos = nx.spring_layout(G, seed=0)
    else:
        pos = circle_pos(G)

    if conn_style=='straight':
        connectionstyle='arc3'
    elif conn_style=='curved':
        connectionstyle = 'arc3, rad=0.1'

    if node_size == 'expression':
        node_size = np.array((diff_node_size*NormalizeData(robust_mean(adata[adata.obs['clusters'].isin([cluster_name]), list(G)].X.toarray())) + base_node_size)).flatten()

    if node_color == 'centrality':
        centrality = nx.betweenness_centrality(G, k=10, endpoints=True)
        node_color = list(centrality.values())


    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.5, cmap=plt.cm.viridis)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array([])
    if colorbar:
        cbar = plt.colorbar(sm, label='Betweenness Centrality')

    epos = []
    eneg = []
    wpos = []
    wneg = []

    for (u, v, d) in G.edges(data=True):
        if d["weight"] > 0:
            epos.append((u, v))
            wpos.append(d["weight"])

        elif d["weight"] < 0:
            eneg.append((u, v))
            wneg.append(d["weight"])

    if edge_width == 'weight':
        edge_width_pos = NormalizeData(np.array(wpos))
        edge_width_neg = NormalizeData(-np.array(wneg))
    else:
        edge_width_pos = edge_width
        edge_width_neg = edge_width

    nx.draw_networkx_edges(G, pos, edgelist=epos, width=edge_width_pos + 0.5, edge_color=pos_edge_color, arrowsize=arrowsize, alpha=arrow_alpha, connectionstyle=connectionstyle)
    nx.draw_networkx_edges(G, pos, edgelist=eneg, width=edge_width_neg + 0.5, edge_color=neg_edge_color, arrowstyle="-[", arrowsize=arrowsize, alpha=arrow_alpha, connectionstyle=connectionstyle)
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_family="sans-serif")


def visualize_network(adata, cluster_name, genes=None, cc_id=0, node_size='expression', edge_width='weight', font_size=10,
                      plot_interactive=True, weight_quantile=.5, node_color=None, pos_style='spring', title=True, base_node_size=300, diff_node_size=600,
                      pos_edge_color='b', neg_edge_color='r', arrowsize=10, arrow_alpha=0.75, conn_style='straight', colorbar=True,
                      showfig=False, savefig=True, figname='core_GRN.pdf', format='pdf', figsize=(4, 3)):
    '''
    plot the connected grn network

    '''

    assert conn_style=='straight' or conn_style=='curved', "Please choose between conn_style=='straight' and conn_style=='curved'"

    if genes == None:
        genes = list(adata.var_names)
    n = len(genes)
    A = adata.uns['average_jac'][cluster_name][0][0:n, n:].copy().T

    q_pos = np.quantile(A[A > 0], weight_quantile)
    q_neg = np.quantile(A[A < 0], 1 - weight_quantile)
    A[(A > q_neg) & (A < q_pos)] = 0

    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.relabel_nodes(G, dict(zip(range(len(genes)), genes)), copy=False)

    G_undirected = G.to_undirected()
    subgraphs = [G.subgraph(c) for c in sorted(nx.connected_components(G_undirected), key=len, reverse=True) if
                 len(c) > 1]

    if len(subgraphs) > 1:
        print("There exist multiple connected components. Choose the parameter cc_id to show other components")

    fig, ax = plt.subplots(figsize=figsize)
    plot_grn(subgraphs[cc_id], node_size=node_size, edge_width=edge_width, font_size=font_size, adata=adata,
             node_color=node_color, cluster_name=cluster_name, pos_style=pos_style, base_node_size=base_node_size, diff_node_size=diff_node_size,
             pos_edge_color=pos_edge_color, neg_edge_color=neg_edge_color, arrowsize=arrowsize, arrow_alpha=arrow_alpha, conn_style=conn_style, colorbar=colorbar)

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    if title:
        plt.title(cluster_name+' core GRN')

    # Resize figure for label readibility
    fig.tight_layout()
    plt.axis("off")
    if showfig:
        plt.show()

    if plot_interactive:
        nt = Network(notebook=True, directed=True)
        nt.from_nx(subgraphs[cc_id])
        display(nt.show('nx.html'))
    if savefig:
        plt.savefig(figname, format=format)



def diff_network(adata, cluster1, cluster2, genes=None, weight_quantile=0.95, cc_id=0, edge_width='weight', conn_style='straight',
                 title=True, figsize=(3.5, 3), colorbar=False, savefig=True, figname='diff_grn.pdf', format='pdf'):

    assert conn_style == 'straight' or conn_style == 'curved', "Please choose between conn_style=='straight' and conn_style=='curved'"

    if genes == None:
        genes = list(adata.var_names)
    n = len(genes)

    A1 = adata.uns['average_jac'][cluster1][0][0:n, n:].copy().T
    A2 = adata.uns['average_jac'][cluster2][0][0:n, n:].copy().T
    A = A2 - A1

    q_pos = np.quantile(A[A > 0], weight_quantile)
    q_neg = np.quantile(A[A < 0], 1 - weight_quantile)
    A[(A > q_neg) & (A < q_pos)] = 0

    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.relabel_nodes(G, dict(zip(range(len(genes)), genes)), copy=False)

    G_undirected = G.to_undirected()
    subgraphs = [G.subgraph(c) for c in sorted(nx.connected_components(G_undirected), key=len, reverse=True) if
                 len(c) > 1]

    if len(subgraphs) > 1:
        print("There exist multiple connected components. Choose the parameter cc_id to show other components")

    fig, ax = plt.subplots(figsize=figsize)

    plot_grn(subgraphs[cc_id], edge_width=edge_width, colorbar=colorbar)

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    if title:
        plt.title(cluster1 + '-' + cluster2 + ' differential GRN')

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)



def diff_interactions(adata, cluster1, cluster2, top_int=10,
                      showfig=False, savefig=True, figname='diff_interactions.pdf', format='pdf'):
    genes = list(adata.var_names)
    n = len(genes)

    A1 = adata.uns['average_jac'][cluster1][0][0:n, n:].copy().T
    A2 = adata.uns['average_jac'][cluster2][0][0:n, n:].copy().T
    A = A2 - A1

    y = np.ndarray.flatten(A)
    sorted = np.flip(np.argsort( np.abs(y) ))

    int_list, name = [], []
    color = []
    for i in range(top_int):
        k, l = np.unravel_index(sorted[i], shape=(n,n))
        # print( sorted[i], genes[k], genes[l], A[k][l] )
        int_list.append(A[k][l])
        name.append( genes[k] + '-' + genes[l] )

        # set qualitative change of interaction
        if A1[k][l]>0 and A2[k][l]>0:
            color.append('r')
        elif A1[k][l]>0 and A2[k][l]<0:
            color.append('b')
        if A1[k][l]<0 and A2[k][l]>0:
            color.append('g')
        if A1[k][l]<0 and A2[k][l]<0:
            color.append('c')
        else:
            print('interaction color undefined')


    fig = plt.figure()

    plt.barh( np.arange(1, len(int_list)+1, 1), int_list, color=color )
    plt.yticks( np.arange(1, len(int_list)+1, 1), name )
    plt.xlabel('Interaction strenght change')


    plt.title( cluster1 + '-' + cluster2 + ' differential interactions' )
    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format)



def robust_mean(X):
    return 0.25 * np.quantile(X, q=0.75, axis=0) + 0.25 * np.quantile(X, q=0.25, axis=0) + 0.5 * np.quantile(X, q=0.5,
                                                                                                             axis=0)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def circle_pos(G):
    nodes = list(G.nodes)
    center, radius = [0,0], 1

    pos = {}
    angle = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)

    for i in range(angle.size):
        coord = np.array([ center[0]+np.sin(angle[i]*radius), center[1]+np.cos(angle[i]*radius) ])
        pos[nodes[i]]=coord
    return pos


# def aggregate_network(adata, pos_style='spring', cc_id=0):
#     int_df = adata.uns['aggregated_grn']
#
#     grn = int_df[['Gene1', 'Gene2', 'sign', 'cluster', 'color']]
#     G = nx.from_pandas_edgelist(grn, source='Gene1', target='Gene2', create_using=nx.DiGraph(), edge_attr=['sign', 'cluster', 'color'])
#
#     G_undirected = G.to_undirected()
#     subgraphs = [G.subgraph(c) for c in sorted(nx.connected_components(G_undirected), key=len, reverse=True) if len(c) > 1]
#     G = subgraphs[cc_id]
#
#     if pos_style=='spring':
#         pos = nx.spring_layout(G, seed=0)
#     else:
#         pos = circle_pos(G)
#
#     fig = plt.figure(figsize=(8,8))
#     ax = plt.subplot(111)
#     for g in G.edges(data=True):
#         if g[2]['sign']=='positive':
#             arr_style='-|>'
#             nx.draw_networkx_edges(G, pos, edgelist=[g], connectionstyle='arc3, rad=0.1', width=2, arrowstyle=arr_style, edge_color=g[2]['color'])
#         elif g[2]['sign']=='negative':
#             arr_style='-['
#             nx.draw_networkx_edges(G, pos, edgelist=[g], connectionstyle='arc3, rad=0.1', width=2, arrowstyle=arr_style, edge_color=g[2]['color'], style="dashed")
#     nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightskyblue', alpha=0.5)
#     nx.draw_networkx_labels(G, pos)
#
#     df_cp = int_df[ int_df['cluster']!='NA' ]
#     labels, cols = list(df_cp.cluster.unique()), df_cp.color.unique()
#     patches = [ mpatches.Patch(color=cols[i], label=labels[i]) for i in range(len(labels)) ]
#     plt.legend(bbox_to_anchor=(0.5, 1.1), handles=patches, loc='upper center', ncol=4)
#
#     # plt.ylim([-1.2, 1.5])
#     # print(labels)
#
#
#     plt.show()



### plotting of transitioning genes

def plot_trans_genes(adata, cluster1, cluster2, top_trans_genes=10, fontsize=10, cbar='r', alpha=0.5, title=True, showfig=False, savefig=True, figname='trans_genes.pdf', format='pdf', figsize=(3,3)):

    weight = adata.uns['transitions'][cluster1 + '-' + cluster2]['weights']
    genes = list(adata.var_names)

    ind = np.argsort(weight)

    data, trans_genes = [], []
    for i in range(top_trans_genes):
        trans_genes.append( genes[ind[weight.size - top_trans_genes + i]] )
        data.append( weight[ind[weight.size - top_trans_genes + i]]  )

    fig = plt.figure(figsize=figsize)
    plt.barh(np.arange(1, len(data) + 1, 1), data, height=0.8, align='center', color=cbar, edgecolor='k', linewidth=1, alpha=alpha)
    plt.yticks(np.arange(1, len(data) + 1, 1), trans_genes, fontsize=fontsize)
    plt.xlabel('Instability score', fontsize=fontsize)
    if title:
        plt.title(cluster1 + ' to ' + cluster2, fontsize=fontsize)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)



def scatter_scores(adata, cluster1, cluster2, top_deg_genes=5, top_trans_genes=5, showfig=False, savefig=True, figname='unst_genes.pdf', format='pdf', figsize=(3,3)):

    # get gene instability scores along transition direction
    weight = adata.uns['transitions'][cluster1 + '-' + cluster2]['weights']

    genes = list(adata.var_names)
    # ind = np.flip(np.argsort(proj))
    # ind = np.flip(np.argsort(weight))
    # trans_genes = [genes[j] for j in ind[0:top_trans_genes]]

    # get DEGs for outgoing cluster
    deg = adata.uns['rank_genes_groups']['names'][cluster1][0:top_deg_genes]

    deg_genes = adata.uns['rank_genes_groups']['names'][cluster1]
    scores = adata.uns['rank_genes_groups']['scores'][cluster1]
    ord_weight = []
    for d in deg_genes:
        i = genes.index(d)
        ord_weight.append(weight[i])

    plt.figure(figsize=figsize)
    plt.scatter(scores, ord_weight, c='b')
    plt.xlabel('DEG score')
    plt.ylabel('Instability score')
    plt.title(cluster1 + ' to ' + cluster2)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)


    # # get interactions between selected nodes
    # genes = list(adata.var_names)
    # J = adata.uns['average_jac'][cluster1][0][0:len(genes), len(genes):].copy()



############################
#
### plotting of reduced GRNs
#
############################

def subset_jacobian(J, genes, genelist):
    ind = []
    for s in genelist:
        ind.append(genes.index(s))

    m = int(J.shape[0]/2)
    B = np.transpose(J[0:m, m:])
    B_subset = B[ind][:, ind]
    return B_subset

def core_GRN(adata,
             cluster1,
             cluster2,
             type_color=['orange', 'plum', 'yellowgreen'],
             pos_edge_color='b',
             neg_edge_color='r',
             node_size=500,
             node_alpha=0.5,
             arrowsize=10,
             arrow_alpha=0.75,
             conn_style='arc3, rad=0.1',
             node_font=8,
             legend=True,
             legend_font=10,
             legend_ncol=2,
             legend_loc='lower center',
             axis=False,
             xlim=[-1.2, 1.2],
             ylim=None,
             showfig=False,
             savefig=True,
             figname='core_GRN.pdf',
             format='pdf',
             figsize=(3,3)
             ):
    deg_list, tg_list, both_list = adata.uns['transitions'][cluster1 + '-' + cluster2]['gene_lists']

    # 1) select reduced Jacobian with top DEG and TG
    sel_genes = deg_list+tg_list+both_list
    node_color = [type_color[0] for d in deg_list] + [type_color[1] for t in tg_list] + [type_color[2] for b in both_list]

    B_core = subset_jacobian(adata.uns['average_jac'][cluster1][0], list(adata.var_names), sel_genes)

    # 2) filter number of edges
    A = B_core.copy()
    q_pos = np.quantile(A[A > 0], 0.5)
    q_neg = np.quantile(A[A < 0], 1 - 0.5)
    A[(A > q_neg) & (A < q_pos)] = 0

    # 3) create graph from Jacobian matrix
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.relabel_nodes(G, dict(zip(range(len(sel_genes)), sel_genes)), copy=False)
    pos = circle_pos(G)

    # 4) select positive and negative interactions for separate arrow plotting
    epos = []
    eneg = []
    wpos = []
    wneg = []

    for (u, v, d) in G.edges(data=True):
        if d["weight"] > 0:
            epos.append((u, v))
            wpos.append(d["weight"])

        elif d["weight"] < 0:
            eneg.append((u, v))
            wneg.append(d["weight"])

    # 5) plotting
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=node_alpha)
    nx.draw_networkx_edges(G, pos, edgelist=epos, edge_color=pos_edge_color, arrowsize=arrowsize, alpha=arrow_alpha, connectionstyle=conn_style)
    nx.draw_networkx_edges(G, pos, edgelist=eneg, edge_color=neg_edge_color, arrowstyle="-[", arrowsize=arrowsize, alpha=arrow_alpha, connectionstyle=conn_style)

    nx.draw_networkx_labels(G, pos, font_size=node_font)

    plt.xlim(xlim)
    if ylim==None:
        ylim = [-1.6, 1.2] if legend else [-1.2, 1.2]
    plt.ylim(ylim)

    if legend:
        labels = [cluster1 + ' DEG', cluster1 + ' to ' + cluster2 + ' TG','DEG+TG']
        patches = [mpatches.Patch(color=type_color[i], label=labels[i]) for i in range(len(labels))]
        plt.legend(bbox_to_anchor=(0.5, -0.1), handles=patches, loc=legend_loc, ncol=legend_ncol, fontsize=legend_font)

    if not(axis):
        plt.axis("off")

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)




def bif_GRN(adata, start, end, pos_edge_color='b',
             neg_edge_color='r',
             node_size=750,
             node_alpha=0.5,
             arrowsize=10,
             arrow_alpha=0.75,
             arrowstyle='arc3, rad=0.1',
             node_font=10,
             legend=True,
             legend_font=10,
             legend_ncol=3,
             legend_loc='lower center',
             axis=True,
             xlim=[-1.2, 1.2],
             ylim=None,
             showfig=False,
             savefig=True,
             figname='bif_GRN.pdf',
             format='pdf',
             figsize=(6,6)
            ):
    types = sorted(list(set(list(adata.obs['clusters']))))
    colors = list(plt.cm.Set3.colors)[0:len(types)]

    # get TG of each cluster
    tg, cluster = [], []
    for c in end:
        deg_list, tg_list, both_list = adata.uns['transitions'][start + '-' + c]['gene_lists']
        tg = tg + tg_list + both_list
        cluster = cluster + [c for i in range(len(tg_list) + len(both_list))]

    tg_sel, cluster_sel, node_color = [], [], []
    for i in range(len(tg)):
        if tg[i] not in tg_sel:
            tg_sel.append(tg[i])
            cluster_sel.append(cluster[i])
            node_color.append( colors[types.index(cluster[i])] )
        else:
            j = tg_sel.index(tg[i])
            if cluster_sel[j]!='shared':
                cluster_sel[j] = 'shared'
                node_color[j] = 'lightgray'

    B_core = subset_jacobian(adata.uns['average_jac'][start][0], list(adata.var_names), tg_sel)

    # 2) filter number of edges
    A = B_core.copy()
    q_pos = np.quantile(A[A > 0], 0.5)
    q_neg = np.quantile(A[A < 0], 1 - 0.5)
    A[(A > q_neg) & (A < q_pos)] = 0

    # 3) create graph from Jacobian matrix
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.relabel_nodes(G, dict(zip(range(len(tg_sel)), tg_sel)), copy=False)
    pos = circle_pos(G)

    # 4) select positive and negative interactions for separate arrow plotting
    epos = []
    eneg = []
    wpos = []
    wneg = []

    for (u, v, d) in G.edges(data=True):
        if d["weight"] > 0:
            epos.append((u, v))
            wpos.append(d["weight"])

        elif d["weight"] < 0:
            eneg.append((u, v))
            wneg.append(d["weight"])

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=node_alpha)
    nx.draw_networkx_edges(G, pos, edgelist=epos, edge_color=pos_edge_color, arrowsize=arrowsize, alpha=arrow_alpha, connectionstyle=arrowstyle)
    nx.draw_networkx_edges(G, pos, edgelist=eneg, edge_color=neg_edge_color, arrowstyle="-[", arrowsize=arrowsize, alpha=arrow_alpha, connectionstyle=arrowstyle)

    nx.draw_networkx_labels(G, pos, font_size=node_font)

    plt.xlim(xlim)
    if ylim == None:
        ylim = [-1.6, 1.2] if legend else [-1.2, 1.2]
    plt.ylim(ylim)

    if legend:
        labels = [c+' TG' for c in end] + ['Shared TG']
        type_color = []
        for c in end:
            j = types.index(c)
            type_color.append(colors[j])
        type_color.append('lightgray')

        patches = [mpatches.Patch(color=type_color[i], label=labels[i]) for i in range(len(labels))]
        plt.legend(handles=patches, loc=legend_loc, ncol=legend_ncol, fontsize=legend_font)

    if not(axis):
        plt.axis("off")

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)


### functions to plot signaling hubs and signaling changes

def plot_signaling_hubs(adata, cluster, top_genes=5, criterium='weights', showfig=False, savefig=True, figname='signaling_hub.pdf', format='pdf', figsize=(3.5,3)):

    assert criterium=='edges' or criterium=='weights', "Please choose between criterium=='edges' or criterium=='weights'"

    if criterium=='weights':
        rec, sen = adata.uns['signaling_scores'][cluster]['incoming'], adata.uns['signaling_scores'][cluster]['outgoing']
    elif criterium=='edges':
        rec, sen = adata.uns['signaling_scores'][cluster]['incoming_edges'], adata.uns['signaling_scores'][cluster]['outgoing_edges']
    genes = list(adata.var_names)

    # gene expression for colormap
    S = adata[adata.obs['clusters'] == cluster].layers['spliced'].toarray()
    avgS = np.mean(S, axis=0)
    cmap_lim = np.amax(avgS)  ### set this colormap to log scale?

    tot = rec+sen
    sort_ind = np.argsort(tot)
    x, y, lab = rec[sort_ind[rec.size - top_genes:]], sen[sort_ind[rec.size - top_genes:]], []

    for i in sort_ind[rec.size - top_genes:]:
        lab.append(genes[i])

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    pt = plt.scatter(rec, sen, c=avgS, cmap='Reds', vmin=0, vmax=cmap_lim, edgecolors='k', linewidths=0.5)
    plt.colorbar(pt, label='Expression level')
    # plt.scatter(x, y, c='r')

    for i in range(x.size):
        ax.annotate(lab[i], (x[i], y[i]))

    plt.xlim([-0.1, 1.25*np.amax(rec)])
    plt.ylim([-0.1, 1.1*np.amax(sen)])
    plt.xlabel('Incoming score')
    plt.ylabel('Outgoing score')
    plt.title(cluster)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)



def plot_signaling_change(adata, cluster1, cluster2, top_genes=10, criterium='weights', logscale_scores=True, logscale_fc=True,  x_shift=0.05, y_shift=0.05,
                          showfig=False, savefig=True, figname='signaling_hub_change.pdf', format='pdf', figsize=(3.5,3)):

    assert criterium == 'edges' or criterium == 'weights', "Please choose between criterium=='edges' or criterium=='weights'"
    genes = list(adata.var_names)

    if criterium == 'weights':
        rec1, sen1 = adata.uns['signaling_scores'][cluster1]['incoming'], adata.uns['signaling_scores'][cluster1][
            'outgoing']
        rec2, sen2 = adata.uns['signaling_scores'][cluster2]['incoming'], adata.uns['signaling_scores'][cluster2][
            'outgoing']
    elif criterium == 'edges':
        rec1, sen1 = adata.uns['signaling_scores'][cluster1]['incoming_edges'], adata.uns['signaling_scores'][cluster1][
            'outgoing_edges']
        rec2, sen2 = adata.uns['signaling_scores'][cluster1]['incoming_edges'], adata.uns['signaling_scores'][cluster1][
            'outgoing_edges']

    delta_rec, delta_sen = rec2-rec1, sen2-sen1


    # compute gene expression fold-change between from cluster 1 to cluster 2
    data1, data2 = adata[adata.obs['clusters'] == cluster1], adata[adata.obs['clusters'] == cluster2]
    S1, S2 = data1.layers['spliced'].toarray(), data2.layers['spliced'].toarray()
    avgS1, avgS2 = np.mean(S1, axis=0), np.mean(S2, axis=0)
    fc = (avgS2-avgS1)/avgS1
    if logscale_fc:
        fc = np.sign(fc)*np.log10(np.abs(fc))

    dev = delta_rec**2 + delta_sen**2
    sort_ind = np.argsort(dev)

    x, y, lab = delta_rec[sort_ind[delta_rec.size - top_genes:]], delta_sen[sort_ind[delta_rec.size - top_genes:]], []
    for i in sort_ind[delta_rec.size - top_genes:]:
        lab.append(genes[i])


    xmin, xmax = 1.25*np.amin(delta_rec), 1.25*np.amax(delta_rec)
    if np.abs(xmin)<xmax/2:
        xmin = -xmax / 2
    if xmax<np.abs(xmin)/2:
        xmax = -xmin / 2

    ymin, ymax = 1.25 * np.amin(delta_sen), 1.25 * np.amax(delta_sen)
    if np.abs(ymin)<ymax/3:
        ymin = -ymax / 3
    if ymax<np.abs(ymin)/3:
        ymax = -ymin / 3

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    # delta_rec, delta_sen, fc = delta_rec[sort_ind[delta_rec.size - top_genes:]], delta_sen[sort_ind[delta_rec.size - top_genes:]], fc[sort_ind[delta_rec.size - top_genes:]]
    # delta_rec, delta_sen, fc = delta_rec[sort_ind], delta_sen[sort_ind], fc[sort_ind]

    cmap_lim = np.amax(np.abs(fc))
    pt = plt.scatter(delta_rec[sort_ind], delta_sen[sort_ind], c=fc[sort_ind], s=100, cmap='coolwarm', vmin=-cmap_lim, vmax=cmap_lim, edgecolors='k', linewidths=0.5) # remove edgecolors or add it only for top genes?
    plt.colorbar(pt, label='Expression Fold-change')

    for i in sort_ind[delta_rec.size - top_genes:]:
        ax.annotate(genes[i], (delta_rec[i] + x_shift, delta_sen[i] + y_shift))

    # plt.scatter(x, y, c='r')

    # for i in range(x.size):
    #     ax.annotate(lab[i], (x[i]+x_shift, y[i]+y_shift))

    # repel_labels(ax, delta_rec, delta_sen, lab, k=0.0025)

    plt.plot([xmin, xmax], [0,0], 'k--', lw=0.5)
    plt.plot([0, 0], [ymin, ymax], 'k--', lw=0.5)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel('$\\Delta_{Incoming}$')
    plt.ylabel('$\\Delta_{Outgoing}$')
    plt.title(cluster1+' to '+cluster2)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)






def compare_standout_genes(adata, cluster_list=None, top_genes=5, criterium='centrality', panel_height=1.5, panel_length=5, ylabel='False',
                        showfig=False, savefig=True, figname='cluster_comp.pdf', format='pdf'):

    types = sorted(list(set(list(adata.obs['clusters']))))
    colors = list(plt.cm.Set3.colors)[0:len(types)]

    if cluster_list==None:
        cluster_list = types

    # set colors for plotting
    sel_colors = []
    for i in range(len(cluster_list)):
        sel_colors.append( colors[types.index(cluster_list[i])] )

    if criterium=='centrality':
        score = 0
        y_lab = 'Betweenness\nCentrality'
    elif criterium=='incoming':
        score = 1
        y_lab = 'Incoming\nSignaling'
    elif criterium=='outgoing':
        score = 2
        y_lab = 'Outgoing\nSignaling'
    elif criterium=='total':
        score = 3
        y_lab = 'Incoming+\nOutgoing'

    genes = list(adata.var_names)
    mean, fc = np.zeros((len(cluster_list), len(genes))), np.zeros((len(cluster_list), len(genes)))

    for i in range(len(cluster_list)):
        bt_df = adata.uns['GRN_statistics'][score][cluster_list[i]]
        mean[i] = bt_df.mean(axis=1)
    avg = np.mean(mean, axis=0)

    for i in range(len(cluster_list)):
        fc[i] = mean[i]/avg
    fc = np.nan_to_num(fc)
    fc = np.transpose(fc)

    # find 5 most differentiated genes based on betweenness
    # print( np.unravel_index(fc.argmax(), fc.shape) )
    j = 0
    fig = plt.figure(figsize=(panel_length,top_genes*panel_height))
    ax_list = [511, 512, 513, 514, 515]

    while j<top_genes:
        [a,b] = np.unravel_index(fc.argmax(), fc.shape)


        btn = []
        for i in range(len(cluster_list)):
            bt_df = adata.uns['GRN_statistics'][score][cluster_list[i]]
            btn.append(np.asarray(bt_df.loc[genes[a]]))

        ax = plt.subplot2grid((top_genes, 1), (j, 0), rowspan=1, colspan=1)
        bpt = plt.boxplot(btn, patch_artist=True)
        for patch, color in zip(bpt['boxes'], sel_colors):
            patch.set_facecolor(color)
        if ylabel:
            plt.ylabel(y_lab)

        plt.xticks([])
        plt.title(genes[a])

        j = j + 1
        fc = np.delete(fc, a, axis=0)
        genes.remove( genes[a] )

    plt.xticks(np.arange(1, len(cluster_list) + 1, 1), cluster_list)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)


##################################################
#
### signaling role variation across cell states
#
##################################################

def set_plot_name(method, measure):
    if measure=='centrality':
        measure_name = 'Betweenness Centrality'
    elif measure=='incoming':
        measure_name = 'Incoming Signaling'
    elif measure=='outgoing':
        measure_name = 'Outgoing signaling'
    else:
        measure_name = 'Total Signaling'

    if method=='SD':
        method_name = 'Standard Deviation'
    elif method=='range':
        method_name = 'Range'
    elif method=='inter_range':
        method_name='Interquartile Range'

    return measure_name, method_name


def gene_variation(adata, n_genes='all', method='SD', measure='centrality', bar_color='paleturquoise', alpha=1, edge_color='mediumturquoise', edge_width=1,
                   gene_label_rot=90, gene_label_font=10, fontsize=10,
                   showfig=False, savefig=True, figsize=(6, 3), figname='gene_variation.pdf.pdf', format='pdf'):

    assert 'GRN_statistics' in adata.uns.keys(), "Please run 'grn_statistics' before calling gene_variation()"
    assert measure in ['centrality', 'incoming', 'outgoing', 'signaling'], "Please choose method from the list ['centrality', 'incoming', 'outgoing', 'signaling']"

    measure_name, method_name = set_plot_name(method, measure)

    genes = list(adata.var_names)
    y = adata.uns['cluster_variation'][measure][method]
    ind = np.flip(np.argsort(y))

    if n_genes=='all':
        pass
    else:
        ind = ind[0:n_genes]

    plt.figure(figsize=figsize)

    top = 15

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharey=ax1)

    bars = np.asarray(y[ind])

    # ax1.bar(np.arange(1, ind.size + 1, 1), y[ind], width=1, align='center', color=bar_color, alpha=alpha, edgecolor=edge_color, lw=edge_width,
    #         label=measure_name + ' - ' + method_name + ' across Cell Types')
    # ax2.bar(np.arange(1, ind.size + 1, 1), y[ind], width=1, align='center', color=bar_color, alpha=alpha, edgecolor=edge_color, lw=edge_width,
    #         label=measure_name + ' - ' + method_name + ' across Cell Types')
    ax1.bar(np.arange(1, top+1, 1), bars[0:top], width=1, align='center', color=bar_color, alpha=alpha, edgecolor=edge_color, lw=edge_width)
    ax2.bar(np.arange(bars.size-top+1, bars.size+1, 1), bars[bars.size-top:], width=1, align='center', color=bar_color, alpha=alpha, edgecolor=edge_color, lw=edge_width,
            label=measure_name + '\n' + method_name + '\nacross Cell Types')

    ax1.set_xticks(np.arange(1, top + 1, 1))
    ax2.set_xticks(np.arange(bars.size-top+1, bars.size+1, 1))
    ax1.set_xticklabels([ genes[i] for i in ind[0:top] ], rotation=gene_label_rot, fontsize=gene_label_font)
    ax2.set_xticklabels([ genes[i] for i in ind[bars.size-top:] ], rotation=gene_label_rot, fontsize=gene_label_font)
    ax1.set_xlim(0, top+1)
    ax2.set_xlim(bars.size-top, bars.size+1)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False, labelleft=False)

    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs, lw=1)  # top-left diagonal
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs, lw=1)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (-d, +d), **kwargs, lw=1)  # bottom-left diagonal
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs, lw=1)

    # plt.xticks(np.arange(1, ind.size + 1, 1), [genes[i] for i in ind], rotation=gene_label_rot, fontsize=gene_label_font)
    # plt.xlim([0, ind.size + 1])
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)


def gene_var_detail(adata, n_genes=5, select='top', method='SD', measure='centrality', loc='best', fontsize=10, legend=True, legend_font=10, gene_label_rot=45,
                    showfig=False, savefig=True, figname='gene_var_detail.pdf.pdf', format='pdf'):

    assert 'GRN_statistics' in adata.uns.keys(), "Please run 'grn_statistics' before calling gene_variation()"
    assert measure in ['centrality', 'incoming', 'outgoing','signaling'], "Please choose method from the list ['centrality', 'incoming', 'outgoing', 'signaling']"

    genes = list(adata.var_names)
    types = sorted(list(set(list(adata.obs['clusters']))))
    colors = list(plt.cm.Set2.colors)[0:len(types)]

    measure_name, method_name = set_plot_name(method, measure)

    y = adata.uns['cluster_variation'][measure][method]
    ind = np.asarray(np.flip(np.argsort(y)))

    data = np.zeros((len(types), len(genes)))
    for i in range(len(types)):
        data[i] = np.asarray(adata.uns['GRN_statistics'][3][types[i]].mean(axis=1))


    x = np.arange(1, n_genes+1, 1)

    x_add = np.linspace(-0.25, 0.25, num=len(types))
    dx = x_add[1]-x_add[0]


    plt.figure(figsize=(3, 2))

    for i in range(len(types)):
        if select=='top':
            y = data[i, ind[0:n_genes]]
            label_names=[genes[i] for i in ind[0:n_genes]]
        elif select=='bottom':
            y = data[i, ind[ind.size-n_genes:]]
            label_names = [genes[i] for i in ind[ind.size-n_genes:]]
        plt.bar(x + x_add[i], y, color=colors[i], align='center', width=dx, label=types[i])

    plt.xticks(np.arange(1, x.size + 1, 1), label_names, fontsize=fontsize, rotation=gene_label_rot)
    plt.ylabel(measure_name, fontsize=fontsize)
    if legend:
        plt.legend(loc=loc, fontsize=legend_font)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)


def gene_var_scatter(adata, method='SD', measure='centrality', top_genes=5, fontsize=10, annotate_font=7,
                   showfig=False, savefig=True, figname='gene_var_scatter.pdf.pdf', format='pdf'):

    genes = list(adata.var_names)
    types = sorted(list(set(list(adata.obs['clusters']))))
    colors = list(plt.cm.Set3.colors)[0:len(types)]

    measure_name, method_name = set_plot_name(method, measure)

    x, y = adata.uns['cluster_average'][measure], np.asarray(adata.uns['cluster_variation'][measure][method])

    sort_ind = np.argsort(y)
    x_top, y_top, lab = x[sort_ind[x.size - top_genes:]], y[sort_ind[x.size - top_genes:]], []

    for i in sort_ind[x.size - top_genes:]:
        lab.append(genes[i])

    plt.figure(figsize=(3, 2))
    ax = plt.subplot(111)

    plt.scatter(x, y)
    for i in range(x_top.size):
        ax.annotate(lab[i], (x_top[i], y_top[i]), fontsize=annotate_font)
    plt.xlabel('Average across\nStates', fontsize=fontsize)
    plt.ylabel(method_name + ' across States', fontsize=fontsize)
    plt.title(measure_name)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)







