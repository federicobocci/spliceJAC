'''
plotting resources for GRN visualization
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import networkx as nx
from pyvis.network import Network


def plot_grn(G,
             node_size=200,
             edge_width=1,
             font_size=8,
             adata=None,
             node_color='centrality',
             cluster_name=None,
             pos_style='spring',
             base_node_size=300,
             diff_node_size=600,
             pos_edge_color='b',
             neg_edge_color='r',
             arrowsize=10,
             arrow_alpha=0.75,
             conn_style='straight',
             colorbar=True,
             fontweight='normal'
             ):
    '''Plot the GRN with positive and negative interactions

    Parameters
    ----------
    G: networkx graph object
    node_size: size of nodes. If node_size='expression', the node size is scaled based on gene expression. Otherwise, node_size can be an integer (default=200)
    edge_width: width of connection arrows
    font_size: font size for node labels
    adata: gene count anndata object, must be provided if node_size == 'expression'
    node_color: color of nodes. If node_color='centrality', the color is based on the node centrality. Otherwise, a matplotlib color name can be provided (default='centrality')
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
    fontweight: style of text (default='normal')

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
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cbar_label = 'Betweenness Centrality'
        cmap = plt.cm.viridis
    elif isinstance(node_color, list):
        norm = mpl.colors.Normalize(vmin=min(node_color), vmax=max(node_color))
        cbar_label = 'Expression change'
        cmap = plt.cm.BrBG

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.5, cmap=cmap)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if colorbar:
        cbar = plt.colorbar(sm, label=cbar_label)

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
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_family="sans-serif", font_weight=fontweight)


def visualize_network(adata,
                      cluster_name,
                      genes=None,
                      cc_id=0,
                      node_size='expression',
                      edge_width='weight',
                      font_size=10,
                      plot_interactive=True,
                      weight_quantile=.5,
                      node_color='centrality',
                      pos_style='spring',
                      title=True,
                      base_node_size=300,
                      diff_node_size=600,
                      pos_edge_color='b',
                      neg_edge_color='r',
                      arrowsize=10,
                      arrow_alpha=0.75,
                      conn_style='straight',
                      colorbar=True,
                      fontweight='normal',
                      showfig=False,
                      savefig=True,
                      figname='core_GRN.pdf',
                      format='pdf',
                      figsize=(4, 3)
                      ):
    '''
    Plot the gene regulatory network of a cluster

    Parameters
    ----------
    adata: anndata object
    cluster_name: cell state
    genes: list of genes to include (default=None). If None, all genes are included
    cc_id: connected component of the GRN to plot (default=0)
    node_size: size of nodes in the GRN (default='expression'). If node_size='expression', the node size is proportional to the gene expression in the cell state. Otherwise, a number can be specified
    edge_width: width of GRN edges (default='weight'). If edge_width='weight', the edge width is proportional to the interaction strength
    font_size: font size of the figure (default=10)
    plot_interactive: plot the GRN interactively (default=True)
    weight_quantile: threshold to filter weak interactions between 0 and 1 (default=0.5)
    node_color: color of nodes. If node_color='centrality', the color is based on the node centrality. Otherwise, a matplotlib color name can be provided (default='centrality')
    pos_style: position of nodes. The options are 'spring' and 'circle'. 'spring' uses the networkx spring position function, whereas 'circle' arranges the nodes in a circle
    title: if True, plot title (default=True)
    base_node_size: Minimum node size (used if node_size='expression')
    diff_node_size: difference betwene minimum and maximal node size (used if node_size='expression')
    pos_edge_color: color for positive regulation arrow
    neg_edge_color: color for negative regulation arrow
    arrowsize: size of interaction arrows
    arrow_alpha: shading of interaction arrows
    conn_style: style of interaction arrows
    colorbar: if True, show colorbar (required if node_size='expression')
    fontweight: style of text (default='normal')
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='core_GRN.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(5,4))

    Returns
    -------
    None

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
    plot_grn(subgraphs[cc_id], node_size=node_size, edge_width=edge_width, font_size=font_size, adata=adata, node_color=node_color, cluster_name=cluster_name, pos_style=pos_style,
             base_node_size=base_node_size, diff_node_size=diff_node_size, pos_edge_color=pos_edge_color, neg_edge_color=neg_edge_color,
             arrowsize=arrowsize, arrow_alpha=arrow_alpha, conn_style=conn_style, colorbar=colorbar, fontweight=fontweight)

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



def diff_network(adata,
                 cluster1,
                 cluster2,
                 genes=None,
                 cc_id=0,
                 node_size=500,
                 edge_width='weight',
                 font_size=10,
                 weight_quantile=.5,
                 pos_style='spring',
                 base_node_size=300,
                 diff_node_size=600,
                 pos_edge_color='b',
                 neg_edge_color='r',
                 arrowsize=10,
                 arrow_alpha=0.75,
                 conn_style='straight',
                 colorbar=True,
                 fontweight='normal',
                 title=True,
                 showfig=False,
                 savefig=True,
                 figname='diff_grn.pdf',
                 format='pdf',
                 figsize=(3.5, 3)
                 ):
    '''
    Plot the differential network between two cell states

    Parameters
    ----------
    adata: anndata object
    cluster1: first cell state
    cluster2: second cell state
    genes: list of genes to consider. If None, all genes are considered (default=None)
    cc_id: connected component of the GRN to plot (default=0)
    node_size: size of nodes in the GRN (default=500)
    edge_width: width of GRN edges (default='weight'). If edge_width='weight', the edge width is proportional to the interaction strength
    font_size: font size of the figure (default=10)
    weight_quantile: threshold to filter weak interactions between 0 and 1 (default=0.5)
    pos_style: position of nodes. The options are 'spring' and 'circle'. 'spring' uses the networkx spring position function, whereas 'circle' arranges the nodes in a circle
    base_node_size: Minimum node size (used if node_size='expression')
    diff_node_size: difference betwene minimum and maximal node size (used if node_size='expression')
    pos_edge_color: color for positive regulation arrow
    neg_edge_color: color for negative regulation arrow
    arrowsize: size of interaction arrows
    arrow_alpha: shading of interaction arrows
    conn_style: style of interaction arrows
    colorbar: if True, show colorbar (required if node_size='expression')
    fontweight: style of text (default='normal')
    title: if True, plot title (default=True)
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='diff_grn.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(5,4))

    Returns
    -------
    None

    '''
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

    data1 = adata[adata.obs['clusters'] == cluster1, list(subgraphs[cc_id])].X.toarray()
    data2 = adata[adata.obs['clusters'] == cluster2, list(subgraphs[cc_id])].X.toarray()
    expr_change = list(np.mean(data2, axis=0) - np.mean(data1, axis=0))

    plot_grn(subgraphs[cc_id], node_size=node_size, edge_width=edge_width, font_size=font_size, adata=adata,
             node_color=expr_change, cluster_name=None, pos_style=pos_style,
             base_node_size=base_node_size, diff_node_size=diff_node_size, pos_edge_color=pos_edge_color,
             neg_edge_color=neg_edge_color,
             arrowsize=arrowsize, arrow_alpha=arrow_alpha, conn_style=conn_style, colorbar=colorbar, fontweight=fontweight)
    plt.axis("off")

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    if title:
        plt.title(cluster1 + '-' + cluster2 + ' differential GRN')

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)


def diff_interactions(adata,
                      cluster1,
                      cluster2,
                      top_int=10,
                      loc='best',
                      title=False,
                      fontsize=10,
                      legend_font=10,
                      legend_col=1,
                      showfig=False,
                      savefig=True,
                      figname='diff_interactions.pdf',
                      format='pdf',
                      figsize=(4,5)
                      ):
    '''
    Plot the top differential interactions between two cell states

    Parameters
    ----------
    adata: anndata object
    cluster1: first cell state
    cluster2: second cell state
    top_int: number of top chnaged interactions to plot (default=10)
    loc: location of legend (default='best')
    title: if True, plot title (default=False)
    fontsize: fontsize of figure (default=10)
    legend_font: fontsize of legend (default=10)
    legend_col: number of columns in legend (default=1)
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='diff_interactions.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(5,4))

    Returns
    -------
    None

    '''
    genes = list(adata.var_names)
    n = len(genes)

    A1 = adata.uns['average_jac'][cluster1][0][0:n, n:].copy().T
    A2 = adata.uns['average_jac'][cluster2][0][0:n, n:].copy().T
    A = A2 - A1

    y = np.ndarray.flatten(A)
    sorted = np.flip(np.argsort( np.abs(y) ))

    int_list, name = [], []
    old, new = [], []
    color = []
    for i in range(top_int):
        k, l = np.unravel_index(sorted[i], shape=(n,n))
        int_list.append(A[k][l])
        name.append( genes[k] + ' to ' + genes[l] )
        old.append(A1[k][l])
        new.append(A2[k][l])

    fig = plt.figure(figsize=figsize)

    plt.plot( old, np.arange(1, len(int_list)+1, 1), 'bo', label=cluster1 )
    plt.plot( new, np.arange(1, len(int_list) + 1, 1), 'ro', label=cluster2)
    plt.plot( np.zeros(len(int_list)), np.linspace(0, len(int_list) + 1, len(int_list)), 'k--' )

    for i in range(len(old)):
        plt.arrow( old[i], i+1, 0.95*(new[i]-old[i]), 0, head_length=0.05, head_width=0.5, length_includes_head=True, color='grey')

    plt.yticks( np.arange(1, len(int_list)+1, 1), name )
    plt.xlabel('Interaction strenght change', fontsize=fontsize)
    plt.legend(loc=loc, fontsize=legend_font, ncol=legend_col)
    plt.ylim([0, len(int_list) + 2])

    if title:
        plt.title(title, fontsize=fontsize)
    else:
        plt.title(cluster1 + '-' + cluster2 + ' differential interactions', fontsize=fontsize)
    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format)


def conserved_grn(adata,
                  cluster1,
                  cluster2,
                  genes=None,
                  cc_id=0,
                  node_size=500,
                  edge_width='weight',
                  font_size=10,
                  weight_quantile=.5,
                  pos_style='spring',
                  title=True,
                  base_node_size=300,
                  diff_node_size=600,
                  pos_edge_color='b',
                  neg_edge_color='r',
                  arrowsize=10,
                  arrow_alpha=0.75,
                  conn_style='straight',
                  colorbar=True,
                  fontweight='normal',
                  showfig=False,
                  savefig=True,
                  figname='cons_grn.pdf',
                  format='pdf',
                  figsize=(3.5, 3)
                  ):
    '''
    Plot the conserved network between two cell states

    Parameters
    ----------
    adata: anndata object
    cluster1: first cell state
    cluster2: second cell state
    genes: list of genes to consider. If None, all genes are considered (default=None)
    cc_id: connected component of the GRN to plot (default=0)
    node_size: size of nodes in the GRN (default=500)
    edge_width: width of GRN edges (default='weight'). If edge_width='weight', the edge width is proportional to the interaction strength
    font_size: font size of the figure (default=10)
    weight_quantile: threshold to filter weak interactions between 0 and 1 (default=0.5)
    pos_style: pos_style: position of nodes. The options are 'spring' and 'circle'. 'spring' uses the networkx spring position function, whereas 'circle' arranges the nodes in a circle
    title: if True, plot title (default=True)
    base_node_size: Minimum node size (used if node_size='expression')
    diff_node_size: difference betwene minimum and maximal node size (used if node_size='expression')
    pos_edge_color: color for positive regulation arrow
    neg_edge_color: color for negative regulation arrow
    arrowsize: size of interaction arrows
    arrow_alpha: shading of interaction arrows
    conn_style: style of interaction arrows
    colorbar: if True, show colorbar (required if node_size='expression')
    fontweight: style of text (default='normal')
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='cons_grn.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(5,4))

    Returns
    -------
    None

    '''
    if genes == None:
        genes = list(adata.var_names)
    n = len(genes)

    A1 = adata.uns['average_jac'][cluster1][0][0:n, n:].copy().T
    A2 = adata.uns['average_jac'][cluster2][0][0:n, n:].copy().T

    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if np.sign(A2[i][j]) != np.sign(A1[i][j]):
                A[i][j] = 0.
            else:
                A[i][j] = np.mean([A2[i][j], A1[i][j]])

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

    data1 = adata[adata.obs['clusters'] == cluster1, list(subgraphs[cc_id])].X.toarray()
    data2 = adata[adata.obs['clusters'] == cluster2, list(subgraphs[cc_id])].X.toarray()
    expr_change = list(np.mean(data2, axis=0) - np.mean(data1, axis=0))

    plot_grn(subgraphs[cc_id], node_size=node_size, edge_width=edge_width, font_size=font_size, adata=adata,
             node_color=expr_change, cluster_name=None, pos_style=pos_style, base_node_size=base_node_size,
             diff_node_size=diff_node_size,
             pos_edge_color=pos_edge_color, neg_edge_color=neg_edge_color, arrowsize=arrowsize, arrow_alpha=arrow_alpha,
             conn_style=conn_style, colorbar=colorbar, fontweight=fontweight)
    plt.axis("off")

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    if title:
        plt.title(cluster1 + '-' + cluster2 + ' conserved GRN')

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)



def top_conserved_int(adata,
                      cluster1,
                      cluster2,
                      top_int=10,
                      title=False,
                      fontsize=10,
                      alpha=0.5,
                      showfig=False,
                      savefig=True,
                      figname='conserved_interactions.pdf',
                      format='pdf',
                      figsize=(4,5)
                      ):
    '''
    Plot the top conserved interactions between two cell states

    Parameters
    ----------
    adata: anndata object
    cluster1: first cell state
    cluster2: second cell state
    top_int: number of top chnaged interactions to plot (default=10)
    title: if True, plot title (default=False)
    fontsize: fontsize of figure (default=10)
    alpha: shading of bar plot (default=0.5)
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='conserved_interactions.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(5,4))

    Returns
    -------
    None

    '''
    genes = list(adata.var_names)
    n = len(genes)

    A1 = adata.uns['average_jac'][cluster1][0][0:n, n:].copy().T
    A2 = adata.uns['average_jac'][cluster2][0][0:n, n:].copy().T

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if np.sign(A2[i][j]) != np.sign(A1[i][j]):
                A[i][j] = 0.
            else:
                A[i][j] = np.mean([A2[i][j], A1[i][j]])


    y = np.ndarray.flatten(A)
    sorted = np.flip(np.argsort(np.abs(y)))

    int_list, name = [], []
    color = []
    for i in range(top_int):
        k, l = np.unravel_index(sorted[i], shape=(n, n))
        int_list.append(A[k][l])
        name.append(genes[k] + ' to ' + genes[l])
        color.append('b') if A[k][l]>0 else color.append('r')


    plt.figure(figsize=figsize)

    plt.barh( np.arange(len(int_list), 0, -1), int_list, color=color, alpha=alpha )

    plt.yticks( np.arange(len(int_list), 0, -1), name)
    plt.xlabel('Interaction strenght', fontsize=fontsize)
    plt.ylim([0, len(int_list) + 1])

    if title:
        plt.title(title, fontsize=fontsize)
    else:
        plt.title(cluster1 + '-' + cluster2 + ' conserved interactions', fontsize=fontsize)
    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format)


def robust_mean(X):
    '''
    Calculate gene expression mean based on quantile selection

    Parameters
    ----------
    X: array of gene expression values

    Returns
    -------
    robust mean: array of mean gene expression

    '''
    rob_mean = 0.25 * np.quantile(X, q=0.75, axis=0) + 0.25 * np.quantile(X, q=0.25, axis=0) + 0.5 * np.quantile(X, q=0.5,
                                                                                                             axis=0)
    return rob_mean


def NormalizeData(data):
    '''
    normalize gene expression data

    Parameters
    ----------
    data: data array

    Returns
    -------
    scaled: normalized data

    '''
    scaled = (data - np.min(data)) / (np.max(data) - np.min(data))
    return scaled


def circle_pos(G):
    '''
    Compute coordinates along a unit circle for GRN node positions

    Parameters
    ----------
    G: networkx object

    Returns
    -------
    pos: array of positions

    '''
    nodes = list(G.nodes)
    center, radius = [0,0], 1

    pos = {}
    angle = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)

    for i in range(angle.size):
        coord = np.array([ center[0]+np.sin(angle[i]*radius), center[1]+np.cos(angle[i]*radius) ])
        pos[nodes[i]]=coord
    return pos


def subset_jacobian(J, genes, genelist):
    '''
    select gene-gene interactions for genes in the genelist

    Parameters
    ----------
    J: gene-gene interaction matrix
    genes: full lits of genes
    genelist: list of selected genes

    Returns
    -------
    B_subset: reduced gene-gene interaction matrix

    '''
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
    '''
    Plot a reduced GRN including the top DEG of the starting cluster and the top transition genes
    Parameters
    ----------
    adata: anndata object
    cluster1: starting cluster
    cluster2: final cluster
    type_color: colors for nodes that are DEG, transition genes and both (default=['orange', 'plum', 'yellowgreen'])
    pos_edge_color: color for positive regulation arrow
    neg_edge_color: color for negative regulation arrow
    node_size: size of nodes (default=500)
    node_alpha: shading of nodes (default=0.5)
    arrowsize: size of interaction arrow (default=10)
    arrow_alpha: shading of interaction arrows (default=0.75)
    conn_style: arrow connection style (default='arc3, rad=0.1')
    node_font: font size of node labels (default=8)
    legend: if True, include legend (default=True)
    legend_font: font size for legend (default=10)
    legend_ncol: number of columns in legend (default=2)
    legend_loc: legend location (default='lower center')
    axis: if True, plot axes (default=False)
    xlim: inteval on x-axis (default=[-1.2, 1.2])
    ylim: inteval on y-axis (default=None)
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='core_GRN.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(3,3))

    Returns
    -------
    None

    '''
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
    '''
    Plot the reduced GRN of transition genes involved in different cell state transitions

    Parameters
    ----------
    adata: anndata object
    start: starting cell state
    end: list of ending cell states
    pos_edge_color: color for positive regulation arrow
    neg_edge_color: color for negative regulation arrow
    node_size: size of nodes (default=500)
    node_alpha: shading of nodes (default=0.5)
    arrowsize: size of interaction arrow (default=10)
    arrow_alpha: shading of interaction arrows (default=0.75)
    conn_style: arrow connection style (default='arc3, rad=0.1')
    node_font: font size of node labels (default=8)
    legend: if True, include legend (default=True)
    legend_font: font size for legend (default=10)
    legend_ncol: number of columns in legend (default=2)
    legend_loc: legend location (default='lower center')
    axis: if True, plot axes (default=False)
    xlim: inteval on x-axis (default=[-1.2, 1.2])
    ylim: inteval on y-axis (default=None)
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='bif_GRN.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(6,6))

    Returns
    -------
    None

    '''
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



