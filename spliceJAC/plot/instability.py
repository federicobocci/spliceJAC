'''
plotting functions for instability scores
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_trans_genes(adata,
                     cluster1,
                     cluster2,
                     top_trans_genes=10,
                     fontsize=10,
                     color='r',
                     alpha=0.5,
                     showfig=None,
                     savefig=None,
                     format='pdf',
                     figsize=(3,3)
                     ):
    '''Plot the top transition genes between two cell states

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster1: `str`
        starting cell state
    cluster2: `str`
        final cell state
    top_trans_genes: `int` (default: 10)
        top genes to include in the bar plot
    fontsize: `int` (default: 10)
        fontsize for labels
    color: `str` (default: 'r')
        color for bar plot. A list of named pyplot colors can be found at:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    alpha: `float` (default=0.5)
        shading for bar plot in [0,1]
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format
    figsize: `tuple` (default: (3,3))
        size of figure

    Returns
    -------
    None

    '''
    weight = adata.uns['transitions'][cluster1 + '-' + cluster2]['weights']
    genes = list(adata.var_names)

    ind = np.argsort(weight)

    data, trans_genes = [], []
    for i in range(top_trans_genes):
        trans_genes.append( genes[ind[weight.size - top_trans_genes + i]] )
        data.append( weight[ind[weight.size - top_trans_genes + i]]  )

    fig = plt.figure(figsize=figsize)
    plt.barh(np.arange(1, len(data) + 1, 1), data, height=0.8, align='center', color=color, edgecolor='k', linewidth=1, alpha=alpha)
    plt.yticks(np.arange(1, len(data) + 1, 1), trans_genes, fontsize=fontsize)
    plt.xlabel('Instability score', fontsize=fontsize)
    plt.title(cluster1 + ' to ' + cluster2, fontsize=fontsize)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format, dpi=300)


def scatter_scores(adata,
                   cluster1,
                   cluster2,
                   fontsize=10,
                   color='b',
                   showfig=None,
                   savefig=None,
                   format='pdf',
                   figsize=(3,3)
                   ):
    '''Scatter plot to compare the spliceJAC transition scores with scanpy's DEG scores of the starting cell state

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster1: `str`
        starting cell state
    cluster2: `str`
        final cell state
    fontsize: `int` (default: 10)
        fontsize for labels
    color: `str` (default: 'b')
        color for scatter plot. A list of named pyplot colors can be found at:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format
    figsize: `tuple` (default: (3,3))
        size of figure

    Returns
    -------
    None

    '''
    # get gene instability scores along transition direction
    weight = adata.uns['transitions'][cluster1 + '-' + cluster2]['weights']

    genes = list(adata.var_names)
    deg_genes = adata.uns['rank_genes_groups']['names'][cluster1]
    scores = adata.uns['rank_genes_groups']['scores'][cluster1]
    ord_weight = []
    for d in deg_genes:
        i = genes.index(d)
        ord_weight.append(weight[i])

    plt.figure(figsize=figsize)
    plt.scatter(scores, ord_weight, c=color)
    plt.xlabel('DEG score', fontsize=fontsize)
    plt.ylabel('Instability score', fontsize=fontsize)
    plt.title(cluster1 + ' to ' + cluster2, fontsize=fontsize)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format, dpi=300)


def compare_scvelo_scores(adata,
                          annotate=True,
                          top=5,
                          color='b',
                          panel_height=3,
                          panel_length=3.5,
                          pan_per_row=4,
                          fontsize=10,
                          showfig=None,
                          savefig=None,
                          format='pdf'
                          ):
    '''Scatter plots to compare spliceJAC's transition scores with scVelo's gene likelihood scores

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    annotate: `Bool` (default=True)
        if True, annotate the genes with highest sum of scores
    top: `int` (default=5)
        number of top genes to annotate
    color: `str` (default: 'b')
        color for scatter plot. A list of named pyplot colors can be found at:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    panel_height: `float` (default: 3)
        height of each panel (in inches)
    panel_length: `float` (default: 3.5)
        length of each panel (in inches)
    pan_per_row: `int` (default: 4)
        number of panels per row
    fontsize: `int` (default: 10)
        fontsize for labels
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format

    Returns
    -------
    None

    '''
    types = sorted(list(set(list(adata.obs['clusters']))))
    genes = list(adata.var_names)

    # create dictionary of scvelo scores re-ordered based on gene list (which is the order followed by spliceJAC's transition scores)
    scvelo_genes = {}
    for t1 in types:
        for t2 in types:
            if t1 + '-' + t2 in adata.uns['transitions'].keys():
                gene_rank, rank = list(adata.uns['rank_dynamical_genes']['names'][t1]), adata.uns['rank_dynamical_genes']['scores'][t1]

                # create array of rank_scores for all 50 genes
                complete_rank_score = np.zeros(len(genes))
                for i in range(complete_rank_score.size):
                    if genes[i] in gene_rank:
                        complete_rank_score[i] = rank[ gene_rank.index(genes[i]) ]
                scvelo_genes[t1 + '-' + t2] = complete_rank_score


    n = len(scvelo_genes.keys())
    clusters = sorted(list(scvelo_genes.keys()))

    nrow = int(n / pan_per_row) + 1 if n % pan_per_row > 0 else int(n / pan_per_row)
    ncol = max(n % pan_per_row, pan_per_row)

    fig = plt.figure(figsize=(panel_length * ncol, panel_height * nrow))

    for i in range(n):
        # panel coordinates for plotting
        j = int(i / pan_per_row)
        k = i % pan_per_row

        ax = plt.subplot2grid((nrow, ncol), (j, k), rowspan=1, colspan=1)

        inst_score = adata.uns['transitions'][clusters[i]]['weights']

        plt.scatter( inst_score, scvelo_genes[clusters[i]], color=color )
        plt.title(clusters[i])
        plt.xlabel('Instability score (spliceJAC)', fontsize=fontsize)
        plt.ylabel('Likelihood (scVelo)', fontsize=fontsize)

        if annotate:
            tot = inst_score + scvelo_genes[clusters[i]]
            sort_ind = np.argsort(tot)
            x, y, lab = inst_score[sort_ind[inst_score.size - top:]], scvelo_genes[clusters[i]][sort_ind[scvelo_genes[clusters[i]].size - top:]], []

            for i in sort_ind[inst_score.size - top:]:
                lab.append(genes[i])

            for i in range(x.size):
                ax.annotate(lab[i], (x[i], y[i]))


    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format, dpi=300)