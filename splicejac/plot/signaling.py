'''
functions to plot signaling strength and changes
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_signaling_hubs(adata,
                        cluster,
                        fontsize=10,
                        top_genes=5,
                        line_width=0.5,
                        show_top_genes=True,
                        criterium='weights',
                        cmap='Reds',
                        showfig=None,
                        savefig=None,
                        format='pdf',
                        figsize=(3.5,3)
                        ):
    '''Scatterplot of genes based on their signaling scores in a cell state

    Parameters of matplotlib.pyplot.scatter are explained at:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster: `str`
        cell state
    fontsize: `int` (default=10)
        fontsize of figure
    top_genes: `int` (default=5)
        number of top genes to label with gene name
    line_width: `float` (default=0.5)
        line width for scatter plot
    show_top_genes: `Bool` (default=True)
        if True, annotate the top genes
    criterium: `str` (default='weights')
        criterium to rank top genes. "weights" ranks genes based on the weighted edges of the cell state GRN,
        "edges" ranks genes based on the number of edges
    cmap: `str` (default: 'Reds')
        the pyplot colormap for the scatter plot. A list of accepted color maps can be found at:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
        if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        format of saved figure
    figsize: `tuple` (default:(3.5,3))
        size of figure

    Returns
    -------
    None

    '''
    assert criterium=='edges' or criterium=='weights', "Choose between criterium=='edges' or criterium=='weights'"

    if criterium=='weights':
        rec = adata.uns['signaling_scores'][cluster]['incoming']
        sen = adata.uns['signaling_scores'][cluster]['outgoing']
    elif criterium=='edges':
        rec = adata.uns['signaling_scores'][cluster]['incoming_edges']
        sen = adata.uns['signaling_scores'][cluster]['outgoing_edges']
    genes = list(adata.var_names)

    # gene expression for colormap
    S = adata[adata.obs['clusters'] == cluster].layers['spliced'].toarray()
    avgS = np.mean(S, axis=0)
    cmap_lim = np.amax(avgS)

    tot = rec+sen
    sort_ind = np.argsort(tot)
    x, y, lab = rec[sort_ind[rec.size - top_genes:]], sen[sort_ind[rec.size - top_genes:]], []

    for i in sort_ind[rec.size - top_genes:]:
        lab.append(genes[i])

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    pt = plt.scatter(rec, sen, c=avgS, cmap=cmap, vmin=0, vmax=cmap_lim, edgecolors='k', linewidths=line_width)
    plt.colorbar(pt, label='Expression level')

    if show_top_genes:
        for i in range(x.size):
            ax.annotate(lab[i], (x[i], y[i]), fontsize=fontsize)

    plt.xlim([-0.1, 1.25*np.amax(rec)])
    plt.ylim([-0.1, 1.1*np.amax(sen)])
    plt.xlabel('Incoming score', fontsize=fontsize)
    plt.ylabel('Outgoing score', fontsize=fontsize)
    plt.title(cluster, fontsize=fontsize)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format, dpi=300)


def plot_signaling_change(adata,
                          cluster1,
                          cluster2,
                          fontsize=10,
                          top_genes=10,
                          show_top_genes=True,
                          criterium='weights',
                          logscale_fc=True,
                          x_shift=0.05,
                          y_shift=0.05,
                          cmap='coolwarm',
                          line_width=0.5,
                          showfig=None,
                          savefig=None,
                          format='pdf',
                          figsize=(3.5,3)
                          ):
    '''Scatterplot of the gene signaling changes between two cell states

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster1: `str`
        first cell state
    cluster2: `str`
        second cell state
    fontsize: `int` (default=10)
        fontsize of figure
    top_genes: `int` (default=10)
        number of top genes to label with gene name
    show_top_genes: `Bool` (default=True)
        if True, annotate the top genes
    criterium: `str` (default='weights')
        criterium to rank top genes. "weights" ranks genes based on the weighted edges of the cell state GRN,
        "edges" ranks genes based on the number of edges
    logscale_fc: `Bool` (default=True)
        if True, rescale signaling change scores to logarithmic scale
    x_shift: `float` (default: 0.05)
        displacement on x-axis for gene annotations
    y_shift: `float` (default: 0.05)
        displacement on y-axis for gene annotations
    cmap: `str` (default: 'coolwarm')
        the pyplot colormap for the scatter plot. A list of accepted color maps can be found at:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    line_width: `float` (default=0.5)
        line width for scatter plot
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
        if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        format of saved figure
    figsize: `tuple` (default:(3.5,3))
        size of figure

    Returns
    -------
    None

    '''
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


    # compute gene expression fold-change between cluster 1 to cluster 2
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


    # set axes limits for best visualization
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

    cmap_lim = np.amax(np.abs(fc))
    pt = plt.scatter(delta_rec[sort_ind], delta_sen[sort_ind], c=fc[sort_ind], s=100, cmap=cmap, vmin=-cmap_lim, vmax=cmap_lim, edgecolors='k', linewidths=line_width)
    plt.colorbar(pt, label='Expression Fold-change')

    if show_top_genes:
        for i in sort_ind[delta_rec.size - top_genes:]:
            ax.annotate(genes[i], (delta_rec[i] + x_shift, delta_sen[i] + y_shift), fontsize=fontsize)

    plt.plot([xmin, xmax], [0,0], 'k--', lw=0.5)
    plt.plot([0, 0], [ymin, ymax], 'k--', lw=0.5)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel('$\\Delta_{Incoming}$', fontsize=fontsize)
    plt.ylabel('$\\Delta_{Outgoing}$', fontsize=fontsize)
    plt.title(cluster1+' to '+cluster2, fontsize=fontsize)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format)