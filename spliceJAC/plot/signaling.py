'''
functions to plot signaling strength and changes
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_signaling_hubs(adata,
                        cluster,
                        fontsize=10,
                        top_genes=5,
                        show_top_genes=True,
                        criterium='weights',
                        showfig=False,
                        savefig=True,
                        figname='signaling_hub.pdf',
                        format='pdf',
                        figsize=(3.5,3)
                        ):
    '''
    Scatterplot of genes based on their signaling scores in a cell state

    Parameters
    ----------
    adata: anndata object
    cluster: cell state
    fontsize: fontsize of figure (default=10)
    top_genes: number of top genes to label with gene name (default=5)
    show_top_genes: if True, annotate the top genes (default=True)
    criterium: criterium to rank top genes. "weights" ranks genes based on the weighted edges of the cell state GRN, "edges" ranks genes based on the number of edges (default='weights')
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='signaling_hub.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(3.5,3))

    Returns
    -------
    None

    '''

    assert criterium=='edges' or criterium=='weights', "Please choose between criterium=='edges' or criterium=='weights'"

    if criterium=='weights':
        rec, sen = adata.uns['signaling_scores'][cluster]['incoming'], adata.uns['signaling_scores'][cluster]['outgoing']
    elif criterium=='edges':
        rec, sen = adata.uns['signaling_scores'][cluster]['incoming_edges'], adata.uns['signaling_scores'][cluster]['outgoing_edges']
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
    pt = plt.scatter(rec, sen, c=avgS, cmap='Reds', vmin=0, vmax=cmap_lim, edgecolors='k', linewidths=0.5)
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
        plt.savefig(figname, format=format, dpi=300)



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
                          showfig=False,
                          savefig=True,
                          figname='signaling_hub_change.pdf',
                          format='pdf',
                          figsize=(3.5,3)
                          ):
    '''
    Scatterplot of the gene signaling changes between two cell states

    Parameters
    ----------
    adata: anndata object
    cluster1: first cell state
    cluster2: second cell state
    fontsize: fontsize of figure (default=10)
    top_genes: number of top genes to label with gene name (default=5)
    show_top_genes: if True, annotate the top genes (default=True)
    criterium: criterium to rank top genes. "weights" ranks genes based on the weighted edges of the cell state GRN, "edges" ranks genes based on the number of edges (default='weights')
    logscale_fc: if True, rescale signaling change scores to logarithmic scale (default=True)
    x_shift: displacement on x-axis for gene annotations
    y_shift: displacement on y-axis for gene annotations
    showfig: if True, show the figure (default=True)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='signaling_hub_change.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(3.5,3))

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
    pt = plt.scatter(delta_rec[sort_ind], delta_sen[sort_ind], c=fc[sort_ind], s=100, cmap='coolwarm', vmin=-cmap_lim, vmax=cmap_lim, edgecolors='k', linewidths=0.5)
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
        plt.savefig(figname, format=format, dpi=300)