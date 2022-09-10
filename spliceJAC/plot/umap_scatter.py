'''
functions for umap scatter plot visualization
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from . import plotting_util

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
                 showfig=None,
                 savefig=None,
                 format='pdf'):
    '''2D UMAP plot of the data

    Parameters
    ----------
    adata: anndata object of gene counts
    ax: pyplot axis, if False generate a new figure (default=False)
    order: order of cluster labels in the figure legend, if None the order is random (default=None)
    axis: if true, draw axes, otherwise do not show axes (default=False)
    fontsize: fontsize of axes and legend labels (default=10)
    alpha: shading of individual cells (default=0.5)
    show_cluster_center: if True, plot the center of each cluster (default=True)
    s: size of individual cells (default=2)
    legens_pos: position of figure legend by axis coordinates (default=(0.5, 1.2))
    legend_loc: position of figure legend (default='upper center')
    ncol: number of columns in the figure legend (default=4)
    figsize: size of figure (default=(4,4))
    showfig: if True, show the figure (default=None)
    savefig: if True, save the figure using the savefig path (default=None)
    format: figure format (default='pdf')

    Returns
    -------
    None

    '''
    if 'colors' not in adata.uns.keys():
        plotting_util.plot_setup(adata)

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
            plt.savefig(savefig, format=format, dpi=300)