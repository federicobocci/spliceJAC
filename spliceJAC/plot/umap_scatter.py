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
                 s_center=50,
                 line_width=0.5,
                 legens_pos=(0.5, 1.2),
                 legend_loc='upper center',
                 ncol=4,
                 figsize=(4, 4),
                 showfig=None,
                 savefig=None,
                 format='pdf'):
    '''2D UMAP plot of the data

    Parameters of matplotlib.pyplot.scatter are explained at:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    ax: `pyplot axis` or `False` (default: False)
        if False generate a new figure
    order: `list` (default: `None`)
        ordered list of cluster labels in the figure legend, if `None` the order is alphabetical
    axis: `Bool` (default: False)
        if true draw axes, otherwise do not show axes
    fontsize: `int` (default: 10)
        fontsize of axes and legend labels.
    alpha: `float` (default=0.5)
        shading of individual cells between [0,1]
    show_cluster_center: `Bool` (default: True)
        if True, plot the center of each cluster
    s: `int` (default=2)
        size of individual cells
    s_center: `int` (default=50)
        size of cluster center
    line_width: `float` (default=0.5)
        line width for cluster centers
    legens_pos: `tuple` (default: (0.5, 1.2))
        position of figure legend by axis coordinates. Details on legend location can be found at:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    legend_loc: `str` (default='upper center')
        position of figure legend
    ncol: `int` (default: 4)
        number of columns in the figure legend
    figsize: `tuple` (default: (4,4))
        size of figure
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
        plt.scatter(x_umap, y_umap, c=list(adata.uns['colors'].values()), s=s_center, edgecolors='k', linewidths=line_width)
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