'''
functions to plot gene variation across cell states
'''
import numpy as np
import matplotlib.pyplot as plt

def set_plot_name(method,
                  measure
                  ):
    '''Set the plot name based on method and measure used

    Parameters
    ----------
    method: `str`
        method used
    measure: `str`
        measure used

    Returns
    -------
    measure_name: `str`
        string with name of measure
    method_name: `str`
        string with name of method

    '''
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


def gene_variation(adata,
                   n_genes='all',
                   method='SD',
                   measure='centrality',
                   bar_color='paleturquoise',
                   alpha=1,
                   edge_color='mediumturquoise',
                   edge_width=1,
                   gene_label_rot=90,
                   fontsize=10,
                   showfig=None,
                   savefig=None,
                   format='pdf',
                   figsize=(6, 3)
                   ):
    '''Bar plot of gene role variation across cell states

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    n_genes: `str` or `int` (default n_genes='all')
        number of genes to consider. If an integer (n) is provided, the top n genes are selected.
        Otherwise, all genes are used if n_genes='all'
    method: `float` (default: "SD")
        method to estimate gene role variation across cell states, choose between standard deviation ('SD'),
        range ('range'), and interquartile range ('inter_range')
    measure: `float` (default: 'centrality')
        measure to estimate gene role variation, choose between 'centrality', 'incoming', 'outgoing', 'signaling'
    bar_color: `str` (default: 'paleturquoise')
        color for bar plot. A full list of accepted named colors can be found at:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    alpha: `float` (default: 1)
        shading of bar plot between 0 and 1
    edge_color: `str` (default: 'paleturquoise')
        edge color for bar plot. A full list of accepted named colors can be found at:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    edge_width: `float` (default: 1)
        edge width for bar plot
    gene_label_rot: `int` (default: 90)
        rotation of labels on x-axis
    fontsize: `int` (default: 10)
        fontsize of figure
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format
    figsize: `tuple` (default: (6,3))
        size of figure

    Returns
    -------
    None

    '''
    assert 'GRN_statistics' in adata.uns.keys(), "Please run 'grn_statistics' before calling gene_variation()"
    assert measure in ['centrality', 'incoming', 'outgoing', 'signaling'], \
        "Please choose method from the list ['centrality', 'incoming', 'outgoing', 'signaling']"

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

    ax1.bar(np.arange(1, top+1, 1), bars[0:top], width=1, align='center', color=bar_color, alpha=alpha,
            edgecolor=edge_color, lw=edge_width)
    ax2.bar(np.arange(bars.size-top+1, bars.size+1, 1), bars[bars.size-top:], width=1, align='center',
            color=bar_color, alpha=alpha, edgecolor=edge_color, lw=edge_width,
            label=measure_name + '\n' + method_name + '\nacross Cell Types')

    ax1.set_xticks(np.arange(1, top + 1, 1))
    ax2.set_xticks(np.arange(bars.size-top+1, bars.size+1, 1))
    ax1.set_xticklabels([ genes[i] for i in ind[0:top] ], rotation=gene_label_rot, fontsize=fontsize)
    ax2.set_xticklabels([ genes[i] for i in ind[bars.size-top:] ], rotation=gene_label_rot, fontsize=fontsize)
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

    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format, dpi=300)


def gene_var_detail(adata,
                    n_genes=5,
                    select='top',
                    method='SD',
                    measure='centrality',
                    loc='best',
                    fontsize=10,
                    legend=True,
                    legend_font=10,
                    gene_label_rot=45,
                    showfig=None,
                    savefig=None,
                    format='pdf',
                    figsize=(5, 4)
                    ):
    '''Plot the detailed variation in gene signaling role for the top genes in the dataset

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    n_genes: `int` (default:5)
        number of top genes
    select: `str` (default: 'top')
        choose to select genes with larger variation between cell states (select='top')
        or small variation (select='bottom')
    method: `float` (default: "SD")
        method to estimate gene role variation across cell states, choose between standard deviation ('SD'),
        range ('range'), and interquartile range ('inter_range')
    measure: `float` (default: 'centrality')
        measure to estimate gene role variation, choose between 'centrality', 'incoming', 'outgoing', 'signaling'
    legend_loc: `str` (default: 'best')
        legend location.  Details on legend location can be found at:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    fontsize: `int` (default: 10)
        fontsize of figure
    legend: `Bool` (default: True)
        if True, include legend
    legend_font: `int` (default: 10)
        font of legend
    gene_label_rot: `int` (default: 45)
        rotation of labels on x-axis
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format
    figsize: `tuple` (default: (5,4))
        size of figure

    Returns
    -------
    None

    '''
    assert 'GRN_statistics' in adata.uns.keys(), "Please run 'grn_statistics' before calling gene_variation()"
    assert measure in ['centrality', 'incoming', 'outgoing','signaling'], \
        "Please choose method from the list ['centrality', 'incoming', 'outgoing', 'signaling']"

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

    plt.figure(figsize=figsize)

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
        plt.savefig(savefig, format=format, dpi=300)


def gene_var_scatter(adata,
                     method='SD',
                     measure='centrality',
                     top_genes=5,
                     fontsize=10,
                     color='b',
                     showfig=None,
                     savefig=None,
                     format='pdf',
                     figsize=(5, 4)
                     ):
    '''Scatter plot of gene role variation across cell states

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    method: `float` (default: "SD")
        method to estimate gene role variation across cell states, choose between standard deviation ('SD'),
        range ('range'), and interquartile range ('inter_range')
    measure: `float` (default: 'centrality')
        measure to estimate gene role variation, choose between 'centrality', 'incoming', 'outgoing', 'signaling'
    top_genes: `int` (default: 5)
        top genes to annotate
    fontsize: `int` (default: 10)
        fontsize of figure
    color: `str` (default: 'b')
        color of scatter plot. A full list of accepted named colors can be found at:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format
    figsize: `tuple` (default: (5,4))
        size of figure

    Returns
    -------
    None

    '''
    genes = list(adata.var_names)

    measure_name, method_name = set_plot_name(method, measure)

    x, y = adata.uns['cluster_average'][measure], np.asarray(adata.uns['cluster_variation'][measure][method])

    sort_ind = np.argsort(y)
    x_top, y_top, lab = x[sort_ind[x.size - top_genes:]], y[sort_ind[x.size - top_genes:]], []

    for i in sort_ind[x.size - top_genes:]:
        lab.append(genes[i])

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    plt.scatter(x, y, color=color)
    for i in range(x_top.size):
        ax.annotate(lab[i], (x_top[i], y_top[i]), fontsize=fontsize)
    plt.xlabel('Average across\nStates', fontsize=fontsize)
    plt.ylabel(method_name + ' across States', fontsize=fontsize)
    plt.title(measure_name)
    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format, dpi=300)


def compare_standout_genes(adata,
                           cluster_list=None,
                           top_genes=5,
                           criterium='centrality',
                           panel_height=1.5,
                           panel_length=5,
                           ylabel=False,
                           showfig=None,
                           savefig=None,
                           format='pdf'
                           ):
    '''
    Boxplot to compare the standout genes with state-specific roles across many cell states

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster_list: `str`
        list of cell states to compare
    top_genes: `str` (default: 5)
        number of top genes to consider
    criterium: `str` (default='centrality')
        measure to use to evaluate gene role, choose between 'centrality', 'incoming', 'outgoing', 'signaling'
    panel_height: `float` (default: 1.5)
        height of each panel (in inches)
    panel_length: `float` (default: 5)
        length of each panel (in inches)
    ylabel: `Bool` (default: False)
        if True, print label of y-axis
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
    j = 0
    fig = plt.figure(figsize=(panel_length,top_genes*panel_height))

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
        plt.savefig(savefig, format=format, dpi=300)


