'''
functions to quantitatively compare GRNs across cell states
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_grn_comparison(adata,
                        score='AUPRC',
                        edges='all',
                        cmap='Reds',
                        title=False,
                        fontsize=10,
                        showfig=None,
                        savefig=None,
                        format='pdf',
                        figsize=(5,4)
                        ):
    '''Plot a heatmap of pairwise similarities between GRNs of different cell states

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    score: `str` (default: 'AUPRC')
        metric to use, choose between 'AUROC' and 'AUPRC'
    edges: `str` (default: 'all')
        which edges to consider for similarity, choose between 'all', 'positive', 'negative'
    cmap: `pyplot colormap` (default='Reds')
        colormap to use. A list of accepted colormaps can be found at:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    title: `str` or `Bool` (default='False')
        plot title, must be provided as a string
    fontsize: `int` (default: 10)
        fontsize for figure
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
    assert score in ['AUROC', 'AUPRC'], 'Choose between score="AUROC" and score="AUPRC"'
    assert edges in ['all', 'positive', 'negative'], 'Choose between edges="all", "positive" or "negative"'

    types = sorted(list(set(list(adata.obs['clusters']))))

    mat = adata.uns['comparison_scores'][score + '_' + edges]

    plt.figure(figsize=figsize)
    plt.pcolor(mat, cmap=cmap)
    plt.colorbar().set_label(label=score+' score', size=fontsize)
    plt.xticks(np.arange(0.5, mat[0].size,1), types, rotation=45)
    plt.yticks(np.arange(0.5, mat[0].size, 1), types)
    plt.xlabel('Predictor State', fontsize=fontsize)
    plt.ylabel('Observed State', fontsize=fontsize)
    if title:
        plt.title(title)
    else:
        plt.title('Detection of ' + edges + ' edges')

    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format, dpi=300)

def adjecent_grn_score(adata,
                       path,
                       score='AUPRC',
                       edges='all',
                       loc='best',
                       fontsize=10,
                       color='r',
                       errorline_color='b',
                       elinewidth=1,
                       showfig=None,
                       savefig=None,
                       format='pdf',
                       figsize=(5,3)
                       ):
    '''
    Plot the pairwise GRN similarity between consecitive cell states along a transition

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    path: `list`
        list of cell states along the transition path
    score: `str` (default: 'AUPRC')
        metric to use, choose between 'AUROC' and 'AUPRC'
    edges: `str` (default: 'all')
        which edges to consider for similarity, choose between 'all', 'positive', 'negative'
    loc: `str` (default='best')
        location of legend. Details on legend location can be found at:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    fontsize: `int` (default: 10)
        fontsize for figure
    color: `str` (default: 'r')
        color of plot. A full list of accepted named colors can be found at:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    errorline_color: `str` (default: 'b')
        color of deviation bars. A full list of accepted named colors can be found at:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    elinewidth: `float` (default: 1)
        width of deviation bars
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format
    figsize: `tuple` (default: (5,3))
        size of figure

    Returns
    -------
    None

    '''
    assert score in ['AUROC', 'AUPRC'], 'Choose between score="AUROC" and score="AUPRC"'
    assert edges in ['all', 'positive', 'negative'], 'Choose between edges="all", "positive" or "negative"'

    types = sorted(list(set(list(adata.obs['clusters']))))
    mat = adata.uns['comparison_scores'][score + '_' + edges]

    path_score, average_score, std_score = np.zeros(len(path)-1), np.zeros(len(path)-1), np.zeros(len(path)-1)
    labs = []

    for i in range(len(path)-1):
        j, k = types.index(path[i]), types.index(path[i+1])
        path_score[i] = mat[j][k]
        average_score[i] = np.mean(np.delete(mat[j], j))
        std_score[i] = np.std(np.delete(mat[j], j))
        labs.append(path[i] + ' to\n' + path[i+1])

    plt.figure(figsize=figsize)

    plt.plot(np.arange(0, path_score.size, 1), path_score, 'o--', color=color, label='Transition')
    plt.errorbar(np.arange(0, path_score.size, 1), average_score, yerr=std_score, fmt='o--', color=errorline_color, elinewidth=elinewidth, label='Average over dataset')
    plt.xticks(np.arange(0, path_score.size, 1), labs)
    plt.ylabel(score + ' score', fontsize=fontsize)
    plt.legend(loc=loc, fontsize=fontsize)

    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format, dpi=300)