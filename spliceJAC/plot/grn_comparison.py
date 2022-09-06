'''
functions to quantitatively compare GRNs across cell states
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_grn_comparison(adata, score='AUPRC', edges='all', cmap='Reds', title=False, fontsize=10,
                        showfig=False, savefig=True, figname='grn_scores.pdf', format='pdf', figsize=(5,4)):
    '''
    Plot a heatmap of pairwise similarities between GRNs of different cell states

    Parameters
    ----------
    adata: anndata object
    score: metric to use, choose between 'AUROC' and 'AUPRC' (default='AUPRC')
    edges: which edges to consider for similarity, choose between 'all', 'positive', 'negative' (default='all')
    cmap: pyplot colormap (default='Reds')
    title: plot title, must be provided as a string (default='False')
    fontsize: font size of figure (default=10)
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='grn_scores.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(5,4))

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
        plt.savefig(figname, format=format, dpi=300)

def adjecent_grn_score(adata, path, score='AUPRC', edges='all',
                       loc='best', fontsize=10, color='r', errorline_color='b', elinewidth=1,
                       showfig=False, savefig=True, figname='adjacent_grn_score.pdf', format='pdf', figsize=(5,3)):
    '''
    Plot the pairwise GRN similarity between consecitive cell states along a transition

    Parameters
    ----------
    adata: anndata object
    path: list of cell states along the transition path
    score: metric to use, choose between 'AUROC' and 'AUPRC' (default='AUPRC')
    edges: which edges to consider for similarity, choose between 'all', 'positive', 'negative' (default='all')
    loc: legend location (default='best')
    fontsize: font size of figure (default=10)
    color: color of plot (default='r')
    errorline_color: color of deviation bars (default='b')
    elinewidth: width of deviation bars (default=1)
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=True)
    figname: name of saved figure including path (default='grn_scores.pdf')
    format: format of saved figure (default='pdf')
    figsize: size of figure (default=(5,4))

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

    plt.plot(np.arange(0, path_score.size, 1), path_score, color=color, 'o--', label='Transition')
    plt.errorbar(np.arange(0, path_score.size, 1), average_score, yerr=std_score, fmt='o--', color=errorline_color, elinewidth=elinewidth, label='Average over dataset')
    plt.xticks(np.arange(0, path_score.size, 1), labs)
    plt.ylabel(score + ' score', fontsize=fontsize)
    plt.legend(loc=loc, fontsize=fontsize)

    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)