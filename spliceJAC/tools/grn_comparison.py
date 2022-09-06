import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def edge_detection_score(m1, m2):
    '''
    Compute AUROC/AUPRC related metrics between matrices m1 and m2 disregarding the interaction signs

    Parameters
    ----------
    m1: observation matrix
    m2: ground truth matrix

    Returns
    -------
    fpr: false positive rate
    tpr: true positive rate
    auroc: area under the receiver characteristic curve
    precision: precision
    recall: recall
    auprc: area under the precision recall curve

    '''
    obs = np.abs(np.ndarray.flatten(m1))
    truth = np.abs(np.ndarray.flatten(np.sign(m2)))

    fpr, tpr, thresholds = metrics.roc_curve(truth, obs)
    auroc = metrics.auc(fpr, tpr)

    precision, recall, thresholds = metrics.precision_recall_curve(truth, obs)
    auprc = metrics.auc(recall, precision)

    return fpr, tpr, auroc, precision, recall, auprc

def sign_detection_score(m1, m2):
    '''
    Compute AUROC/AUPRC related metrics between matrices m1 and m2 when considering the interaction signs

    Parameters
    ----------
    m1: observation matrix
    m2: ground truth matrix

    Returns
    -------
    fpr: false positive rate
    tpr: true positive rate
    auroc: area under the receiver characteristic curve
    precision: precision
    recall: recall
    auprc: area under the precision recall curve

    '''

    obs = np.ndarray.flatten(m1)
    truth = np.ndarray.flatten(np.sign(m2))

    # compute positive and negative ground truth arrays
    grnd_pos, grnd_neg = np.zeros(truth.size), np.zeros(truth.size)
    for i in range(grnd_pos.size):
        if truth[i] == 1:
            grnd_pos[i] = 1
        elif truth[i] == -1:
            grnd_neg[i] = -1

    fpr, tpr, auroc = dict(), dict(), dict()
    fpr[0], tpr[0], thresholds = metrics.roc_curve(grnd_pos, obs)
    fpr[1], tpr[1], thresholds = metrics.roc_curve(grnd_neg, obs, pos_label=-1)
    auroc[0], auroc[1] = metrics.auc(fpr[0], tpr[0]), metrics.auc(fpr[1], tpr[1])

    recall, precision, auprc = dict(), dict(), dict()
    precision[0], recall[0], thresholds = metrics.precision_recall_curve(grnd_pos, obs)
    precision[1], recall[1], thresholds = metrics.precision_recall_curve(grnd_neg, obs, pos_label=-1)
    auprc[0], auprc[1] = metrics.auc(recall[0], precision[0]), metrics.auc(recall[1], precision[1])

    return fpr, tpr, auroc, precision, recall, auprc


def grn_comparison(adata):
    '''
    Compute AUROC/AUPRC scores for all pairs of state-specific gene regulatory networks
    Results are stored in adata.uns['comparison_scores']

    Parameters
    ----------
    adata: anndata object

    Returns
    -------
    None

    '''
    types = sorted(list(set(list(adata.obs['clusters']))))
    auroc_tot, auprc_tot = np.zeros((len(types), len(types))), np.zeros((len(types), len(types)))
    auroc_pos, auprc_pos = np.zeros((len(types), len(types))), np.zeros((len(types), len(types)))
    auroc_neg, auprc_neg = np.zeros((len(types), len(types))), np.zeros((len(types), len(types)))

    for i in range(len(types)):
        for j in range(len(types)):
            if types[i]!=types[j]:
                m1, m2 = adata.uns['average_jac'][types[i]][0], adata.uns['average_jac'][types[j]][0]
                fpr, tpr, auroc_tot[i][j], precision, recall, auprc_tot[i][j] = edge_detection_score(m1, m2)
                fpr, tpr, auroc, precision, recall, auprc = sign_detection_score(m1, m2)
                auroc_pos[i][j], auroc_neg[i][j] = auroc[0], auroc[1]
                auprc_pos[i][j], auprc_neg[i][j] = auprc[0], auprc[1]

    scores = { 'AUROC_all': auroc_tot, 'AUPRC_all': auprc_tot, 'AUROC_positive':auroc_pos, 'AUPRC_positive':auprc_pos,
               'AUROC_negative':auroc_neg, 'AUPRC_negative': auprc_neg }
    adata.uns['comparison_scores'] = scores


#
# def plot_grn_comparison(adata, score='AUPRC', edges='all', cmap='Reds', title=False, fontsize=10,
#                         showfig=False, savefig=True, figname='grn_scores.pdf', format='pdf', figsize=(5,4)):
#     assert score in ['AUROC', 'AUPRC'], 'Choose between score="AUROC" and score="AUPRC"'
#     assert edges in ['all', 'positive', 'negative'], 'Choose between edges="all", "positive" or "negative"'
#
#     types = sorted(list(set(list(adata.obs['clusters']))))
#
#     mat = adata.uns['comparison_scores'][score + '_' + edges]
#
#     plt.figure(figsize=figsize)
#     plt.pcolor(mat, cmap=cmap)
#     plt.colorbar().set_label(label=score+' score', size=fontsize)
#     plt.xticks(np.arange(0.5, mat[0].size,1), types, rotation=45)
#     plt.yticks(np.arange(0.5, mat[0].size, 1), types)
#     plt.xlabel('Predictor State', fontsize=fontsize)
#     plt.ylabel('Observed State', fontsize=fontsize)
#     if title:
#         plt.title(title)
#     else:
#         plt.title('Detection of ' + edges + ' edges')
#
#     plt.tight_layout()
#
#     if showfig:
#         plt.show()
#     if savefig:
#         plt.savefig(figname, format=format, dpi=300)
#
# def adjecent_grn_score(adata, path, score='AUPRC', edges='all',
#                        loc='best', fontsize=10, errorline_color='b', elinewidth=1,
#                        showfig=False, savefig=True, figname='adjacent_grn_score.pdf', format='pdf', figsize=(5,3)):
#     assert score in ['AUROC', 'AUPRC'], 'Choose between score="AUROC" and score="AUPRC"'
#     assert edges in ['all', 'positive', 'negative'], 'Choose between edges="all", "positive" or "negative"'
#
#     types = sorted(list(set(list(adata.obs['clusters']))))
#     mat = adata.uns['comparison_scores'][score + '_' + edges]
#
#     path_score, average_score, std_score = np.zeros(len(path)-1), np.zeros(len(path)-1), np.zeros(len(path)-1)
#     labs = []
#
#     for i in range(len(path)-1):
#         j, k = types.index(path[i]), types.index(path[i+1])
#         path_score[i] = mat[j][k]
#         average_score[i] = np.mean(np.delete(mat[j], j))
#         std_score[i] = np.std(np.delete(mat[j], j))
#         labs.append(path[i] + ' to\n' + path[i+1])
#
#     plt.figure(figsize=figsize)
#
#     plt.plot(np.arange(0, path_score.size, 1), path_score, 'ro--', label='Transition')
#     plt.errorbar(np.arange(0, path_score.size, 1), average_score, yerr=std_score, fmt='o--', color=errorline_color, elinewidth=elinewidth, label='Average over dataset')
#     plt.xticks(np.arange(0, path_score.size, 1), labs)
#     plt.ylabel(score + ' score', fontsize=fontsize)
#     plt.legend(loc=loc, fontsize=fontsize)
#
#     plt.tight_layout()
#
#     if showfig:
#         plt.show()
#     if savefig:
#         plt.savefig(figname, format=format, dpi=300)
