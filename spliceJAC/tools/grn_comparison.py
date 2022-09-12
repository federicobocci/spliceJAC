'''
functions to quantify grn similarity across cell states
'''
import numpy as np
from sklearn import metrics

def edge_detection_score(m1,
                         m2
                         ):
    '''Compute AUROC/AUPRC related metrics between matrices m1 and m2 disregarding the interaction signs

    Parameters
    ----------
    m1: `~numpy.ndarray`
        observation matrix
    m2: `~numpy.ndarray`
        ground truth matrix

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

def sign_detection_score(m1,
                         m2
                         ):
    '''Compute AUROC/AUPRC related metrics between matrices m1 and m2 when considering the interaction signs

    Parameters
    ----------
    m1: observation matrix
    m2: ground truth matrix

    Returns
    -------
    fpr: `float`
        false positive rate
    tpr: `float`
        true positive rate
    auroc: `float`
        area under the receiver characteristic curve
    precision: `float`
        precision
    recall: `float`
        recall
    auprc: `float`
        area under the precision recall curve

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
    '''Compute AUROC/AUPRC scores for all pairs of state-specific gene regulatory networks
    Results are stored in adata.uns['comparison_scores']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix

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
