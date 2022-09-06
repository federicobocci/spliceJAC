'''
test the inference sensitivity to cell subsampling
'''

import numpy as np
import pandas as pd

from .aux_functions import parameter_regression
# from . import analysis

def count_sign_change(v1, v2):
    '''
    Count the number of sign changes between two matrices

    Parameters
    ----------
    v1: matrix 1
    v2: matrix 2

    Returns
    -------
    fraction of changed signs

    '''
    assert v1.shape == v2.shape, 'Matrices with different shapes'
    c, n = 0, v1.shape[0]
    for i in range(n):
        for j in range(n):
            if np.sign(v1[i][j])!=np.sign(v2[i][j]):
                c = c + 1
    return float(c)/(n*n)

def mat_distance(v1, v2):
    '''
    Compute the distance between two matrices by summing element-wise difference

    Parameters
    ----------
    v1: matrix 1
    v2: matrix 2

    Returns
    -------
    matrix distance normalized by number of elements

    '''
    assert v1.shape==v2.shape, 'Matrices with different shapes'
    n = v1.shape[0]
    return np.sum( np.abs(v1-v2) )/(n*n)

def count_weight_sign(v1, v2):
    '''
    Computes the distance between two matrices by summing element-wise difference

    Parameters
    ----------
    v1: matrix 1
    v2: matrix 2

    Returns
    -------
    matrix distance normalized by number of elements

    '''
    assert v1.shape == v2.shape, 'Matrices with different shapes'
    c, n = 0, v1.shape[0]
    for i in range(n):
        for j in range(n):
            if np.sign(v1[i][j])!=np.sign(v2[i][j]):
                c = c + np.abs(v1[i][j]-v2[i][j])
    return float(c)/(n*n)

def test_sub_sampling(adata,
                      cluster,
                      frac,
                      nsim=10
                      ):
    '''
    Test the inference of gene-gene interaction matrix with subsampling for a cluster

    Parameters
    ----------
    adata: anndata object of mRNA counts
    cluster: cluster selected for inference
    frac: fraction of cells to randomly select
    nsim: number of independent simulations

    Returns
    -------
    sign_frac: fraction of correct signs
    dist: distance between reference gene-gene interaction matrix (using all cluster cells) and inferred matrix using only a fraction of cells
    weight_sign: distance based on weighted sum of correct signs

    '''
    sel_adata = adata[adata.obs['clusters'] == cluster]
    U = sel_adata.layers['unspliced'].toarray()
    S = sel_adata.layers['spliced'].toarray()
    B_ref, C, G = parameter_regression(U, S)

    n = int( U.shape[0]*frac )
    sign_frac, dist, weight_sign = np.zeros(nsim), np.zeros(nsim), np.zeros(nsim)

    for i in range(nsim):
        keep = np.random.choice(np.arange(0, U.shape[0], 1), n, replace=False)
        U_sub, S_sub = U[keep], S[keep]
        B, C, G = parameter_regression(U_sub, S_sub)
        sign_frac[i] = count_sign_change(B, B_ref)
        dist[i] = mat_distance(B, B_ref)
        weight_sign[i] = count_weight_sign(B, B_ref)
    return sign_frac, dist, weight_sign


def subsampling_sens(adata,
                  frac = np.arange(0.1, 0.91, 0.1),
                  seed=100,
                  nsim=10
                  ):
    '''
    Test the inference of gene-gene interaction matrix as a function of fraction of selected cells
    Results are stored in adata.uns['sens_summary']

    Parameters
    ----------
    adata: anndata object of mRNA counts
    frac: fraction of cells to randomly select (default=np.arange(0.1, 0.91, 0.1))
    seed: seed for random cell selection (default=100)
    nsim: number oof independent simulations (default=10)

    Returns
    -------
    None

    '''
    np.random.seed(seed)

    types = list(set(list(adata.obs['clusters'])))  # assumes location of cluster labels
    sens = {}

    for i in range(len(types)):
        print('Subsampling the ' + types[i] + ' cluster...')
        sign_list, dist_list, weight_list = [], [], []
        avg_sign, avg_dist, avg_weight = np.zeros(frac.size), np.zeros(frac.size), np.zeros(frac.size)
        std_sign, std_dist, std_weight = np.zeros(frac.size), np.zeros(frac.size), np.zeros(frac.size)

        for j in range(frac.size):
            sign, dist, weight_sign = test_sub_sampling(adata, types[i], frac[j], nsim=nsim)
            avg_sign[j], avg_dist[j], avg_weight[j] = np.mean(sign), np.mean(dist), np.mean(weight_sign)
            std_sign[j], std_dist[j], std_weight[j] = np.std(sign), np.std(dist), np.std(weight_sign)

            sign_list.append(sign)
            dist_list.append(dist)
            weight_list.append(weight_sign)

        df = pd.DataFrame(data={'frac': frac, 'sign_frac': avg_sign, 'dist': avg_dist, 'weighted_sign': avg_weight})
        sens[types[i]] = df
    adata.uns['sens_summary'] = sens