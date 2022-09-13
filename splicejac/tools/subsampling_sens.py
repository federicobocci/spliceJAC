'''
test the inference sensitivity to cell subsampling
'''

import numpy as np
import pandas as pd

from .aux_functions import parameter_regression

def count_sign_change(v1, v2):
    '''Count the number of sign changes between two matrices

    Parameters
    ----------
    v1: `~numpy.ndarray`
        matrix 1
    v2: `~numpy.ndarray`
        matrix 2

    Returns
    -------
    sign_frac: `float`
        fraction of changed signs

    '''
    assert v1.shape == v2.shape, 'Matrices with different shapes'
    c, n = 0, v1.shape[0]
    for i in range(n):
        for j in range(n):
            if np.sign(v1[i][j])!=np.sign(v2[i][j]):
                c = c + 1
    sign_frac = float(c)/(n*n)
    return sign_frac

def mat_distance(v1,
                 v2
                 ):
    '''Compute the distance between two matrices by summing element-wise difference

    Parameters
    ----------
    v1: `~numpy.ndarray`
        matrix 1
    v2: `~numpy.ndarray`
        matrix 2

    Returns
    -------
    mat_dist: `float`
        matrix distance normalized by number of elements

    '''
    assert v1.shape==v2.shape, 'Matrices with different shapes'
    n = v1.shape[0]
    mat_dist = np.sum( np.abs(v1-v2) )/(n*n)
    return mat_dist

def count_weight_sign(v1,
                      v2
                      ):
    '''Computes the distance between two matrices by summing element-wise difference

    Parameters
    ----------
    v1: `~numpy.ndarray`
        matrix 1
    v2: `~numpy.ndarray`
        matrix 2

    Returns
    -------
    elem_dist: `~numpy.ndarray`
        matrix distance normalized by number of elements

    '''
    assert v1.shape == v2.shape, 'Matrices with different shapes'
    c, n = 0, v1.shape[0]
    for i in range(n):
        for j in range(n):
            if np.sign(v1[i][j])!=np.sign(v2[i][j]):
                c = c + np.abs(v1[i][j]-v2[i][j])
    elem_dist = float(c)/(n*n)
    return elem_dist

def test_sub_sampling(adata,
                      cluster,
                      frac,
                      nsim=10
                      ):
    '''Test the inference of gene-gene interaction matrix with subsampling for a cluster

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster: `str`
        cluster selected for inference
    frac: `float`
        fraction of cells to randomly select between [0,1]
    nsim: `int` (default: 10)
        number of independent simulations

    Returns
    -------
    sign_frac: `~numpy.ndarray`
        fraction of correct signs
    dist: `~numpy.ndarray`
        distance between reference gene-gene interaction matrix and inferred matrix using only a fraction of cells
    weight_sign: `~numpy.ndarray`
        distance based on weighted sum of correct signs

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
    '''Test the inference of gene-gene interaction matrix as a function of fraction of selected cells
    Results are stored in adata.uns['sens_summary']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    frac: `~numpy.ndarray` (default: numpy.arange(0.1, 0.91, 0.1))
        fraction of cells to randomly select
    seed: `int` (default=100)
        seed for random cell selection for reproducibility
    nsim: `int` (default: 10)
        number of independent simulations

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