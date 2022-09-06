'''
tools for jacobian inference
'''

import numpy as np
import collections

import scvelo as scv

from .aux_functions import parameter_regression, estimate_degr, construct_jac, set_gene_axes, instability_scores


def initial_check(adata,
                  frac,
                  n_top_genes
                  ):
    '''
    Check that the selected number of top genes (n_top_genes) is not larger than
    the number of observables for spliceJAC inference

    Parameters
    ----------
    adata: anndata object of mRNA counts
    frac: fraction of cell selected from each cluster for bootstrapping inference
    n_top_genes: number of top genes to keep when running spliceJAC

    Returns
    -------
    None

    '''
    clusters = list(adata.obs['clusters'])
    counter = collections.Counter(clusters)
    if n_top_genes>int( frac*min(list(counter.values())) ):
        raise Exception('The selected number of n_top_genes is too large given the size of the smallest cluster (n=' + str( min(list(counter.values())) ) + ' cells) . Either decrease "n_top_genes" or increase "frac"')


def rescale_counts(mat):
    '''
    Rescale count of each gene to a zero mean

    Parameters
    ----------
    mat: mRNA count matrix

    Returns
    -------
    rescaled count matrix

    '''
    ncell, ngene = mat.shape
    avg = np.mean(mat, axis=0)
    center = np.asarray([avg]*ncell)
    return mat-center



def quick_regression(adata,
                     first_moment=True,
                     method='Ridge',
                     alpha=1,
                     beta=1.,
                     rescale=True
                     ):
    '''
    Run a quick cluster-wise Jacobian regression using all cells in each cluster
    Results are saved in adata.uns['all_cells']

    Parameters
    ----------
    adata: anndata object of mRNA counts
    first_moment: if True, use first moment of U and S to run regression
    method: regression method (Linear, Ridge, Lasso, default= Ridge)
    alpha: regularization strength coefficient for Ridge/Lasso (default= 1)
    beta: mRNA splicing rate constant (default= 1)
    rescale: if True, center counts on zero (default= True). If True, regression will enforce fit_int=False

    Returns
    -------
    None

    '''
    types = list(set(list(adata.obs['clusters'])))  # assumes location of cluster labels
    degr = adata.uns['degr_rates']
    inference = {}

    for i in range(len(types)):
        sel_adata = adata[adata.obs['clusters'] == types[i]]

        if first_moment:
            U = sel_adata.layers['Mu']
            S = sel_adata.layers['Ms']
        else:
            U = sel_adata.layers['unspliced'].toarray()
            S = sel_adata.layers['spliced'].toarray()

        fit_int=True
        if rescale:
            U, S = rescale_counts(U), rescale_counts(S)
            fit_int=False

        B, C, G = parameter_regression(U, S, method=method, alpha=alpha, fit_int=fit_int)
        J = construct_jac(B, degr, b=beta)
        w, v = np.linalg.eig(J)
        inference[types[i]] = [J, w, v]
    adata.uns['all_cells'] = inference


def long_regression(adata,
                    first_moment=True,
                    method='Ridge',
                    alpha=1,
                    beta=1.,
                    rescale=True,
                    nsim=10,
                    frac=0.9
                    ):
    '''
    Cluster-wise Jacobian regression multiple times with a randomly-selected subset of cells
    results are saved in adata.uns['jacobian_lists']

    Parameters
    ----------
    adata: anndata object of mRNA counts
    first_moment: if True, use first moment of U and S to run regression
    method: regression method (Linear, Ridge, Lasso, default: Ridge)
    alpha: regularization strength coefficient for Ridge/Lasso (default: 1)
    beta: mRNA splicing rate constant (default: 1)
    rescale: if True, center counts on zero (default: True). If True, regression will enforce fit_int=False
    nsim: number of independent regressions per cluster (default: 10)
    frac: fraction of cells to randomly select (default: 0.9)

    Returns
    -------
    None

    '''

    types = list(set(list(adata.obs['clusters'])))  # assumes location of cluster labels
    degr = adata.uns['degr_rates']

    jacobian_lists = {}
    for i in range(len(types)):
        print('Running subset regression on the ' + types[i] + ' cluster...')

        sel_adata = adata[adata.obs['clusters'] == types[i]]
        if first_moment:
            U = sel_adata.layers['Mu']
            S = sel_adata.layers['Ms']
        else:
            U = sel_adata.layers['unspliced'].toarray()
            S = sel_adata.layers['spliced'].toarray()
        n, m = U.shape

        fit_int = True
        if rescale:
            U, S = rescale_counts(U), rescale_counts(S)
            fit_int = False

        jac_list, w_list, v_list = [], [], []

        for j in range(nsim):
            indices = np.sort(np.random.choice(np.arange(0, n, 1, dtype='int'), size=int(frac * n), replace=False))
            U_sel, S_sel = U[indices], S[indices]

            B, C, G = parameter_regression(U_sel, S_sel, method=method, alpha=alpha, fit_int=fit_int)
            J = construct_jac(B, degr, b=beta)
            w, v = np.linalg.eig(J)
            jac_list.append(J)
            w_list.append(w)
            v_list.append(v)
        jacobian_lists[types[i]] = [jac_list, w_list, v_list]
    adata.uns['jacobian_lists'] = jacobian_lists


def compute_avg_jac(adata,
                    eps=0.9
                    ):
    '''
    Compute average Jacobian matrix from long_regression() results
    sets the fraction of smallest Jacobian elements to zero (by absolute value)

    Parameters
    ----------
    adata: anndata object of mRNA counts
    eps: fraction of weak interactions that are set to zero (default= 0.9)

    Returns
    -------
    None

    '''
    types = list(set(list(adata.obs['clusters'])))
    avg_jac = {}

    for i in range(len(types)):
        jac_list = adata.uns['jacobian_lists'][types[i]][0]
        J = np.mean(jac_list, axis=0)

        coeffs = np.sort(np.abs(np.ndarray.flatten(J)))
        t = coeffs[int(eps*coeffs.size)]

        J_filter = np.copy(J)
        n = J_filter.shape[0]
        for k in range(n):
            for j in range(n):
                if np.abs(J_filter[k][j])<t:
                    J_filter[k][j] = 0.

        w, v = np.linalg.eig(J_filter)
        avg_jac[types[i]] = [J_filter, w, v]

    adata.uns['average_jac'] = avg_jac


### do we need this function??? ###
# def vary_ngenes(adata,
#                 first_moment=True,
#                 method='Ridge',
#                 alpha=1, beta=1.,
#                 rescale=True,
#                 frac=0.9,
#                 ngenes=None,
#                 min_shared_counts=20
#                 ):
#     '''
#     Run regression for variable number of top_genes, results are saved in adata.uns['vary_ngenes']
#
#     Parameters
#     ----------
#     adata: anndata object of mRNA counts
#     first_moment: if True, use first moment of U and S to run regression
#     method:  regression method (Linear, Ridge, Lasso, default: Ridge)
#     alpha: regularization strength coefficient for Ridge/Lasso (default=1)
#     beta: mRNA splicing rate constant (default=1)
#     rescale: if True, center counts on zero (default=True). If True, regression will enforce fit_int=False
#     frac: fraction of cells to randomly select (default: 0.9)
#     ngenes: array with number of genes to use
#     min_shared_counts: minimum number of counts (unspliced+spliced) to keep a gene during filtering
#
#     Returns
#     -------
#     None
#
#     '''
#     clusters = list(adata.obs['clusters'])
#     counter = collections.Counter(clusters)
#     lim = int(frac * min(list(counter.values())))
#     if ngenes.any()==None:
#         ngenes = np.arange(10, lim, 10)
#     else:
#         assert ngenes[-1]<lim, f"The maximum number of genes that could used for Jacobian inference is {str(lim)} with the current setting: frac={str(frac)}"
#
#     scv.settings.verbosity = 0  #only show errors
#     ngene_var = []
#     for n in ngenes:
#         adata_copy = scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n, copy=True)
#         adata_copy.uns['degr_rates'] = estimate_degr(adata_copy, first_moment=first_moment)
#         quick_regression(adata_copy, first_moment=first_moment, method=method, alpha=alpha, beta=beta, rescale=rescale)
#         ngene_var.append(adata_copy.uns['all_cells'])
#         del adata_copy
#     adata.uns['vary_ngenes'] = [ngenes, ngene_var]


def estimate_jacobian(adata,
                      first_moment=True,
                      method='Ridge',
                      alpha=1,
                      beta=1.,
                      rescale=True,
                      nsim=10,
                      frac=0.9,
                      filter_and_norm=True,
                      min_shared_counts=20,
                      n_top_genes=20,
                      eps=0.9,
                      seed=100
                      ):
    '''
    Run cluster-wise Jacobian inference

    Parameters
    ----------
    adata: anndata object of mRNA counts
    first_moment: if True, use first moment of U and S to run regression
    method: regression method (Linear, Ridge, Lasso, default=Ridge)
    alpha: regularization strength coefficient for Ridge/Lasso (default=1)
    beta: mRNA splicing rate constant (default=1)
    rescale: if True, center counts on zero (default=True). If True, regression will enforce fit_int=False
    nsim: number of independent regressions per cluster (default=10)
    frac: fraction of cells to randomly select (default=0.9)
    filter_and_norm: if True, apply scvelo filter_and_normalize function
    min_shared_count: minimum number of shared count for scvelo filter_and_normalize function (degfault=20)
    n_top_genes: number of top genes to keep for scvelo filter_and_normalize function (default=20)
    eps: fraction of weak interactions that are set to zero (default=0.9)
    seed: seed for numpy random number generator (default=100)

    Returns
    -------
    None

    '''
    np.random.seed(seed)

    initial_check(adata, frac, n_top_genes)

    scv.settings.verbosity = 3
    if filter_and_norm:
        scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes)

    # compute moments if not done already
    if first_moment and 'Mu' not in adata.layers.keys():
        scv.pp.moments(adata)

    # step 0: set axes names
    set_gene_axes(adata)

    # step 1: compute degradation rates
    adata.uns['degr_rates'] = estimate_degr(adata, first_moment=first_moment)

    # step 2: run regression once with all cells
    print('Running quick regression...')
    quick_regression(adata, first_moment=first_moment, method=method, alpha=alpha, beta=beta, rescale=rescale)

    # step 3: run regression nsim times with only a fraction of cells per cluster - results saved in adata.uns['jacobian_lists']
    long_regression(adata, first_moment=first_moment, method=method, alpha=alpha, beta=beta, rescale=rescale, nsim=nsim, frac=frac)

    # step 4: take average jacobian - results saved in adata.uns['average_jac']
    compute_avg_jac(adata, eps=eps)

    # step 5: instability scores of individual genes
    instability_scores(adata)


