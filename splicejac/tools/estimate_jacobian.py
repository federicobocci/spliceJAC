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
    '''Check that the selected number of genes (n_top_genes) does not exceed the number of observables

    The maximum value admitted for the n_top_genes variable to ensure a unique solution is given by the number of cells
    in the smaller cluster times the fraction of cells per cluster used during each iteration of inference (frac)

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    frac: `float`
        fraction of cell selected from each cluster for bootstrapping inference
    n_top_genes: `int`
        number of top genes to keep when running spliceJAC

    Returns
    -------
    None

    '''
    clusters = list(adata.obs['clusters'])
    counter = collections.Counter(clusters)
    if n_top_genes>int( frac*min(list(counter.values())) ):
        raise Exception('The selected number of n_top_genes is too large given the size of the smallest cluster (n=' +
                        str( min(list(counter.values())) ) + ' cells) . Decrease "n_top_genes" or increase "frac"')


def rescale_counts(mat):
    '''
    Rescale count of each gene to a zero mean

    Parameters
    ----------
    mat: `~numpy.ndarray`
        count matrix

    Returns
    -------
    rescaled: `~numpy.ndarray`
        rescaled count matrix

    '''
    ncell, ngene = mat.shape
    avg = np.mean(mat, axis=0)
    center = np.asarray([avg]*ncell)
    rescaled = mat-center
    return rescaled



def quick_regression(adata,
                     first_moment=True,
                     method='Ridge',
                     alpha=1,
                     beta=1.,
                     rescale=True
                     ):
    '''Run a single cluster-wise regression using all cells in each cluster
    Results are saved in adata.uns['all_cells']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    first_moment: `Bool` (default: True)
        if True, use first moment of U and S to run regression
    method: `str` (default: Ridge)
        regression method, choose between Linear, Ridge or Lasso
    alpha: `float` (default: 1)
        regularization coefficient for Ridge and Lasso
    beta: `float` (default: 1)
        mRNA splicing rate constant
    rescale: `Bool` (default: True)
        if True, center counts on zero (default= True). rescale=True enforces fit_int=False

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
    '''Run cluster-wise Jacobian regression multiple times with a randomly-selected subset of cells
    results are saved in adata.uns['jacobian_lists']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    first_moment: `Bool` (default: True)
        if True, use first moment of U and S to run regression
    method: `str` (default: Ridge)
        regression method, choose between Linear, Ridge or Lasso
    alpha: `float` (default: 1)
        regularization coefficient for Ridge and Lasso
    beta: `float` (default: 1)
        mRNA splicing rate constant
    rescale: `Bool` (default: True)
        if True, center counts on zero (default= True). rescale=True enforces fit_int=False
    nsim: `int` (default: 10)
        number of independent regressions per cluster
    frac: `float` (default: 0.9)
        fraction of cells to randomly select (bound in [0,1])

    Returns
    -------
    None

    '''
    types = list(set(list(adata.obs['clusters'])))
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
        # for each cluster, save vectors of Jacobians, engenvalues, and eigenvectors from multiple inferences
        jacobian_lists[types[i]] = [jac_list, w_list, v_list]
    adata.uns['jacobian_lists'] = jacobian_lists


def compute_avg_jac(adata,
                    eps=0.9
                    ):
    '''Compute average Jacobian matrix from long_regression() results
    The percentage of smallest Jacobian elements (by absolute value) are set to zero based on the parameter epsilon

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    eps: `float` (default= 0.9)
        fraction of weakest Jacobian elements that are set to zero (by absolute value)

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

        # change this to only the GRN part of the Jacobian?
        # compute eigenspecturm before this?
        J_filter = np.copy(J)
        n = J_filter.shape[0]
        for k in range(n):
            for j in range(n):
                if np.abs(J_filter[k][j])<t:
                    J_filter[k][j] = 0.

        w, v = np.linalg.eig(J_filter)
        avg_jac[types[i]] = [J_filter, w, v]

    adata.uns['average_jac'] = avg_jac


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
    '''Run cluster-wise Jacobian inference

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    first_moment: `Bool` (default: True)
        if True, use first moment of U and S to run regression
    method: `str` (default: Ridge)
        regression method, choose between Linear, Ridge or Lasso
    alpha: `float` (default: 1)
        regularization coefficient for Ridge and Lasso
    beta: `float` (default: 1)
        mRNA splicing rate constant
    rescale: `Bool` (default: True)
        if True, center counts on zero (default= True). rescale=True enforces fit_int=False
    nsim: `int` (default: 10)
        number of independent regressions per cluster
    frac: `float` (default: 0.9)
        fraction of cells to randomly select (bound in [0,1])
    filter_and_norm: `Bool` (default: True)
        if True, apply scvelo filter_and_normalize function to the count matrix
    min_shared_count: `int` (default: 20)
        minimum number of shared count for the scvelo filter_and_normalize function
    n_top_genes: `int` (default: 20)
        number of top genes for the scvelo filter_and_normalize function
    eps: `float` (default= 0.9)
        fraction of weakest Jacobian elements that are set to zero (by absolute value)
    seed: `int` (default: 100)
        seed for numpy random number generator (for reproducibility)

    Returns
    -------
    None

    '''
    assert eps>0 and eps<1, "The parameter `epsilon` must be bound in [0,1] (default: 0.9)"
    assert frac>0 and frac<1, "The parameter `frac` must be bound in [0,1] (default: 0.9)"
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
    # should this step be canceled?
    print('Running quick regression...')
    quick_regression(adata, first_moment=first_moment, method=method, alpha=alpha, beta=beta, rescale=rescale)

    # step 3: run regression nsim times
    long_regression(adata, first_moment=first_moment, method=method, alpha=alpha, beta=beta, rescale=rescale, nsim=nsim,
                    frac=frac)

    # step 4: Compute average Jacobian
    compute_avg_jac(adata, eps=eps)

    # step 5: instability scores of individual genes
    instability_scores(adata)


