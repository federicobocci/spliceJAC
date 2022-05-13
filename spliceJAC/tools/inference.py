'''
tools for jacobian inference
'''

import numpy as np
import pandas as pd
import collections
from sklearn.linear_model import Ridge, LinearRegression, Lasso

import scvelo as scv

from . import analysis


def initial_check(adata,
                  frac,
                  n_top_genes
                  ):
    '''Check that the selected number of top genes (n_top_genes) is not larger than the number of observables for spliceJAC inference

    Parameters
    ----------
    adata:
        anndata object of mRNA counts
    frac:
        fraction of cell selected from each cluster for bootstrapping inference
    n_top_genes:
        number of top genes to keep when running spliceJAC

    Returns
    -------
    None

    '''
    clusters = list(adata.obs['clusters'])
    counter = collections.Counter(clusters)
    if n_top_genes>int( frac*min(list(counter.values())) ):
        raise Exception('The selected number of n_top_genes is too large given the size of the smallest cluster (n=' + str( min(list(counter.values())) ) + ' cells) . Either decrease "n_top_genes" or increase "frac"')


def rescale_counts(mat):
    ''' rescale count of each gene to a zero mean

    Parameters
    ----------
    mat:
        mRNA count matrix

    Returns
    -------
    ndarray
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
    '''Run a quick cluster-wise Jacobian regression using all cells in each cluster - results are saved in adata.uns['all_cells']

    Parameters
    ----------
    adata:
        anndata object of mRNA counts
    first_moment:
        if True, use first moment of U and S to run regression
    method:
        regression method (Linear, Ridge, Lasso, default= Ridge)
    alpha:
        regularization strength coefficient for Ridge/Lasso (default= 1)
    beta:
        mRNA splicing rate constant (default= 1)
    rescale:
        if True, center counts on zero (default= True). If True, regression will enforce fit_int=False

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
    '''Cluster-wise Jacobian regression multiple times with a randomly-selected subset of cells
    results are saved in adata.uns['jacobian_lists']

    Parameters
    ----------
    adata:
        anndata object of mRNA counts
    first_moment:
        if True, use first moment of U and S to run regression
    method:
        regression method (Linear, Ridge, Lasso, default: Ridge)
    alpha:
        regularization strength coefficient for Ridge/Lasso (default: 1)
    beta:
        mRNA splicing rate constant (default: 1)
    rescale:
        if True, center counts on zero (default: True). If True, regression will enforce fit_int=False
    nsim:
        number of independent regressions per cluster (default: 10)
    frac:
        fraction of cells to randomly select (default: 0.9)

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
    ''' compute average Jacobian matrix from long_regression() results
    sets the fraction of smallest Jacobian elements to zero (by absolute value)

    Parameters
    ----------
    adata:
        anndata object of mRNA counts
    eps:
        fraction of weak interactions that are set to zero (default= 0.9)

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



def vary_ngenes(adata,
                first_moment=True,
                method='Ridge',
                alpha=1, beta=1.,
                rescale=True,
                frac=0.9,
                ngenes=None,
                min_shared_counts=20
                ):
    '''Run regression for variable number of top_genes, results are saved in adata.uns['vary_ngenes']

    Parameters
    ----------
    adata:
        anndata object of mRNA counts
    first_moment:
        if True, use first moment of U and S to run regression
    method:
        regression method (Linear, Ridge, Lasso, default: Ridge)
    alpha:
        regularization strength coefficient for Ridge/Lasso (default=1)
    beta:
        mRNA splicing rate constant (default=1)
    rescale:
        if True, center counts on zero (default=True). If True, regression will enforce fit_int=False
    frac:
        fraction of cells to randomly select (default: 0.9)
    ngenes:
        array with number of genes to use
    min_shared_counts:
        minimum number of counts (unspliced+spliced) to keep a gene during filtering

    Returns
    -------
    None

    '''
    clusters = list(adata.obs['clusters'])
    counter = collections.Counter(clusters)
    lim = int(frac * min(list(counter.values())))
    if ngenes.any()==None:
        ngenes = np.arange(10, lim, 10)
    else:
        assert ngenes[-1]<lim, f"The maximum number of genes that could used for Jacobian inference is {str(lim)} with the current setting: frac={str(frac)}"

    scv.settings.verbosity = 0  #only show errors
    ngene_var = []
    for n in ngenes:
        adata_copy = scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n, copy=True)
        adata_copy.uns['degr_rates'] = estimate_degr(adata_copy, first_moment=first_moment)
        quick_regression(adata_copy, first_moment=first_moment, method=method, alpha=alpha, beta=beta, rescale=rescale)
        ngene_var.append(adata_copy.uns['all_cells'])
        del adata_copy
    adata.uns['vary_ngenes'] = [ngenes, ngene_var]


def estimate_jacobian(adata,
                      first_moment=True,
                      method='Ridge',
                      alpha=1, beta=1.,
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
    adata:
        anndata object of mRNA counts
    first_moment:
        if True, use first moment of U and S to run regression
    method:
        regression method (Linear, Ridge, Lasso, default=Ridge)
    alpha:
        regularization strength coefficient for Ridge/Lasso (default=1)
    beta:
        mRNA splicing rate constant (default=1)
    rescale:
        if True, center counts on zero (default=True). If True, regression will enforce fit_int=False
    nsim:
        number of independent regressions per cluster (default=10)
    frac:
        fraction of cells to randomly select (default=0.9)
    filter_and_norm:
        if True, apply scvelo filter_and_normalize function
    min_shared_count:
        minimum number of shared count for scvelo filter_and_normalize function (degfault=20)
    n_top_genes:
        number of top genes to keep for scvelo filter_and_normalize function (default=20)
    eps:
        fraction of weak interactions that are set to zero (default=0.9)
    seed:
        seed for numpy random number generator (default=100)

    Returns
    -------
    None

    '''
    np.random.seed(seed)

    initial_check(adata, frac, n_top_genes)

    # print('Running regression with variable n_top_genes...')
    # vary_ngenes(adata)

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
    analysis.instability_scores(adata)

    # step 6: construct aggregated network
    # analysis.unravel_jac(adata)
    # analysis.filter_int(adata, eps=eps)
    # analysis.consensus_sign(adata, p=p)


def parameter_regression(U_data,
                         S_data,
                         method='Ridge',
                         alpha=1,
                         fit_int=True
                         ):
    '''run regression to infer spliced-unspliced interaction coefficients

    Parameters
    ----------
    U_data:
        n_obs x n_genes count matrix of unspliced counts
    S_data:
        n_obs x n_genes count matrix of spliced counts
    method:
        regression method, either Linear, Ridge or Lasso (default=Ridge)
    alpha:
        regularization coefficient
    fit_int:
        if True, set the fit_intercept parameter to True (default=True)

    Returns
    -------
    mat:
        gene-gene interaction matrix
    interc:
        intercept vector
    degr:
        degradation coefficient vector

    '''
    assert method == 'Ridge' or method == 'Lasso' or method=='Linear', 'Please choose either Ridge or Lasso as method option'

    if method=='Linear':
        reg = LinearRegression(fit_intercept=fit_int)
    elif method == 'Ridge':
        reg = Ridge(alpha=alpha, fit_intercept=fit_int)
    elif method == 'Lasso':
        reg = Lasso(alpha=alpha, fit_intercept=fit_int)

    ncell, ngene = U_data.shape
    mat = np.zeros((ngene, ngene))
    interc = np.zeros(ngene)
    degr = np.zeros(ngene)

    for i in range(ngene):
        S_use = np.delete(S_data, i, 1)
        reg.fit(S_use, U_data[:, i])
        coeffs = reg.coef_

        mat[i][0:i] = coeffs[0:i]
        mat[i][i + 1:] = coeffs[i:]
        interc[i] = reg.intercept_

        # fit spliced degradation rate
        reg_g = LinearRegression(fit_intercept=False)
        reg_g.fit(S_data[:, [i]], U_data[:, i])
        degr[i] = reg_g.coef_

    return mat, interc, degr


def estimate_degr(adata,
                  first_moment=True
                  ):
    '''Estimate degradation rate coefficient vector

    Parameters
    ----------
    adata:
        anndata object of mRNA counts
    first_moment:
        if True, use first moment of U and S to run regression

    Returns
    -------
    degr:
        degradation coefficient vector

    '''
    if first_moment:
        U_data = adata.layers['Mu']
        S_data = adata.layers['Ms']
    else:
        U_data = adata.layers['unspliced'].toarray()
        S_data = adata.layers['spliced'].toarray()

    ncell, ngene = U_data.shape
    degr = np.zeros(ngene)

    for i in range(ngene):
        reg_g = LinearRegression(fit_intercept=False)
        reg_g.fit(S_data[:, [i]], U_data[:, i])
        degr[i] = reg_g.coef_
    return degr


def construct_jac(mat,
                  degr,
                  b=1
                  ):
    '''Construct a Jacobian matrix given the gene-gene interactions and degradation rates

    Parameters
    ----------
    mat: matrix of gene-gene interactions computed with parameter_regression()
    degr: degradation coefficient vector computed with estimate_degr()
    b: splicing rate constant (default=1)

    Returns
    -------
    J
        Jacobian matrix

    '''
    ngene = mat.shape[0]

    jac1 = np.diag(-b*np.ones(ngene))   # unspliced-unspliced part
    jac2 = np.diag(b*np.ones(ngene))    # spliced-unspliced part
    jac3 = np.diag(-degr)               # spliced-spliced part

    J1 = np.concatenate([jac1, mat], axis=1)
    J2 = np.concatenate([jac2, jac3], axis=1)
    J = np.concatenate([J1, J2])

    return J


def set_gene_axes(adata):
    '''Set up a axes name list with unspliced and spliced genes

    Parameters
    ----------
    adata:
        anndata object of mRNA counts

    Returns
    -------
    None

    '''
    genes = list(adata.var_names)
    axes = ['' for i in range(2*len(genes))]
    for i in range(len(genes)):
        axes[i] = genes[i] + '_U'
        axes[i + len(genes)] = genes[i]
    adata.uns['axes'] = axes




######## sensitivity to method

def test_sampling_method(adata,
                         alpha_ridge = np.array([0.01, 0.1, 1, 10, 100]),
                         alpha_lasso = np.array([0.0001, 0.001, 0.01, 0.1, 1])
                         ):
    '''Compare methods for gene-gene interaction parameter regression
    Results are stored in adata.uns['method_sens'] and adata.uns['sens_coeff']

    Parameters
    ----------
    adata:
        anndata object of mRNA counts
    alpha_ridge:
        array of shrinkage coefficients to test for Ridge regression
    alpha_lasso:
        array of shrinkage coefficients to test for Lasso regression

    Returns
    -------
    None

    '''
    types = list(set(list(adata.obs['clusters'])))

    method_sens = {}

    x = np.logspace(-5, 0, num=100)
    y_reg, y_ridge, y_lasso = np.zeros(x.size), np.zeros((alpha_ridge.size, x.size)), np.zeros((alpha_lasso.size, x.size))

    for i in range(len(types)):
        print('Running method sensitivity on the ' + types[i] + ' cluster...')

        ridge_jac, lasso_jac = [], []

        sel_adata = adata[adata.obs['clusters'] == types[i]]
        U = sel_adata.layers['unspliced'].toarray()
        S = sel_adata.layers['spliced'].toarray()

        # linear regression
        B_lin, C, G = parameter_regression(U, S, method='Linear')
        y_reg = analysis.coeff_dist(B_lin, x=x)

        # Ridge regression
        for j in range(alpha_ridge.size):
            B_ridge, C, G = parameter_regression(U, S, alpha=alpha_ridge[j])
            ridge_jac.append(B_ridge)
            y_ridge[j] = analysis.coeff_dist(B_ridge)

        # Lasso regression
        for j in range(alpha_lasso.size):
            B_lasso, C, G = parameter_regression(U, S, method='Lasso', alpha=alpha_lasso[j])
            lasso_jac.append(B_lasso)
            y_lasso[j] = analysis.coeff_dist(B_lasso)


        method_sens[types[i]] = [B_lin, ridge_jac, lasso_jac]
    adata.uns['method_sens'] = [alpha_ridge, alpha_lasso, method_sens]
    adata.uns['sens_coeff'] = [x, y_reg, y_ridge, y_lasso]




def test_sub_sampling(adata,
                      cluster,
                      frac,
                      nsim=10
                      ):
    '''Test the inference of gene-gene interaction matrix with subsampling

    Parameters
    ----------
    adata:
        anndata object of mRNA counts
    cluster:
        cluster selected for inference
    frac:
        fraction of cells to randomly select
    nsim:
        number of independent simulations

    Returns
    -------
    sign_frac
    dist
    weight_sign

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
        sign_frac[i] = analysis.count_sign_change(B, B_ref)
        dist[i] = analysis.mat_distance(B, B_ref)
        weight_sign[i] = analysis.count_weight_sign(B, B_ref)
    return sign_frac, dist, weight_sign


def sampling_sens(adata,
                  frac = np.arange(0.1, 0.91, 0.1),
                  seed=100,
                  nsim=10
                  ):
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

