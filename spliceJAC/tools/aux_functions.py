'''
auxiliary functions for spliceJAC inference and analysis
'''

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, Lasso

def parameter_regression(U_data,
                         S_data,
                         method='Ridge',
                         alpha=1,
                         fit_int=True
                         ):
    '''run regression to infer spliced-unspliced interaction coefficients

    Parameters
    ----------
    U_data: n_obs x n_genes count matrix of unspliced counts
    S_data: n_obs x n_genes count matrix of spliced counts
    method: regression method, either Linear, Ridge or Lasso (default=Ridge)
    alpha: regularization coefficient
    fit_int: if True, set the fit_intercept parameter to True (default=True)

    Returns
    -------
    mat: gene-gene interaction matrix
    interc: intercept vector
    degr: degradation coefficient vector

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
    adata: anndata object of mRNA counts
    first_moment: if True, use first moment of U and S to run regression

    Returns
    -------
    degr: degradation coefficient vector

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
    J: Jacobian matrix

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
    adata: anndata object of mRNA counts

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


def instability_scores(adata):
    '''
    Construct an instability score for each gene in each cluster by looking at eigenvector components in the cluster unstable manifold
    results are saved in adata.uns['inst_scores']
    Parameters
    ----------
    adata: anndata object

    Returns
    -------
    None

    '''
    assert 'jacobian_lists' in adata.uns.keys(), "Run regression before computing instability scores"

    types = sorted(list(set(list(adata.obs['clusters']))))
    ngenes = 2 * len(list(adata.var_names))

    inst_scores = {}

    for k in types:
        J_list, w_list, v_list = adata.uns['jacobian_lists'][k]
        nsim = len(J_list)
        ins_score = np.zeros((nsim, ngenes))

        for j in range(nsim):
            w, v = w_list[j], v_list[j]
            score = np.zeros(w.size)
            ind = np.argwhere(np.real(w) > 0)
            for i in ind:
                score = score + np.ndarray.flatten(np.real(v[:, i]))**2
            score = score / len(ind)
            ins_score[j] = score
        inst_scores[k] = ins_score
    adata.uns['inst_scores'] = inst_scores