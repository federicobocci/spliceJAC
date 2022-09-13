'''
test the sensitivity to the regression method and parameters
'''

import numpy as np

from .aux_functions import parameter_regression

def coeff_dist(G,
               x = np.logspace(-5, 0, num=100)
               ):
    '''Construct the coefficient distribution for the gene-gene interaction matrix G

    Parameters
    ----------
    G: `~numpy.ndarray`
        interaction matrix
    x: `~numpy.ndarray` (default: numpy.logspace(-5, 0, num=100))
        vector of thresholds to group matrix coefficients

    Returns
    -------
    y: `~numpy.ndarray`
        The distribution of coefficients

    '''
    pars = np.abs(np.ndarray.flatten(G))
    coeffs = pars/np.amax(pars)
    y = np.zeros(x.size)
    for i in range(x.size):
        y[i] = coeffs[coeffs > x[i]].size
    y = y/pars.size
    return y


def regr_method_sens(adata,
                     alpha_ridge = np.array([0.01, 0.1, 1, 10, 100]),
                     alpha_lasso = np.array([0.0001, 0.001, 0.01, 0.1, 1])
                     ):
    '''Compare methods for gene-gene interaction parameter regression
    Results are stored in adata.uns['method_sens'] and adata.uns['sens_coeff']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    alpha_ridge: `~numpy.ndarray` (default: numpy.array([0.01, 0.1, 1, 10, 100]))
        array of shrinkage coefficients to test for Ridge regression
    alpha_lasso: `~numpy.ndarray` (default: numpy.array([0.0001, 0.001, 0.01, 0.1, 1]))
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
        y_reg = coeff_dist(B_lin, x=x)

        # Ridge regression
        for j in range(alpha_ridge.size):
            B_ridge, C, G = parameter_regression(U, S, alpha=alpha_ridge[j])
            ridge_jac.append(B_ridge)
            y_ridge[j] = coeff_dist(B_ridge)

        # Lasso regression
        for j in range(alpha_lasso.size):
            B_lasso, C, G = parameter_regression(U, S, method='Lasso', alpha=alpha_lasso[j])
            lasso_jac.append(B_lasso)
            y_lasso[j] = coeff_dist(B_lasso)

        method_sens[types[i]] = [B_lin, ridge_jac, lasso_jac]
    adata.uns['method_sens'] = [alpha_ridge, alpha_lasso, method_sens]
    adata.uns['sens_coeff'] = [x, y_reg, y_ridge, y_lasso]