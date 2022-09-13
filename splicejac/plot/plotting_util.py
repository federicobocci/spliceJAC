'''
plotting utilities
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_setup(adata,
               cmap=plt.cm.Set2.colors
               ):
    '''Assign a color to each cluster from a colormap
    Results are stored in adata.uns['colors']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cmap: `pyplot colormap` (default: plt.cm.Set2.colors)
        Colormap for cell state. To use another colormap, provide argument following the same syntax:
        plt.cm. + chosen_colormap + .colors. A list of accepted colormaps can be found at:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Returns
    -------
    None

    '''
    clusters = list(adata.obs['clusters'])
    types = sorted(list(set(clusters)))
    colors = list(cmap)[0:len(types)]
    adata.uns['colors'] = {types[i]: colors[i] for i in range(len(types))}

def jac_regr_cons(adata,
                  cluster
                  ):
    '''Check the consistency of Jacobian regression by computing element-wise distance between Jacobians from different
    simulations for the same cell state

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster: `str`
        cell state

    Returns
    -------
    dev: a list of element-wise difference between all pairs of Jacobians

    '''
    assert 'jacobian_lists' in adata.uns.keys(), 'Run cluster stability first'

    jac_list = adata.uns['jacobian_lists'][cluster][0]
    nsim = len(jac_list)

    dev = []
    for i in range(nsim):
        for j in range( i ):
            J1, J2 = jac_list[i], jac_list[j]
            dev.append( np.linalg.norm(np.abs(J1 - J2)) )
    return dev


def compare_wrong_signs(j1,
                        j2,
                        eps=0.9
                        ):
    '''
    Count fraction of wrong signs between two matrices

    Parameters
    ----------
    j1: `~numpy.ndarray`
        matrix 1
    j2: `~numpy.ndarray`
        matrix 2
    eps: `float` (default: 0.9)
        cutoff quantile to select only the largest elements (in absolute value) in the two matrices in the range [0,1]

    Returns
    -------
    perc_sign: percentage of wrong signs between the two matrices

    '''
    n = j1.shape[0]
    j = np.sort(np.abs(np.ndarray.flatten(np.concatenate((j1, j2)))))
    t = j[int(eps*j.size)]

    count = 0
    for i in range(n):
        for j in range(n):
            if np.abs(j1[i][j] ) >t or np.abs(j2[i][j] ) >t:
                if np.sign(j1[i][j] )!=np.sign(j2[i][j]):
                    count = count + 1
    perc_sign = 100*count/float(( 1 -eps ) * n *n)
    return perc_sign


def count_wrong_signs(adata,
                      cluster,
                      eps=0.9
                      ):
    '''Quantify the fraction of wring signs between Jacobians of different simulations for the same cell state

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster: `str`
        cell state
    eps: `float` (default: 0.9)
        cutoff quantile to select only the largest elements (in absolute value) in the two matrices in the range [0,1]

    Returns
    -------
    wrong_sign: a list of wrong sign percentages between all pairs of Jacobians

    '''
    assert 'jacobian_lists' in adata.uns.keys(), 'Run cluster stability first'

    jac_list = adata.uns['jacobian_lists'][cluster][0]
    nsim = len(jac_list)

    wrong_sign = []
    for i in range(nsim):
        for j in range(i):
            J1, J2 = jac_list[i], jac_list[j]
            wrong_sign.append( compare_wrong_signs(J1, J2, eps=eps) )
    return wrong_sign


def count_pos_eig(adata,
                  cluster
                  ):
    '''
    Count the number of positive eigenvalues of the Jacobians of different simulations for the same cell state

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster: `str`
        cell state

    Returns
    -------
    pos_eig: list with number of positive eigenvalues for each Jacobian matrix

    '''
    assert 'jacobian_lists' in adata.uns.keys(), 'Run cluster stability first'

    w_list = adata.uns['jacobian_lists'][cluster][1]
    nsim = len(w_list)
    m = 2 * len(list(adata.var_names))

    pos_eig = []
    for i in range(nsim):
        w = w_list[i]
        pos_eig.append( 100 *np.real(w)[np.real(w) >0].size/float(m) )
    return pos_eig
