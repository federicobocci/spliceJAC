'''
plotting resources for Jacobian and eigenvalues visualization
'''
import numpy as np
import matplotlib.pyplot as plt

def visualize_jacobian(adata,
                       panel_height=3,
                       panel_length=3.5,
                       pan_per_row=4,
                       fontsize=10,
                       cmap='RdBu_r',
                       showfig=None,
                       savefig=None,
                       format='pdf'
                       ):
    '''Plot the inferred gene-gene interaction matrices of each cell state

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    panel_height: `float` (default: 3)
        height of each panel (in inches)
    panel_length: `float` (default: 3.5)
        length of each panel (in inches)
    pan_per_row: `int` (default: 4)
        number of panels per row
    fontsize: `int` (default: 10)
        fontsize for labels
    cmap: `str` (default: 'RdBu_r')
        colormap for Jacobian visualization. Accepted colormaps ca be found at:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format

    Returns
    -------
    None

    '''
    inf_dict = adata.uns['average_jac']
    n = len(inf_dict.keys())
    clusters = sorted(list(inf_dict.keys()))

    nrow = int(n / pan_per_row) + 1 if n % pan_per_row > 0 else int(n / pan_per_row)
    ncol = max(n % pan_per_row, pan_per_row)

    fig = plt.figure(figsize=(panel_length * ncol, panel_height * nrow))

    for i in range(n):
        # panel coordinates for plotting
        j = int(i / pan_per_row)
        k = i % pan_per_row

        ax = plt.subplot2grid((nrow, ncol), (j, k), rowspan=1, colspan=1)

        mat = inf_dict[clusters[i]][0]
        m = int(mat.shape[0] / 2)
        J = mat[0:m, m:]
        lim = np.amax(np.abs(J))
        x = np.arange(0, J.shape[0] + 1, 1)
        pt = ax.pcolor(x, x, J, cmap=cmap, vmin=-lim, vmax=lim)
        cbar = plt.colorbar(pt, label='coefficient')
        plt.title(clusters[i])
        plt.xlabel('Regulator gene', fontsize=fontsize)
        plt.ylabel('Target gene', fontsize=fontsize)

        # matrix_heatmap(ax, inf_dict[clusters[i]][0], title=clusters[i])

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format)


def eigen_spectrum(adata,
                   panel_height=4,
                   panel_length=4,
                   pan_per_row=4,
                   fontsize=10,
                   show_frac=True,
                   loc='lower right',
                   show_zero=True,
                   showfig=None,
                   savefig=None,
                   format='pdf'
                   ):
    '''Plot the eigenvalues of each cell state

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    panel_height: `float` (default: 3)
        height of each panel (in inches)
    panel_length: `float` (default: 3.5)
        length of each panel (in inches)
    pan_per_row: `int` (default: 4)
        number of panels per row
    fontsize: `int` (default: 10)
        fontsize for labels
    show_frac: `Bool` (default: True)
        show legend with fraction of positive eigenvalues
    loc: `str` (default: 'lower right')
        location of legend. Accepted legend positions can be found at:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    show_zero: `Bool` (default: True)
        plot horizontal line to highlight change of sign
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format

    Returns
    -------
    None

    '''
    inf_dict = adata.uns['average_jac']
    n = len(inf_dict.keys())
    clusters = sorted(list(inf_dict.keys()))

    nrow = int(n / pan_per_row) + 1 if n % pan_per_row > 0 else int(n / pan_per_row)
    ncol = max(n % pan_per_row, pan_per_row)

    fig = plt.figure(figsize=(panel_length*ncol, panel_height*nrow))

    for i in range(n):

        # panel coordinates for plotting
        j = int(i / pan_per_row)
        k = i % pan_per_row

        ax = plt.subplot2grid((nrow, ncol), (j, k), rowspan=1, colspan=1)
        spectrum_plot(ax, np.sort(np.real(inf_dict[clusters[i]][1])), clusters[i], show_frac=show_frac, loc=loc, show_zero=show_zero, fontsize=fontsize)

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format)


def spectrum_plot(ax,
                  v,
                  title,
                  show_frac=True,
                  loc='lower right',
                  show_zero=True,
                  fontsize=10
                  ):
    '''Plot the eigenspectrum of a cell state on a matplotlib axis

    Parameters
    ----------
    ax: pyplot axis
        axis to plot
    v: `~numpy.ndarray`
        ranked real components of eigenvalues
    title: `str`
        title of plot
    show_frac: `Bool` (default: True)
        show legend with fraction of positive eigenvalues
    loc: `str` (default: 'lower right')
        location of legend. Accepted legend positions can be found at:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    show_zero: `Bool` (default: True)
        plot horizontal line to highlight change of sign
    fontsize: `int` (default: 10)
        fontsize for labels

    Returns
    -------
    None

    '''
    if show_frac:
        frac = round(100*float(v[v>0].size)/v.size, 2)
        ax.plot(np.arange(1, v.size + 1, 1), v, 'ko', label='$\\rho=$'+str(frac)+'%')
        plt.legend(loc=loc)
    else:
        ax.plot(np.arange(1, v.size + 1, 1), v, 'ko')
    if show_zero:
        ax.plot([1, v.size], [0, 0], 'r--')
    plt.xlabel('Eigenvalue rank', fontsize=fontsize)
    plt.ylabel('Eigenvalue', fontsize=fontsize)
    plt.xlim([1, v.size])
    if title:
        plt.title(title, fontsize=fontsize)