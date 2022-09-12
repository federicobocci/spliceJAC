'''
plotting resources for sensitivity analysis
'''
import numpy as np
import matplotlib.pyplot as plt

from . import plotting_util

def regression_sens(adata,
                   font_size=10,
                   legend_font=10,
                   title_font=12,
                   legend_loc='best',
                   showfig=None,
                   savefig=None,
                   format='pdf',
                   figsize=(12,4)
                   ):
    '''Plot the summary of spliceJAC inference as a function of regression methods and parameters

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    font_size: `int` (default: 10)
        fontsize of axes and legend labels
    legend_font: `int` (default=12)
        font size of legend
    title_font: `int` (default=12)
        font size of title
    legend_loc: `str` (default='best')
        legend location
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format
    figsize: `tuple` (default: (12,4))
        size of figure

    Returns
    -------
    None

    '''
    [alpha_ridge, alpha_lasso, method_sens] = adata.uns['method_sens']
    x, y_reg, y_ridge, y_lasso = adata.uns['sens_coeff']

    fig = plt.figure(figsize=figsize)

    ax1 = plt.subplot(131)
    ax1.set_xscale('log')
    plt.plot(x, y_reg, '-')
    ax1.set_xlim([x[0], x[-1]])
    ax1.set_xlabel('$\\epsilon$', fontsize=font_size)
    ax1.set_ylabel('Fraction of normalized coefficients > $\\epsilon$', fontsize=font_size)
    ax1.set_title('Linear regression', fontsize=title_font)

    ax2 = plt.subplot(132)
    ax2.set_xscale('log')
    for i in range(alpha_ridge.size):
        plt.plot(x, y_ridge[i], '-', label='$\\alpha_{Ridge}$=' + str(alpha_ridge[i]))
    ax2.set_xlim([x[0], x[-1]])
    ax2.set_xlabel('$\\epsilon$', fontsize=font_size)
    ax2.legend(loc=legend_loc, fontsize=legend_font)
    ax2.set_ylabel('Fraction of normalized coefficients > $\\epsilon$', fontsize=font_size)
    ax2.set_title('Ridge regression', fontsize=title_font)

    ax3 = plt.subplot(133)
    ax3.set_xscale('log')
    for i in range(alpha_lasso.size):
        plt.plot(x, y_lasso[i], '-', label='$\\alpha_{Lasso}$=' + str(alpha_lasso[i]))
    ax3.set_xlim([x[0], x[-1]])
    ax3.set_xlabel('$\\epsilon$', fontsize=font_size)
    ax3.legend(loc=legend_loc, fontsize=legend_font)
    ax3.set_ylabel('Fraction of normalized coefficients > $\\epsilon$', fontsize=font_size)
    ax3.set_title('Lasso regression', fontsize=title_font)

    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format)


def sampling_sens(adata,
                  font_size=10,
                  legend_font=10,
                  legend_loc='best',
                  showfig=None,
                  savefig=None,
                  format='pdf',
                  figsize=(12,4)
                  ):
    '''Plot the summary of spliceJAC inference sensitivity to subsampling of cells in each cell state

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    font_size: `int` (default: 10)
        fontsize of axes and legend labels
    legend_font: `int` (default=10)
        font size of legend
    legend_loc: `str` (default='best')
        legend location
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format
    figsize: `tuple` (default: (12,4))
        size of figure

    Returns
    -------
    None

    '''
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    sens_dict = adata.uns['sens_summary']

    for k in sens_dict.keys():
        ax1.plot(sens_dict[k]['frac'], sens_dict[k]['sign_frac'], 'o-', label=k)
        ax2.plot(sens_dict[k]['frac'], sens_dict[k]['dist'], 'o-', label=k)
        ax3.plot(sens_dict[k]['frac'], sens_dict[k]['weighted_sign'], 'o-', label=k)

    ax1.set_xlabel('Subsampling size (cell fraction)', fontsize=font_size)
    ax2.set_xlabel('Subsampling size (cell fraction)', fontsize=font_size)
    ax3.set_xlabel('Subsampling size (cell fraction)', fontsize=font_size)
    ax1.set_ylabel('Fraction of wrong signs', fontsize=font_size)
    ax2.set_ylabel('Matrix distance', fontsize=font_size)
    ax3.set_ylabel('Weighted sign distance', fontsize=font_size)
    ax1.legend(loc=legend_loc, fontsize=legend_font)
    ax2.legend(loc=legend_loc, fontsize=legend_font)
    ax3.legend(loc=legend_loc, fontsize=legend_font)

    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format)


def subsample_stability(adata,
                        font_size=10,
                        dist_color='firebrick',
                        sign_color='seagreen',
                        eig_color='mediumturquoise',
                        showfig=None,
                        savefig=None,
                        format='pdf',
                        figsize=(12,4)
                        ):
    '''Summary plot of the robustness of spliceJAC inference over multiple inferences with a fraction of the total
    number of cells in each cell state

    Details on accepted colors for pyplot boxplot can be found at:
    https://matplotlib.org/stable/gallery/color/named_colors.html

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    font_size: `int` (default: 10)
        fontsize of axes and legend labels
    dist_color: `str` (default: 'firebrick')
        face color for jacobian distance boxplot
    sign_color: `str` (default: 'seagreen')
        face color for wrong sign boxplot
    eig_color: `str` (default: 'mediumturquoise')
        face color for positive eigenvalues boxplot
    showfig: `Bool` or `None` (default: `None`)
        if True, show the figure
    savefig: `Bool` or `None` (default: `None`)
         if True, save the figure using the savefig path
    format: `str` (default: 'pdf')
        figure format
    figsize: `tuple` (default: (12,4))
        size of figure

    Returns
    -------
    None

    '''
    types = sorted(list(set(list(adata.obs['clusters']))))

    dev_list = [[] for i in range(len(types))]
    wrong_list = [[] for i in range(len(types))]
    pos_eig_list = [[] for i in range(len(types))]

    for i in range(len(types)):
        dev = plotting_util.jac_regr_cons(adata, types[i])
        dev_list[i] = dev
        wrong = plotting_util.count_wrong_signs(adata, types[i])
        wrong_list[i] = wrong
        pos_eig = plotting_util.count_pos_eig(adata, types[i])
        pos_eig_list[i] = pos_eig

    fig = plt.figure(figsize=figsize)

    ax1 = plt.subplot(131)
    bpt = plt.boxplot(dev_list, patch_artist=True)
    for patch in bpt['boxes']:
        patch.set_facecolor(dist_color)
    plt.xticks(np.arange(1, len(dev_list) + 1, 1), types, rotation=45)
    plt.ylabel('Jacobian distance', fontsize=font_size)

    ax2 = plt.subplot(132)
    bpt = plt.boxplot(wrong_list, patch_artist=True)
    for patch in bpt['boxes']:
        patch.set_facecolor(sign_color)
    plt.xticks(np.arange(1, len(wrong_list) + 1, 1), types, rotation=45)
    plt.ylabel('Wrong signs (%)', fontsize=font_size)

    ax3 = plt.subplot(133)
    bpt = plt.boxplot(pos_eig_list, patch_artist=True)
    for patch in bpt['boxes']:
        patch.set_facecolor(eig_color)
    plt.xticks(np.arange(1, len(pos_eig_list) + 1, 1), types, rotation=45)
    plt.ylabel('Positive eigenvalues (%)', fontsize=font_size)

    plt.tight_layout()

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(savefig, format=format)
