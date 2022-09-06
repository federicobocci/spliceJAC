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
                       showfig=False,
                       savefig=False,
                       figname='jacobian.pdf',
                       format='pdf',
                       dpi=300
                       ):
    '''
    Plot the inferred gene-gene interaction matrices of each cell state

    Parameters
    ----------
    adata: anndata object
    panel_height: height of each panel (in inches) (default=3)
    panel_length: length of each panel (in inches) (default=3.5)
    pan_per_row: number of panels per row (default=4)
    fontsize: fontsize for labels (default=10)
    cmap: colormap for Jacobian visualization, any pyplot colormap name can be provided (default='RdBu_r')
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=False)
    figname: name of figure file to save (default='jacobian.pdf')
    format: format of figure file to save (default='pdf')
    dpi: dpi of saved figure (default=300)

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
        plt.savefig(figname, format=format, dpi=dpi)


def eigen_spectrum(adata,
                   panel_height=4,
                   panel_length=4,
                   pan_per_row=4,
                   fontsize=10,
                   show_frac=True,
                   loc='lower right',
                   show_zero=True,
                   showfig=False,
                   savefig=False,
                   figname='eigenvalues.pdf',
                   format='pdf',
                   dpi=300
                   ):
    '''
    Plot the eigenvalues of each cell state

    Parameters
    ----------
    adata: anndata object
    panel_height: height of each panel (in inches) (default=3)
    panel_length: length of each panel (in inches) (default=3.5)
    pan_per_row: number of panels per row (default=4)
    fontsize: fontsize for labels (default=10)
    show_frac: show legend with fraction of positive eigenvalues (default=True)
    loc: location of legend (default='lower right')
    show_zero: plot horizontal line to highlight change of sign (default=True)
    showfig: if True, show the figure (default=False)
    savefig: if True, save the figure (default=False)
    figname: name of figure file to save (default='eigenvalues.pdf')
    format: format of figure file to save (default='pdf')
    dpi: dpi of saved figure (default=300)

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
        plt.savefig(figname, format=format, dpi=dpi)


def spectrum_plot(ax,
                  v,
                  title,
                  show_frac=True,
                  loc='lower right',
                  show_zero=True,
                  fontsize=10
                  ):
    '''
    Plot the eigenspectrum of a cell state on a matplotlib axis

    Parameters
    ----------
    ax: axis objext
    v: ranked real components of eigenvalues
    title: title of plot
    show_frac: show legend with fraction of positive eigenvalues (default=True)
    loc: location of legend (default='lower right')
    show_zero: plot horizontal line to highlight change of sign (default=True)
    fontsize: fontsize for labels (default=10)

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



##########################################
##########################################
##########################################
##########################################
##########################################
#
#
#
# # needed?
# def diff_matrix(adata, cluster1, cluster2, genes=None, fontsize=10,
#                 figsize=(5,4), showfig=False, savefig=True, figname='diff_matrix.pdf', format='pdf'):
#     if genes == None:
#         genes = list(adata.var_names)
#     n = len(genes)
#
#     A1 = adata.uns['average_jac'][cluster1][0][0:n, n:].copy()
#     A2 = adata.uns['average_jac'][cluster2][0][0:n, n:].copy()
#     J = A2 - A1
#
#     fig = plt.figure(figsize=figsize)
#
#     ax = plt.subplot(111)
#     lim = np.amax(np.abs(J))
#     x = np.arange(0, J.shape[0] + 1, 1)
#     pt = ax.pcolor(x, x, J, cmap='RdBu_r', vmin=-lim, vmax=lim)
#     cbar = plt.colorbar(pt, label='Change in coefficient')
#     plt.title(cluster1 + '-' + cluster2 + ' differential GRN')
#     plt.xlabel('Regulator gene', fontsize=fontsize)
#     plt.ylabel('Target gene', fontsize=fontsize)
#
#     plt.tight_layout()
#     if showfig:
#         plt.show()
#     if savefig:
#         plt.savefig(figname, format=format, dpi=300)
#
#
# # needed?
# def genes_bar_plot(ax, v, title=False):
#     plt.bar(np.arange(0, v.size, 1), v, align='center', width=0.8)
#     # plt.xticks(np.arange(0, v.size, 1), lab, rotation=45)
#     plt.ylabel('Instability score')
#     if title:
#         plt.title(title)
#
#
#
# # needed?
# def plot_gene_stability(adata, top_genes=10, panel_height=4, panel_length=4, pan_per_row=4):
#     types = sorted(list(set(list(adata.obs['clusters']))))
#     axes = adata.uns['axes']
#     n = len(types)
#     m = len(axes)
#
#     nrow = int(n / pan_per_row) + 1 if n % pan_per_row > 0 else int(n / pan_per_row)
#     ncol = max(n % pan_per_row, pan_per_row)
#
#     fig = plt.figure(figsize=(panel_length * ncol, panel_height * nrow))
#     colors = list(plt.cm.Set3.colors)[0:top_genes]
#
#
#     for i in range(n):
#
#         # score = adata.uns['stability'][types[i]]['inst_score']
#         score = adata.uns['inst_scores'][types[i]]
#         avg = np.mean(score, axis=0)
#         ind = np.argsort(avg)
#         data, names = [], []
#         for j in range(top_genes):
#             names.append(axes[ind[m - top_genes + j]])
#             data.append(score[:, ind[m - top_genes + j]])
#
#         # panel coordinates for plotting
#         j = int(i / pan_per_row)
#         k = i % pan_per_row
#
#         ax = plt.subplot2grid((nrow, ncol), (j, k), rowspan=1, colspan=1)
#         bpt = plt.boxplot(data, vert=False, positions=np.arange(1, len(data) + 1, 1), patch_artist=True)
#         for patch, color in zip(bpt['boxes'], colors):
#             patch.set_facecolor(color)
#
#         plt.yticks(np.arange(1, len(data) + 1, 1), names)
#         plt.xlabel('Instability score')
#         plt.title(types[i])
#
#     plt.tight_layout()
#     plt.show()
#
# # needed?
# def plot_pos_eig(adata, panel_height=4, panel_length=4, pan_per_row=4):
#     types = sorted(list(set(list(adata.obs['clusters']))))
#     n = len(types)
#     m = 2*len(list(adata.var_names))
#
#     nrow = int(n / pan_per_row) + 1 if n % pan_per_row > 0 else int(n / pan_per_row)
#     ncol = max(n % pan_per_row, pan_per_row)
#
#     fig = plt.figure(figsize=(panel_length * ncol, panel_height * nrow))
#
#     for i in range(n):
#
#         w_list = adata.uns['jacobian_lists'][types[i]][1]
#         nsim = len(w_list)
#         pos_eig = np.zeros(nsim)
#
#         for j in range(nsim):
#             w = w_list[j]
#             pos_eig[j] = 100*np.real(w)[np.real(w)>0].size/float(m)
#
#         # panel coordinates for plotting
#         j = int(i / pan_per_row)
#         k = i % pan_per_row
#
#         ax = plt.subplot2grid((nrow, ncol), (j, k), rowspan=1, colspan=1)
#         x = np.arange(1, pos_eig.size+1, 1)
#         plt.bar(x, pos_eig)
#         plt.xlabel('Simulation ID')
#         plt.ylabel('Positive eigenvalues (%)')
#         plt.title(types[i])
#
#     plt.tight_layout()
#     plt.show()
#
# # needed?
# def plot_cluster_inst(adata):
#     types = sorted(list(set(list(adata.obs['clusters']))))
#     score_list = [[] for i in range(len(types))]
#
#     for i in range(len(types)):
#         score = plotting_util.cluster_inst_score(adata, types[i])
#         score_list[i] = score
#
#     fig = plt.figure(figsize=(4, 4))
#
#     ax1 = plt.subplot(111)
#     bpt = plt.boxplot(score_list, patch_artist=True)
#     for patch in bpt['boxes']:
#         patch.set_facecolor('firebrick')
#     plt.xticks(np.arange(1, len(score_list) + 1, 1), types, rotation=45)
#     plt.ylabel('Instability score')
#
#     plt.tight_layout()
#     plt.show()
