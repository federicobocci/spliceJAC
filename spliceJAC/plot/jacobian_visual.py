'''
plotting resources for Jacobian and eigenvalues visualization
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import copy

from . import plotting_util


def visualize_jacobian(adata, panel_height=3, panel_length=3.5, pan_per_row=4,
                       showfig=False, savefig=True, figname='jacobian.pdf', format='pdf'):

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
        pt = ax.pcolor(x, x, J, cmap='RdBu_r', vmin=-lim, vmax=lim)
        cbar = plt.colorbar(pt, label='coefficient')
        plt.title(clusters[i])
        plt.xlabel('Regulator gene')
        plt.ylabel('Target gene')

        # matrix_heatmap(ax, inf_dict[clusters[i]][0], title=clusters[i])

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)


def matrix_heatmap(ax, mat, title=False):
    n = int(mat.shape[0]/2)
    J = mat[0:n, n:]
    lim = np.amax(np.abs(J))
    x = np.arange(0, J.shape[0]+1, 1)
    pt = ax.pcolor(x, x, J, cmap='RdBu', vmin=-lim, vmax=lim)
    plt.colorbar(pt, cax=ax, label='coefficient')
    if title:
        plt.title(title)



def eigen_spectrum(adata, panel_height=4, panel_length=4, pan_per_row=4,
                   showfig=False, savefig=True, figname='eigenvalues.pdf', format='pdf'):
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
        spectrum_plot(ax, np.sort(np.real(inf_dict[clusters[i]][1])), title=clusters[i])

    plt.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname, format=format, dpi=300)


def spectrum_plot(ax, v, title=False, show_frac=True, loc='lower right', show_zero=True):
    if show_frac:
        frac = round(100*float(v[v>0].size)/v.size, 2)
        ax.plot(np.arange(1, v.size + 1, 1), v, 'ko', label='$\\rho=$'+str(frac)+'%')
        plt.legend(loc=loc)
    else:
        ax.plot(np.arange(1, v.size + 1, 1), v, 'ko')
    if show_zero:
        ax.plot([1, v.size], [0, 0], 'r--')
    plt.xlabel('Eigenvalue rank')
    plt.ylabel('Eigenvalue')
    plt.xlim([1, v.size])  # change to [0, v.size+1]?
    if title:
        plt.title(title)


def genes_bar_plot(ax, v, title=False):
    plt.bar(np.arange(0, v.size, 1), v, align='center', width=0.8)
    # plt.xticks(np.arange(0, v.size, 1), lab, rotation=45)
    plt.ylabel('Instability score')
    if title:
        plt.title(title)






def plot_gene_stability(adata, top_genes=10, panel_height=4, panel_length=4, pan_per_row=4):
    types = sorted(list(set(list(adata.obs['clusters']))))
    axes = adata.uns['axes']
    n = len(types)
    m = len(axes)

    nrow = int(n / pan_per_row) + 1 if n % pan_per_row > 0 else int(n / pan_per_row)
    ncol = max(n % pan_per_row, pan_per_row)

    fig = plt.figure(figsize=(panel_length * ncol, panel_height * nrow))
    colors = list(plt.cm.Set3.colors)[0:top_genes]


    for i in range(n):

        # score = adata.uns['stability'][types[i]]['inst_score']
        score = adata.uns['inst_scores'][types[i]]
        avg = np.mean(score, axis=0)
        ind = np.argsort(avg)
        data, names = [], []
        for j in range(top_genes):
            names.append(axes[ind[m - top_genes + j]])
            data.append(score[:, ind[m - top_genes + j]])

        # panel coordinates for plotting
        j = int(i / pan_per_row)
        k = i % pan_per_row

        ax = plt.subplot2grid((nrow, ncol), (j, k), rowspan=1, colspan=1)
        bpt = plt.boxplot(data, vert=False, positions=np.arange(1, len(data) + 1, 1), patch_artist=True)
        for patch, color in zip(bpt['boxes'], colors):
            patch.set_facecolor(color)

        plt.yticks(np.arange(1, len(data) + 1, 1), names)
        plt.xlabel('Instability score')
        plt.title(types[i])

    plt.tight_layout()
    plt.show()


def plot_pos_eig(adata, panel_height=4, panel_length=4, pan_per_row=4):
    types = sorted(list(set(list(adata.obs['clusters']))))
    n = len(types)
    m = 2*len(list(adata.var_names))

    nrow = int(n / pan_per_row) + 1 if n % pan_per_row > 0 else int(n / pan_per_row)
    ncol = max(n % pan_per_row, pan_per_row)

    fig = plt.figure(figsize=(panel_length * ncol, panel_height * nrow))

    for i in range(n):

        w_list = adata.uns['jacobian_lists'][types[i]][1]
        nsim = len(w_list)
        pos_eig = np.zeros(nsim)

        for j in range(nsim):
            w = w_list[j]
            pos_eig[j] = 100*np.real(w)[np.real(w)>0].size/float(m)

        # panel coordinates for plotting
        j = int(i / pan_per_row)
        k = i % pan_per_row

        ax = plt.subplot2grid((nrow, ncol), (j, k), rowspan=1, colspan=1)
        x = np.arange(1, pos_eig.size+1, 1)
        plt.bar(x, pos_eig)
        plt.xlabel('Simulation ID')
        plt.ylabel('Positive eigenvalues (%)')
        plt.title(types[i])

    plt.tight_layout()
    plt.show()


def plot_cluster_inst(adata):
    types = sorted(list(set(list(adata.obs['clusters']))))
    score_list = [[] for i in range(len(types))]

    for i in range(len(types)):
        score = plotting_util.cluster_inst_score(adata, types[i])
        score_list[i] = score

    fig = plt.figure(figsize=(4, 4))

    ax1 = plt.subplot(111)
    bpt = plt.boxplot(score_list, patch_artist=True)
    for patch in bpt['boxes']:
        patch.set_facecolor('firebrick')
    plt.xticks(np.arange(1, len(score_list) + 1, 1), types, rotation=45)
    plt.ylabel('Instability score')

    plt.tight_layout()
    plt.show()
