import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scvelo as scv
import scanpy as sc
import spliceJAC as sp

def get_pos_frac(eig_list):
    nsim, nval = len(eig_list), eig_list[0].size
    pos_frac = []
    for i in range(nsim):
        w = np.real(eig_list[i])
        pos_frac.append(w[w > 0].size)
    return pos_frac

panc_data = sc.read_h5ad('datasets/Pancreas/pancreas.h5ad')
types = sorted(list(set(list(panc_data.obs['clusters']))))

sp.tl.inference.estimate_jacobian(panc_data, n_top_genes=50)


jacobians = panc_data.uns['jacobian_lists']
eig = []
for t in types:
    jac_list, w_list, v_list = jacobians[t]
    pos_frac = get_pos_frac(w_list)
    eig.append(pos_frac)
np.savetxt('/Users/federicobocci/Desktop/stability_study/pancreas_eig.txt', np.asarray(eig))


# mix states together
nsim = 50
ind = np.arange(0, len(types), 1)

clusters = list(panc_data.obs['clusters'])

np.random.seed(100)
random_eig = []
for i in range(nsim):
    print(i)
    new_clst = ['' for k in range(len(clusters))]
    np.random.shuffle(ind)
    shuff = ind
    for j in range(len(clusters)):
        if clusters[j]==types[shuff[0]] or clusters[j]==types[shuff[1]]:
            new_clst[j] = 'Group 1'
        elif clusters[j]==types[shuff[2]] or clusters[j]==types[shuff[3]]:
            new_clst[j] = 'Group 2'
        elif clusters[j]==types[shuff[4]] or clusters[j]==types[shuff[5]]:
            new_clst[j] = 'Group 3'
        else:
            new_clst[j] = 'Group 4'

    new_adata = panc_data.copy()
    new_adata.obs['clusters'] = new_clst
    sp.tl.inference.estimate_jacobian(new_adata, n_top_genes=50, filter_and_norm=False)
    new_types = sorted(list(set(list(new_adata.obs['clusters']))))

    jacobians = new_adata.uns['jacobian_lists']
    for t in new_types:
        jac_list, w_list, v_list = jacobians[t]
        pos_frac = get_pos_frac(w_list)
        random_eig.append(pos_frac)

np.savetxt('/Users/federicobocci/Desktop/stability_study/random_eig.txt', np.asarray(random_eig))



