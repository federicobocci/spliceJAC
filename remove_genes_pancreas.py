import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import scvelo as scv
import scanpy as sc
import spliceJAC as sp

def load_pancreas_data():
    panc_data = sc.read_h5ad('datasets/Pancreas/pancreas.h5ad')

    scv.pp.filter_genes(panc_data, min_shared_counts=20)
    scv.pp.normalize_per_cell(panc_data)
    scv.pp.filter_genes_dispersion(panc_data, n_top_genes=2000)
    scv.pp.log1p(panc_data)

    scv.tl.velocity(panc_data)
    scv.tl.velocity_graph(panc_data)

    panc_data.uns['neighbors']['distances'] = panc_data.obsp['distances']
    panc_data.uns['neighbors']['connectivities'] = panc_data.obsp['connectivities']

    scv.tl.paga(panc_data, groups='clusters')
    panc_data.uns['PAGA_paths'] = scv.get_df(panc_data, 'paga/transitions_confidence').T

    return panc_data

def write_grn_to_file(grn, file):
    n = grn.shape[0]
    for i in range(n):
        for j in range(n):
            file.write( str(grn[i][j]) + '\t' )
        file.write('\n')


def run_control(adata):
    types = sorted(list(set(list(adata.obs['clusters']))))

    sp.tl.inference.estimate_jacobian(adata, n_top_genes=50)
    sp.tl.analysis.grn_statistics(adata)
    sp.tl.analysis.signaling_score(adata)
    sc.tl.rank_genes_groups(adata, 'clusters', method='t-test')
    sp.tl.analysis.trans_from_PAGA(adata)

    # save results

    # GRNs
    f = open('/Users/federicobocci/Desktop/benchmarking/remove_genes_pancreas/control_grn.txt', 'w')

    for t in types:
        J = adata.uns['average_jac'][t][0]
        m = int(J.shape[0] / 2)
        grn = J[0:m, m:]
        write_grn_to_file(grn, f)
    f.close()

    # transition scores
    transitions = adata.uns['transitions'].keys()
    tr_dict = {'genes': list(adata.var_names)}

    for tr in transitions:
        tr_dict[tr] = adata.uns['transitions'][tr]['weights']
    tr_df = pd.DataFrame.from_dict(tr_dict)
    tr_df.to_csv('/Users/federicobocci/Desktop/benchmarking/remove_genes_pancreas/control_tg.csv')



def run_remove(adata, n_remove=5, nsim=10):
    types = sorted(list(set(list(adata.obs['clusters']))))

    # select only the top 50 genes , so the random 5 genes are all meaningful genes
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=50)

    genes = np.asarray(adata.var_names)

    for i in range(nsim):
        print('sim ' + str(i + 1))

        # get partial anndata without 5 random genes
        samp = random.sample(range(50), n_remove)
        sel_genes = [g for g in genes if g not in genes[samp] ]
        sel_adata = adata[:, sel_genes].copy()

        sp.tl.inference.estimate_jacobian(sel_adata, n_top_genes=45)
        sp.tl.analysis.grn_statistics(sel_adata)
        sp.tl.analysis.signaling_score(sel_adata)
        sc.tl.rank_genes_groups(sel_adata, 'clusters', method='t-test')
        sp.tl.analysis.trans_from_PAGA(sel_adata)

        # save results
        f = open('/Users/federicobocci/Desktop/benchmarking/remove_genes_pancreas/sim_' + str(i + 1) + '_grn.txt',
                 'w')

        for t in types:
            J = sel_adata.uns['average_jac'][t][0]
            m = int(J.shape[0] / 2)
            grn = J[0:m, m:]
            write_grn_to_file(grn, f)
        f.close()

        transitions = sel_adata.uns['transitions'].keys()
        tr_dict = {'genes': list(sel_adata.var_names)}

        for tr in transitions:
            tr_dict[tr] = sel_adata.uns['transitions'][tr]['weights']
        tr_df = pd.DataFrame.from_dict(tr_dict)
        tr_df.to_csv(
            '/Users/federicobocci/Desktop/benchmarking/remove_genes_pancreas/sim_' + str(i + 1) + '_tg.csv')



def main():
    panc_data = load_pancreas_data()
    # geneset = run_control(panc_data)
    run_remove(panc_data)


if __name__=='__main__':
    main()