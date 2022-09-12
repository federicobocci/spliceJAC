'''
evaluate gene statistics of the state-specific GRNs
'''

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import iqr

def signaling_score(adata):
    '''Compute the signaling scores of all genes in each cell states

    Scores are defined based on number of incoming/outgoing edges and weighted sum of incoming/outgoing edges
    Results are saved in  adata.uns['signaling_scores']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix

    Returns
    -------
    None

    '''
    types = sorted(list(set(list(adata.obs['clusters']))))
    n = len(list(adata.var_names))

    jac_df = adata.uns['average_jac']
    sign_score = {}

    for t in types:
        M = jac_df[t][0][0:n, n:].copy()
        rec, sen = np.sum(np.abs(M), axis=1), np.sum(np.abs(M), axis=0)

        rec_edges, sen_edges = np.zeros(M[0].size), np.zeros(M[0].size)
        for i in range(rec_edges.size):
            rec_edges[i] = np.count_nonzero(np.abs(M[i]))
            sen_edges[i] = np.count_nonzero(np.abs(M[:,i]))

        sign_score[t] = {'incoming': rec, 'outgoing': sen, 'incoming_edges': rec_edges, 'outgoing_edges': sen_edges}
    adata.uns['signaling_scores'] = sign_score

def compute_metrics(dat):
    '''
    Compute statistical metrics including standard deviation, range, and interquartile range of a 1D vector

    Parameters
    ----------
    dat: `~numpy.ndarray`
        1D array of measurements

    Returns
    -------
    metrics_df: `~pandas.dataFrame`
        dataframe of metrics

    '''
    SD, rng, int_qrt = np.zeros(dat[0].size), np.zeros(dat[0].size), np.zeros(dat[0].size)

    SD = np.std(dat, axis=0)

    for i in range(rng.size):
        rng[i] = np.amax(dat[:, i]) - np.amin(dat[:, i])

    int_qrt = iqr(dat, axis=0)

    d = {'SD':SD, 'range':rng, 'inter_range':int_qrt}
    metrics_df = pd.DataFrame(d)
    return metrics_df


def GRN_cluster_variation(adata):
    '''Compute state-specific GRN statistics including gene centrality, incoming, outgoing and total signaling strength
    Results are stored in adata.uns['cluster_variation'] and adata.uns['cluster_average']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix

    Returns
    -------
    None

    '''
    types = sorted(list(set(list(adata.obs['clusters']))))
    genes = list(adata.var_names)

    dat_BC, dat_in, dat_out, dat_tot = np.zeros((len(types), len(genes))), np.zeros((len(types), len(genes))), np.zeros((len(types), len(genes))), np.zeros((len(types), len(genes)))

    for i in range(len(types)):
        dat_BC[i] = np.asarray(adata.uns['GRN_statistics'][0][types[i]].mean(axis=1))
        dat_in[i] = np.asarray(adata.uns['GRN_statistics'][1][types[i]].mean(axis=1))
        dat_out[i] = np.asarray(adata.uns['GRN_statistics'][2][types[i]].mean(axis=1))
        dat_tot[i] = np.asarray(adata.uns['GRN_statistics'][3][types[i]].mean(axis=1))

    gene_average = { 'centrality': dat_BC.mean(axis=0), 'incoming': dat_in.mean(axis=0), 'outgoing': dat_out.mean(axis=0), 'signaling': dat_tot.mean(axis=0) }
    adata.uns['cluster_average'] = gene_average

    gene_variation = {}
    gene_variation['centrality'] = compute_metrics(dat_BC)
    gene_variation['incoming'] = compute_metrics(dat_in)
    gene_variation['outgoing'] = compute_metrics(dat_out)
    gene_variation['signaling'] = compute_metrics(dat_tot)
    adata.uns['cluster_variation'] = gene_variation

def grn_statistics(adata,
                   weight_quantile=0.5,
                   k=None,
                   normalized=True,
                   weight=None,
                   endpoints=False,
                   seed=None
                   ):
    '''
    Computes various statistics on the cell state specific GRNs. The statistics are added to adata.uns['GRN_statistics']

    For a more detailed discussion of several of the parameters, please see the betweenness_centrality from Networkx
    (https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html)
    Results are stored in adata.uns['GRN_statistics'], adata.uns['cluster_variation'] and adata.uns['cluster_average']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    weight_quantile: `float` (default: 0.5)
        cutoff for weak gene-gene interactions between 0 and 1
    k: `int` or `None` (default=`None`)
        number of nodes considered to compute betweenness centrality. k=None implies that all edges are used
    normalized: `Bool` (default: True)
        if True, betweenness centrality values are normalized
    weight: `str` (default: `None`)
        If None, all edge weights are considered equal
    endpoints: `Bool` (default: False)
        If True, include the endpoints in the shortest path counts during betweenness centrality calculation
    seed: `int` (default: None)
        seed for betweenness centrality calculation

    Returns
    -------
    None

    '''
    assert 'jacobian_lists' in adata.uns.keys(), "Please run 'tl.estimate_jacobian' before calling 'grn_statistics'"

    genes = list(adata.var_names)
    n = len(genes)

    jacobian_lists = adata.uns['jacobian_lists']
    types = jacobian_lists.keys()

    betwenness_cent, incoming, outgoing, total_sign = {}, {}, {}, {}

    # repeat analysis for all clusters
    for t in types:
        df_bc, df_inc = pd.DataFrame( index=genes ), pd.DataFrame( index=genes )
        df_out, df_tot = pd.DataFrame( index=genes ), pd.DataFrame( index=genes )

        jacs = jacobian_lists[t][0]
        for j in range(len(jacs)):
            # extract 'interacting' quadrant of the Jacobian
            A = jacs[j][0:n, n:].copy().T

            rec, sen = np.sum(np.abs(A), axis=0), np.sum(np.abs(A), axis=1)
            tot_sign = rec + sen

            q_pos = np.quantile(A[A > 0], weight_quantile)
            q_neg = np.quantile(A[A < 0], 1 - weight_quantile)
            A[(A > q_neg) & (A < q_pos)] = 0
            G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
            centrality = nx.betweenness_centrality(G, k=k, normalized=normalized, weight=weight, endpoints=endpoints, seed=seed)

            df_bc[j] = centrality.values()
            df_inc[j] = rec
            df_out[j] = sen
            df_tot[j] = tot_sign

        betwenness_cent[t] = df_bc
        incoming[t] = df_inc
        outgoing[t] = df_out
        total_sign[t] = df_tot

    adata.uns['GRN_statistics'] = [betwenness_cent, incoming, outgoing, total_sign]

    GRN_cluster_variation(adata)

    # compute signaling scores of individual genes in each cell state
    signaling_score(adata)