import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import iqr

from . import inference

### methods to quantify differences between Jacobian matrices

def count_sign_change(v1, v2):
    '''
    count number of sign changes between two matrices
    :param v1: matrix 1
    :param v2: matrix 2
    :return: fraction of changed signs
    '''
    assert v1.shape == v2.shape, 'Matrices with different shapes'
    c, n = 0, v1.shape[0]
    for i in range(n):
        for j in range(n):
            if np.sign(v1[i][j])!=np.sign(v2[i][j]):
                c = c + 1
    return float(c)/(n*n)

def mat_distance(v1, v2):
    '''
    distance between two matrices by summing element-wise difference
    :param v1: matrix 1
    :param v2: matrix 2
    :return: matrix distance normalized by number of elements
    '''
    assert v1.shape==v2.shape, 'Matrices with different shapes'
    n = v1.shape[0]
    return np.sum( np.abs(v1-v2) )/(n*n)

def count_weight_sign(v1, v2):
    '''
    distance between two matrices by summing element-wise difference
    :param v1: matrix 1
    :param v2: matrix 2
    :return: matrix distance normalized by number of elements
    '''
    assert v1.shape == v2.shape, 'Matrices with different shapes'
    c, n = 0, v1.shape[0]
    for i in range(n):
        for j in range(n):
            if np.sign(v1[i][j])!=np.sign(v2[i][j]):
                c = c + np.abs(v1[i][j]-v2[i][j])
    return float(c)/(n*n)



def coeff_dist(J, x = np.logspace(-5, 0, num=100)):

    pars = np.abs(np.ndarray.flatten(J))
    coeffs = pars/np.amax(pars)
    # x = np.logspace(-5, 0, num=100)
    y = np.zeros(x.size)
    for i in range(x.size):
        y[i] = coeffs[coeffs > x[i]].size
    y = y/pars.size
    return y


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


### functions to quantify incoming and outgoing interaction strength

def signaling_score(adata):
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


def energy(adata):
    n = len(list(adata.var_names))
    energy, cluster = np.zeros(len(list(adata.obs_names))), list(adata.obs['clusters'])
    count = adata.X.toarray()
    for c, cell in enumerate(adata.obs_names):
        s = count[c]
        J = adata.uns['average_jac'][cluster[c]][0][0:n, n:].copy()
        # print( c, cell, s.size, J.shape, np.matmul(J, s).shape )
        energy[c] = np.vdot(s, np.matmul(J, s))

    adata.obs['energy'] = energy



### functions to identify transition genes

def find_dir(w,
             v,
             dir_method='top_eig',
             eig_number=5
             ):

    ind = np.flip(np.argsort(np.real(w)))
    dir = np.zeros(w.size)

    if dir_method == 'top_eig':
        for i in range(eig_number):
            dir = dir + np.real(np.ndarray.flatten(v[:, ind[i]]))
        dir = dir/eig_number

    # look only at positive eigenvalues
    elif dir_method == 'positive':
        w = np.real(w)
        for i in range(w.size):
            if w[i] > 0.:
                dir = dir + np.real(np.ndarray.flatten(v[:, i]))
    return dir


def find_trans_genes(adata,
                     cluster1,
                     cluster2,
                     dir_method='top_eig',
                     eig_number=5,
                     first_moment=True
                     ):

    assert dir_method=='top_eig' or dir_method=='positive', "Please choose between dir_method='top_eig' or dir_method='positive'"

    if 'transitions' not in adata.uns.keys():
        adata.uns['transitions'] = {}

    # compute centers of clusters
    data1 = adata[adata.obs['clusters'] == cluster1]
    data2 = adata[adata.obs['clusters'] == cluster2]
    if first_moment:
        U1, S1 = data1.layers['Mu'], data1.layers['Ms']
        U2, S2 = data2.layers['Mu'], data2.layers['Ms']
    else:
        U1, S1 = data1.layers['unspliced'].toarray(), data1.layers['spliced'].toarray()
        U2, S2 = data2.layers['unspliced'].toarray(), data2.layers['spliced'].toarray()

    avgU1, avgS1 = np.mean(U1, axis=0), np.mean(S1, axis=0)
    avgU2, avgS2 = np.mean(U2, axis=0), np.mean(S2, axis=0)
    C1, C2 = np.concatenate((avgU1, avgS1)), np.concatenate((avgU2, avgS2))
    b = (C2-C1)/np.linalg.norm(C2-C1)

    # select eigenvalues, eigenvectors of clusters
    avg_jac = adata.uns['average_jac']
    w, v = avg_jac[cluster1][1], avg_jac[cluster1][2]
    dir = find_dir(w, v, dir_method=dir_method, eig_number=eig_number)
    proj = np.sum(dir * b) * b
    # normalize
    m = int(proj.size / 2)
    proj = proj[0:m] + proj[m:]
    proj = proj / np.linalg.norm(proj)
    return np.sqrt(proj ** 2)



def select_top_trans_genes(adata,
                     cluster1,
                     cluster2,
                     top_DEG=5,
                     top_TG=5
                     ):
    '''Returns lists of top differentially expressed genes (DEG), transition genes (TF), and both for transition from cluster1 to cluster2

    If some genes are both top DEG and top TG, they are classified in the both_list,
    and n genes are selected until a total of top_DEG+top_TG genes are selected

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        anndata object
    cluster1: 'string'
        starting cluster
    cluster2: 'string'
        ending cluster
    top_DEG: 'int'
        number of top DEG to select (default: top_DEG=5)
    top_TG: 'int'
        number of top TG to select (default: top_TG=5)

    Returns
    -------
    None

    '''
    assert 'rank_genes_groups' in adata.uns.keys(), "Please run 'sc.tl.rank_genes_groups()' first"

    weight = adata.uns['transitions'][cluster1 + '-' + cluster2]['weights']
    genes = list(adata.var_names)

    deg_genes = adata.uns['rank_genes_groups']['names'][cluster1]
    scores = adata.uns['rank_genes_groups']['scores'][cluster1]
    ord_weight = []
    for d in deg_genes:
        i = genes.index(d)
        ord_weight.append(weight[i])
    ord_weight = np.asarray(ord_weight)

    i_s, i_w = np.flip(np.argsort(scores)), np.flip(np.argsort(ord_weight))
    deg_list, tg_list, both_list = [], [], []

    for i in range(top_DEG):
        deg_list.append(deg_genes[i_s[i]])
    for i in range(top_TG):
        tg_list.append(deg_genes[i_w[i]])

    both_list = list(set(deg_list) & set(tg_list))
    for b in both_list:
        deg_list.remove(b)
        tg_list.remove(b)

    if len(deg_list) + len(tg_list) + len(both_list) < top_DEG + top_TG:
        cum_score = np.sqrt((ord_weight / np.amax(ord_weight))**2 + ((scores - np.amin(scores)) / (np.amax(scores) - np.amin(scores)))**2)
        cum_ind = np.flip(np.argsort(cum_score))
        for c in cum_ind:
            if deg_genes[c] not in deg_list+tg_list+both_list:
                a, b = (ord_weight[c]/np.amax(ord_weight))/cum_score[c], ((scores[c] - np.amin(scores))/(np.amax(scores) - np.amin(scores)))/cum_score[c]
                if a>b:
                    tg_list.append(deg_genes[c])
                else:
                    deg_list.append(deg_genes[c])
            if len(deg_list)+len(tg_list)+len(both_list)==(top_DEG+top_TG):
                break
    adata.uns['transitions'][cluster1 + '-' + cluster2]['gene_lists'] = [deg_list, tg_list, both_list]


def transition_genes(adata,
                     cluster1,
                     cluster2,
                     dir_method='top_eig',
                     eig_number=5,
                     top_DEG=5,
                     top_TG=5
                     ):
    weights = find_trans_genes(adata, cluster1, cluster2, dir_method=dir_method, eig_number=eig_number)
    trans_dct = {'weights': weights}
    adata.uns['transitions'][cluster1 + '-' + cluster2] = trans_dct
    select_top_trans_genes(adata, cluster1, cluster2, top_DEG=top_DEG, top_TG=top_TG)


def trans_from_PAGA(adata,
                    dir_method='top_eig',
                    eig_number=5,
                    top_DEG=5,
                    top_TG=5
                    ):
    trans_df = adata.uns['PAGA_paths']
    clusters = list(trans_df.index)

    rates = trans_df.to_numpy()
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if rates[i][j]>0.:
                weights = find_trans_genes(adata, clusters[i], clusters[j], dir_method=dir_method, eig_number=eig_number)
                trans_dct = {'weights': weights}
                adata.uns['transitions'][clusters[i] + '-' + clusters[j]] = trans_dct
                select_top_trans_genes(adata, clusters[i], clusters[j], top_DEG=top_DEG, top_TG=top_TG)



# def bif_grn(adata, start, end, dir_method='top_eig', eig_number=5, top_DEG=5, top_TG=5):
#     # get DEG of starting cluster
#
#     # get projection weights from start to all ending clusters
#     weights = []
#     for cluster in end:
#         if start+'-'+cluster not in adata.uns['transitions'].keys():
#             transition_genes(adata, start, cluster, dir_method=dir_method, eig_number=eig_number, top_DEG=top_DEG, top_TG=top_TG)
#         weights.append(adata.uns['transitions'][start + '-' + cluster]['weights'])
#
#     # compute membership of each gene
#     weights = np.asarray(weights)
#     norm = np.sum(weights**2, axis=0)
#     memb = np.asarray([ (weights[i]**2)/norm for i in range(weights[:,0].size) ])
#     # for i in range(memb[0].size):
#     #     print(memb[:,i])
#
#     # find top unique TG for each cluster
#
#
#


#
#
# ### functions to construct aggregate GRN ###
#
# def unravel_jac(adata):
#     '''
#     Create a dataframe with gene-gene interaction strength in each cluster
#     Results are saved in adata.uns['aggregated_grn']
#     Parameters
#     ----------
#     adata: anndata object
#
#     Returns
#     -------
#     None
#
#     '''
#     types = sorted(list(set(list(adata.obs['clusters']))))
#     n = len(list(adata.var_names))
#
#     int_data = {}
#
#     genes = list(adata.var_names)
#     gene1, gene2 = [], []
#     for i in range(len(genes)):
#         for j in range(len(genes)):
#             gene1.append(genes[i])
#             gene2.append(genes[j])
#     int_data['Gene1'] = gene1
#     int_data['Gene2'] = gene2
#
#     for t in types:
#         int_list = np.zeros(n*n)
#         J = adata.uns['average_jac'][t][0]
#         m = int(J.shape[0]/2)
#         B = np.transpose(J[0:m, m:])
#         c = 0
#         for i in range(n):
#             for j in range(n):
#                 int_list[c] = B[i][j]
#                 c = c + 1
#         int_data[t] = int_list
#     int_df = pd.DataFrame.from_dict(int_data)
#     adata.uns['aggregated_grn'] = int_df
#
#
# def filter_int(adata, eps=0.1):
#     '''
#     Filter the aggregated_grn dataframe by keeping only the nodes with highest interaction strength
#     Parameters
#     ----------
#     adata: anndata object
#     eps: fraction of interactions to keep (default: eps=0.1)
#
#     Returns
#     -------
#     None
#
#     '''
#     int_df = adata.uns['aggregated_grn']
#     weight = np.asarray( np.sqrt(np.square(int_df.iloc[:,2:]).sum(axis=1)) )
#     int_df['weight'] = weight
#     thr = np.sort(weight)[int((1-eps)*weight.size)]
#     int_df = int_df[int_df['weight']>thr]
#
#     int_df = int_df.reset_index()
#     int_df.drop(labels=['weight', 'index'], axis=1, inplace=True)
#     # for index, row in int_df.iterrows():
#     #     v = row[2:].to_numpy()
#     #     if np.amax(np.abs(v))<eps:
#     #         int_df.drop([index], inplace=True)
#     # int_df = int_df.reset_index()
#     # int_df.drop(['index'], axis=1, inplace=True)
#     adata.uns['aggregated_grn'] = int_df
#
#
# def consensus_sign(adata, p=0.75):
#     types = sorted(list(set(list(adata.obs['clusters']))))
#     int_df = adata.uns['aggregated_grn']
#     n_int = int_df.shape[0]
#     pp, pm = np.zeros(n_int), np.zeros(n_int)
#     sign, cluster, color = [], [], []
#
#     colors = list(plt.cm.Set3.colors)[0:len(types)]
#
#     for index, row in int_df.iterrows():
#         w = row[2:].to_numpy()
#         fp, fm = np.sum(np.abs(w[w>0])), np.sum(np.abs(w[w<0]))
#         pp[index], pm[index] = fp/(fp+fm), fm/(fp+fm)
#         if pp[index]>p:
#             sign.append('positive')
#             i = np.argmax(w)
#             cluster.append(types[i])
#             color.append(colors[i])
#         elif pm[index]>p:
#             sign.append('negative')
#             i = np.argmin(w)
#             cluster.append(types[i])
#             color.append(colors[i])
#         else:
#             sign.append('undetermined')
#             cluster.append('NA')
#             color.append('NA')
#     int_df['sign'] = sign
#     int_df['cluster'] = cluster
#     int_df['color'] = color
#     adata.uns['aggregated_grn'] = int_df
#
#



### GRN statistic  section

def compute_metrics(dat):

    SD, rng, int_qrt = np.zeros(dat[0].size), np.zeros(dat[0].size), np.zeros(dat[0].size)

    SD = np.std(dat, axis=0)

    for i in range(rng.size):
        rng[i] = np.amax(dat[:, i]) - np.amin(dat[:, i])

    int_qrt = iqr(dat, axis=0)

    d = {'SD':SD, 'range':rng, 'inter_range':int_qrt}

    return pd.DataFrame(d)


def GRN_cluster_variation(adata):
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
    Computes various statistics on the cell type specific GRNs. The statistics are added to adata.uns['GRN_statistics'].
    For a more detailed discussion of several of the parameters, please see the betweenness_centrality function from Networkx
    (https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html)
    Parameters
    ----------
    adata: anndata object
    weight_quantile: cutoff for weak interactions in the Jacobian matrix
    k: number of nodes considered to compute betweenness centrality (default=None)
    normalized: if True, betweenness values are normalized (default=True)
    weight: If None, all edge weights are considered equal (default=None)
    endpoints: If True include the endpoints in the shortest path counts (default=False)
    seed: seed for betweenness centrality calculation (default=None)

    Returns
    -------
    None

    '''
    assert 'jacobian_lists' in adata.uns.keys(), "Please run the 'estimate_jacobian' function before calling 'grn_statistics'"

    genes = list(adata.var_names)
    n = len(genes)

    jacobian_lists = adata.uns['jacobian_lists']
    types = jacobian_lists.keys()

    betwenness_cent, incoming, outgoing, total_sign = {}, {}, {}, {}

    # repeat analysis for all clusters
    for t in types:
        df_bc, df_inc, df_out, df_tot = pd.DataFrame( index=genes ), pd.DataFrame( index=genes ), pd.DataFrame( index=genes ), pd.DataFrame( index=genes )

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
            # https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html
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








