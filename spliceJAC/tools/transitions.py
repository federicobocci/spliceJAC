'''
functions to identify transition driver genes
'''
import numpy as np

def find_dir(w,
             v,
             dir_method='top_eig',
             eig_number=5
             ):
    '''Identify the unstable transition directions given the eigenspectrum of the starting cell state

    Parameters
    ----------
    w: `~numpy.ndarray`
        eigenvalues of cell state Jacobian
    v: `~numpy.ndarray`
        eigenvectors of cell state Jacobian
    dir_method: `str` (default: 'top_eig')
        method to select the unstable directions, choose between 'top_eig' and 'positive'.
        'top_eig' uses the largest eigenvalues irrespective of sign; 'positive' strictly uses positive eigenvalues
    eig_number: `int` (default: 5)
        number of largest eigenvalues to consider, required for dir_method='top_eig'

    Returns
    -------
    dir: `~numpy.ndarray`
        set of unstable directions

    '''
    ind = np.flip(np.argsort(np.real(w)))
    dir = np.zeros(w.size)

    if dir_method == 'top_eig':
        for i in range(eig_number):
            dir = dir + np.real(np.ndarray.flatten(v[:, ind[i]]))
        dir = dir/eig_number

    # look only at positive eigenvalues, raise an error of no positive eigenvalues were detected
    elif dir_method == 'positive':
        w = np.real(w)
        if np.amax(w)<0.:
            raise Exception("No positive eigenvalues were detected, please set dir_method='top_eig'")
        else:
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
    '''Compute the gene instability scores for transition from cluster1 to cluster2

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster1: `str`
        starting cell state
    cluster2: `str`
        final cell states
    dir_method: `str` (default: 'top_eig')
        method to select the unstable directions, choose between 'top_eig' and 'positive'.
        'top_eig' uses the largest eigenvalues irrespective of sign; 'positive' strictly uses positive eigenvalues
    eig_number: `int` (default: 5)
        number of largest eigenvalues to consider, required for dir_method='top_eig'
    first_moment: `Bool` (default: True)
        if True, use first moments of unspliced/spliced counts

    Returns
    -------
    weight: `~numpy.ndarray`
        weight of each gene for the specified transition

    '''
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
    # normalize weights
    m = int(proj.size / 2)
    proj = proj[0:m] + proj[m:]
    proj = proj / np.linalg.norm(proj)
    weight = np.sqrt(proj ** 2)
    return weight


def select_top_trans_genes(adata,
                     cluster1,
                     cluster2,
                     top_DEG=5,
                     top_TG=5
                     ):
    '''Returns lists of top differentially expressed genes (DEG), transition genes (TF), and both for transition from
    cluster1 to cluster2

    If some genes are both top DEG and top TG, they are classified in the both_list,
    and n genes are selected until a total of top_DEG+top_TG genes are selected
    Results are stored in adata.uns['transitions']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster1: `str`
        starting cell state
    cluster2: `str`
        final cell states
    top_DEG: `int` (default: 5)
        number of top DEG to select
    top_TG: `int` (default: 5)
        number of top TG to select

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
                     top_TG=5,
                     first_moment=True
                     ):
    '''Compute the gene instability scores for transition from cluster1 to cluster2
    Results are stored in adata.uns['transitions']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster1: `str`
        starting cell state
    cluster2: `str`
        final cell states
    dir_method: `str` (default: 'top_eig')
        method to select the unstable directions, choose between 'top_eig' and 'positive'.
        'top_eig' uses the largest eigenvalues irrespective of sign; 'positive' strictly uses positive eigenvalues
    eig_number: `int` (default: 5)
        number of largest eigenvalues to consider, required for dir_method='top_eig'
    top_DEG: `int` (default: 5)
        number of top DEG to select
    top_TG: `int` (default: 5)
        number of top TG to select
    first_moment: `Bool` (default: True)
        if True, use first moments of unspliced/spliced counts

    Returns
    -------
    None

    '''
    weights = find_trans_genes(adata, cluster1, cluster2, dir_method=dir_method, eig_number=eig_number, first_moment=first_moment)
    trans_dct = {'weights': weights}
    adata.uns['transitions'][cluster1 + '-' + cluster2] = trans_dct
    select_top_trans_genes(adata, cluster1, cluster2, top_DEG=top_DEG, top_TG=top_TG)


def trans_from_PAGA(adata,
                    dir_method='top_eig',
                    eig_number=5,
                    top_DEG=5,
                    top_TG=5,
                    first_moment=True
                    ):
    '''Compute the gene instability scores for all transitions identified with PAGA
    PAGA transitions must be stored as a dataframe in adata.uns['PAGA_paths']
    results are stored in adata.uns['transitions']

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    dir_method: `str` (default: 'top_eig')
        method to select the unstable directions, choose between 'top_eig' and 'positive'.
        'top_eig' uses the largest eigenvalues irrespective of sign; 'positive' strictly uses positive eigenvalues
    eig_number: `int` (default: 5)
        number of largest eigenvalues to consider, required for dir_method='top_eig'
    top_DEG: `int` (default: 5)
        number of top DEG to select
    top_TG: `int` (default: 5)
        number of top TG to select
    first_moment: `Bool` (default: True)
        if True, use first moments of unspliced/spliced counts

    Returns
    -------
    None

    '''
    trans_df = adata.uns['PAGA_paths']
    clusters = list(trans_df.index)

    rates = trans_df.to_numpy()
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if rates[i][j]>0.:
                weights = find_trans_genes(adata, clusters[i], clusters[j], dir_method=dir_method, eig_number=eig_number, first_moment=first_moment)
                trans_dct = {'weights': weights}
                adata.uns['transitions'][clusters[i] + '-' + clusters[j]] = trans_dct
                select_top_trans_genes(adata, clusters[i], clusters[j], top_DEG=top_DEG, top_TG=top_TG)


