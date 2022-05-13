import numpy as np


def jac_regr_cons(adata, cluster):

    assert 'jacobian_lists' in adata.uns.keys(), 'Run cluster stability first'

    jac_list = adata.uns['jacobian_lists'][cluster][0]
    nsim = len(jac_list)

    dev = []
    for i in range(nsim):
        for j in range( i ):
            J1, J2 = jac_list[i], jac_list[j]
            dev.append( np.linalg.norm(np.abs(J1 - J2)) )
    return dev


def compare_wrong_signs(j1, j2, eps=0.9):
    n = j1.shape[0]
    j = np.sort(np.abs(np.ndarray.flatten(np.concatenate((j1, j2)))))
    t = j[int(eps*j.size)]

    count = 0
    for i in range(n):
        for j in range(n):
            if np.abs(j1[i][j] ) >t or np.abs(j2[i][j] ) >t:
                if np.sign(j1[i][j] )!=np.sign(j2[i][j]):
                    count = count + 1
    return 100*count/float(( 1 -eps ) * n *n)


def count_wrong_signs(adata, cluster, eps=0.9):
    assert 'jacobian_lists' in adata.uns.keys(), 'Run cluster stability first'

    jac_list = adata.uns['jacobian_lists'][cluster][0]
    nsim = len(jac_list)

    wrong_sign = []
    for i in range(nsim):
        for j in range(i):
            J1, J2 = jac_list[i], jac_list[j]
            wrong_sign.append( compare_wrong_signs(J1, J2, eps=eps) )
    return wrong_sign



def count_pos_eig(adata, cluster):
    assert 'jacobian_lists' in adata.uns.keys(), 'Run cluster stability first'

    w_list = adata.uns['jacobian_lists'][cluster][1]
    nsim = len(w_list)
    m = 2 * len(list(adata.var_names))

    pos_eig = []
    for i in range(nsim):
        w = w_list[i]
        pos_eig.append( 100 *np.real(w)[np.real(w) >0].size/float(m) )
    return pos_eig


def cluster_inst_score(adata, cluster):
    assert 'jacobian_lists' in adata.uns.keys(), 'Run cluster stability first'

    w_list = adata.uns['jacobian_lists'][cluster][1]
    nsim = len(w_list)
    m = 2 * len(list(adata.var_names))

    score = []
    for i in range(nsim):
        w = w_list[i]
        real_w = np.real(w)
        score.append( np.sum(np.linalg.norm(real_w[real_w>0]))/np.sum(np.linalg.norm(real_w)) )
    return score