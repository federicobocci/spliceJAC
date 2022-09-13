'''
methods to export spliceJAC's results
'''
import os
import numpy as np
import pandas as pd

def grn_to_csv(adata,
               cluster,
               filename
               ):
    '''Export the GRN of clusters to a csv file

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster: `str`
        cell state
    filename: `str`
        the name of export file, including path and .csv extension

    Returns
    -------
    None

    '''
    genes = list(adata.var_names)
    n = len(genes)
    grn = adata.uns['average_jac'][cluster][0][0:n, n:].copy()

    sorted = np.flip(np.argsort(np.abs(np.ndarray.flatten(grn))))

    regulator, target, interaction = [], [], []

    for i in range(sorted.size):
        k, l = np.unravel_index(sorted[i], shape=(n, n))

        if grn[k][l] == 0.:
            break

        regulator.append(genes[k])
        target.append(genes[l])
        interaction.append(grn[k][l])

    grn_dict = {'Regulator': regulator, 'Target': target, 'Interaction': interaction}
    grn_pd = pd.DataFrame.from_dict(grn_dict)

    grn_pd.to_csv(filename, index=False)

def export_grn(adata,
               cluster,
               filename=None
               ):
    '''Export the GRN of a cell state to a csv file

    If no filename is provided, a local folder results/exported_results/ will be created (if not existing) and the GRN
    file will be saved with default name "grn" + cluster + ".csv"

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    cluster: `str`
        cell state
    filename: `str` (default: None)
        the name of export file, including path and .csv extension

    Returns
    -------
    None

    '''
    if filename==None:
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists('results/export_output'):
            os.mkdir('results/export_output')
        filename = 'results/export_output/grn' + cluster + '.csv'

    grn_to_csv(adata, cluster, filename)


def export_transition_scores(adata,
                             filename=None
                             ):
    ''' Export gene transition scores for all transitions in a csv file

    If no filename is provided, a local folder results/exported_results/ will be created (if not existing) and the GRN
    file will be saved with default name "grn" + cluster + ".csv"

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    filename: `str` (defaul: None)
        the name of export file, including path and .csv extension

    Returns
    -------
    None

    '''
    if filename==None:
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists('results/export_output'):
            os.mkdir('results/export_output')
        filename = 'results/export_output/transition_score.csv'

    tg_dict = {'gene': list(adata.var_names)}
    for t in adata.uns['transitions'].keys():
        tg_dict[t] = np.asarray( adata.uns['transitions'][t]['weights'] )

    tg_pd = pd.DataFrame.from_dict(tg_dict)
    tg_pd.to_csv(filename, index=False)