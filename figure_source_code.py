'''
This code reproduces the figures about the pancreas endocrinogenesis analysis presented in the main Figure 3 and related Appendix Figures
The easiest way to reproduce the figures is to download the spliceJAC github folder and run this code within the directory.
It is assumed that the count matrix is in the local datasets/Pancreas/ folder
All figures will be saved in a local folder results/pancreas/
Additional note: here, spliceJAC's plotting tools are run with default settings, which might be varied case-by-case to beautify the figures
'''
import os
import scanpy as sc
import scvelo as scv
import splicejac as sp

def load_and_preprocess():
    '''
    Load the pancreas dataset from the local datasets folder and run required calculations

    Returns
    -------
    panc_data: the anndata count matrix for the pancreas dataset

    '''
    panc_data = sc.read_h5ad('datasets/Pancreas/pancreas.h5ad')

    # filter and normalize counts
    scv.pp.filter_genes(panc_data, min_shared_counts=20)
    scv.pp.normalize_per_cell(panc_data)
    scv.pp.filter_genes_dispersion(panc_data, n_top_genes=2000)
    scv.pp.log1p(panc_data)

    # compute RNA velocity, distance and connectivities
    scv.tl.velocity(panc_data)
    scv.tl.velocity_graph(panc_data)
    panc_data.uns['neighbors']['distances'] = panc_data.obsp['distances']
    panc_data.uns['neighbors']['connectivities'] = panc_data.obsp['connectivities']

    # infer transitions with PAGA
    scv.tl.paga(panc_data, groups='clusters')
    # add PAGA transitions for later use
    panc_data.uns['PAGA_paths'] = scv.get_df(panc_data, 'paga/transitions_confidence').T

    return panc_data

def run_splicejac(adata):
    '''
    Run the spliceJAC inference and downstream analysis calculations

    Parameters
    ----------
    adata: the anndata count matrix for the pancreas dataset

    Returns
    -------
    None

    '''
    # run the spliceJAC state-wise inference
    sp.tl.estimate_jacobian(adata, n_top_genes=50)
    # compute GRN statistics
    sp.tl.grn_statistics(adata)
    # compute pairwise GRN similarity
    sp.tl.grn_comparison(adata)

    # DEG scores
    sc.tl.rank_genes_groups(adata, 'clusters', method='t-test')
    # transition driver genes on the transitions identified with PAGA
    sp.tl.trans_from_PAGA(adata)

def Jacobian_figures(adata):
    '''
    Generate figures related to Jacobian visualization and comparison between cell states
    The figures are saved in the local results/pancreas/ folder

    Parameters
    ----------
    adata: the anndata count matrix for the pancreas dataset

    Returns
    -------
    None

    '''
    sp.pl.visualize_jacobian(adata, figname='results/pancreas/jacobians.pdf')
    sp.pl.eigen_spectrum(adata, figname='results/pancreas/eigen_spectrum.pdf')

def state_GRN_figures(adata):
    '''
    Generate figures related to cell state-specific GRN and signaling roles

    Parameters
    ----------
    adata: the anndata count matrix for the pancreas dataset

    Returns
    -------
    None

    '''
    # plot cell state specific GRNs with setting of weight_quantile=.95 to select only the top 5% interactions
    sp.pl.visualize_network(adata, 'Ductal', weight_quantile=.95, plot_interactive=False,
                            figname='results/pancreas/ductal_GRN_wq095.pdf', figsize=(5, 3.75))
    sp.pl.visualize_network(adata, 'Ngn3 low EP', weight_quantile=.95, plot_interactive=False,
                            figname='results/pancreas/Ngn3_lowEP_GRN_wq095.pdf', figsize=(5, 3.75))
    sp.pl.visualize_network(adata, 'Ngn3 high EP', weight_quantile=.95, plot_interactive=False,
                            figname='results/pancreas/Ngn3_highEP_GRN_wq095.pdf', figsize=(5, 3.75))
    sp.pl.visualize_network(adata, 'Pre-endocrine', weight_quantile=.95, plot_interactive=False,
                            figname='results/pancreas/Pre_endocrine_GRN_wq095.pdf', figsize=(5, 3.75))
    sp.pl.visualize_network(adata, 'Alpha', weight_quantile=.95, plot_interactive=False,
                            figname='results/pancreas/Alpha_GRN_wq095.pdf', figsize=(5, 3.75))
    sp.pl.visualize_network(adata, 'Beta', weight_quantile=.95, plot_interactive=False,
                            figname='results/pancreas/Beta_GRN_wq095.pdf', figsize=(5, 3.75))
    sp.pl.visualize_network(adata, 'Delta', weight_quantile=.95, plot_interactive=False,
                            figname='results/pancreas/Delta_GRN_wq095.pdf', figsize=(5, 3.75))
    sp.pl.visualize_network(adata, 'Epsilon', weight_quantile=.95, plot_interactive=False,
                            figname='results/pancreas/Epsilon_GRN_wq095.pdf', figsize=(5, 3.75))

    # plot the signaling hubs for each cell state
    sp.pl.plot_signaling_hubs(adata, 'Ductal', figname='results/pancreas/Ductal_hubs.pdf')
    sp.pl.plot_signaling_hubs(adata, 'Ngn3 low EP', figname='results/pancreas/Ngn3_lowEP_hubs.pdf')
    sp.pl.plot_signaling_hubs(adata, 'Ngn3 high EP', figname='results/pancreas/Ngn3_highEP_hubs.pdf')
    sp.pl.plot_signaling_hubs(adata, 'Pre-endocrine', figname='results/pancreas/Pre_endocrine_hubs.pdf')
    sp.pl.plot_signaling_hubs(adata, 'Alpha', figname='results/pancreas/Alpha_hubs.pdf')
    sp.pl.plot_signaling_hubs(adata, 'Beta', figname='results/pancreas/Beta_hubs.pdf')
    sp.pl.plot_signaling_hubs(adata, 'Delta', figname='results/pancreas/Delta_hubs.pdf')
    sp.pl.plot_signaling_hubs(adata, 'Epsilon', figname='results/pancreas/Epsilon_hubs.pdf')

    # plots about the gene role variation across cell states
    sp.pl.gene_variation(adata, method='inter_range', figname='results/pancreas/node_variability.pdf',
                         figsize=(7, 3))
    sp.pl.gene_var_detail(adata, figname='results/pancreas/gene_var_detail.pdf')
    sp.pl.gene_var_scatter(adata, figname='results/pancreas/gene_var_scatter.pdf')

def transition_figures(adata):
    '''
    Generate figures about cell state transitions

    Parameters
    ----------
    adata: the anndata count matrix for the pancreas dataset

    Returns
    -------
    None

    '''
    # signaling change
    sp.pl.plot_signaling_change(adata, 'Ductal', 'Ngn3 low EP',
                                figname='results/pancreas/Ductal_to_lowEP_sign_change.pdf')
    sp.pl.plot_signaling_change(adata, 'Ngn3 low EP', 'Ngn3 high EP',
                                figname='results/pancreas/lowEP_to_highEP_sign_change.pdf')
    sp.pl.plot_signaling_change(adata, 'Ngn3 high EP', 'Pre-endocrine',
                                figname='results/pancreas/highEP_to_Pre_sign_change.pdf')
    sp.pl.plot_signaling_change(adata, 'Pre-endocrine', 'Alpha',
                                figname='results/pancreas/Pre_to_Alpha_sign_change.pdf')
    sp.pl.plot_signaling_change(adata, 'Pre-endocrine', 'Delta',
                                figname='results/pancreas/Pre_to_Delta_sign_change.pdf')
    sp.pl.plot_signaling_change(adata, 'Pre-endocrine', 'Beta',
                                figname='results/pancreas/Pre_to_Beta_sign_change.pdf')
    sp.pl.plot_signaling_change(adata, 'Pre-endocrine', 'Epsilon',
                                figname='results/pancreas/Pre_to_Epsilon_sign_change.pdf')

    # core GRN of DEG and transition genes
    sp.pl.core_GRN(adata, 'Ductal', 'Ngn3 low EP',
                                figname='results/pancreas/Ductal_to_lowEP_GRN.pdf', figsize=(4, 4))
    sp.pl.core_GRN(adata, 'Ngn3 low EP', 'Ngn3 high EP',
                                figname='results/pancreas/lowEP_to_highEP_GRN.pdf', figsize=(4, 4))
    sp.pl.core_GRN(adata, 'Ngn3 high EP', 'Pre-endocrine',
                                figname='results/pancreas/highEP_to_Pre_GRN.pdf', figsize=(4, 4))
    sp.pl.core_GRN(adata, 'Pre-endocrine', 'Alpha',
                                figname='results/pancreas/Pre_to_Alpha_GRN.pdf', figsize=(4, 4))
    sp.pl.core_GRN(adata, 'Pre-endocrine', 'Delta',
                                figname='results/pancreas/Pre_to_Delta_GRN.pdf', figsize=(4, 4))
    sp.pl.core_GRN(adata, 'Pre-endocrine', 'Beta',
                                figname='results/pancreas/Pre_to_Beta_GRN.pdf', figsize=(4, 4))
    sp.pl.core_GRN(adata, 'Pre-endocrine', 'Epsilon',
                                figname='results/pancreas/Pre_to_Epsilon_GRN.pdf', figsize=(4, 4))

    # transition driver gene scores
    sp.pl.plot_trans_genes(adata, 'Ductal', 'Ngn3 low EP', figname='results/pancreas/Ductal_to_lowEP_genes.pdf')
    sp.pl.plot_trans_genes(adata, 'Ngn3 low EP', 'Ngn3 high EP', figname='results/pancreas/lowEP_to_highEP_genes.pdf')
    sp.pl.plot_trans_genes(adata, 'Ngn3 high EP', 'Pre-endocrine', figname='results/pancreas/highEP_to_Pre_genes.pdf')
    sp.pl.plot_trans_genes(adata, 'Pre-endocrine', 'Alpha',  figname='results/pancreas/Pre_to_Alpha_genes.pdf')
    sp.pl.plot_trans_genes(adata, 'Pre-endocrine', 'Delta', figname='results/pancreas/Pre_to_Delta_genes.pdf')
    sp.pl.plot_trans_genes(adata, 'Pre-endocrine', 'Beta', figname='results/pancreas/Pre_to_Beta_genes.pdf')
    sp.pl.plot_trans_genes(adata, 'Pre-endocrine', 'Epsilon', figname='results/pancreas/Pre_to_Epsilon_genes.pdf')

    # scatter plots to compare spliceJAC's instability scores and DEG gene score in the starting cell state
    sp.pl.scatter_scores(adata, 'Ductal', 'Ngn3 low EP', figname='results/pancreas/Ductal_to_lowEP_scores.pdf')
    sp.pl.scatter_scores(adata, 'Ngn3 low EP', 'Ngn3 high EP', figname='results/pancreas/lowEP_to_highEP_scores.pdf')
    sp.pl.scatter_scores(adata, 'Ngn3 high EP', 'Pre-endocrine', figname='results/pancreas/highEP_to_Pre_scores.pdf')
    sp.pl.scatter_scores(adata, 'Pre-endocrine', 'Alpha', figname='results/pancreas/Pre_to_Alpha_scores.pdf')
    sp.pl.scatter_scores(adata, 'Pre-endocrine', 'Delta', figname='results/pancreas/Pre_to_Delta_scores.pdf')
    sp.pl.scatter_scores(adata, 'Pre-endocrine', 'Beta', figname='results/pancreas/Pre_to_Beta_scores.pdf')
    sp.pl.scatter_scores(adata, 'Pre-endocrine', 'Epsilon', figname='results/pancreas/Pre_to_Epsilon_scores.pdf')

def GRN_similarity(adata):
    '''
    Plots regarding differential and conserved gene-gene interactions during cell state transitions

    Parameters
    ----------
    adata: the anndata count matrix for the pancreas dataset

    Returns
    -------
    None

    '''
    # differential network
    sp.pl.diff_network(adata, 'Ductal', 'Ngn3 low EP', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Diff_GRN_Ductal_to_lowEP.pdf')
    sp.pl.diff_network(adata, 'Ngn3 low EP', 'Ngn3 high EP', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Diff_GRN_lowEP_to_highEP.pdf')
    sp.pl.diff_network(adata, 'Ngn3 high EP', 'Pre-endocrine', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Diff_GRN_highEP_to_Pre.pdf')
    sp.pl.diff_network(adata, 'Pre-endocrine', 'Alpha', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Diff_GRN_Pre_to_Alpha.pdf')
    sp.pl.diff_network(adata, 'Pre-endocrine', 'Beta', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Diff_GRN_Pre_to_Beta.pdf')
    sp.pl.diff_network(adata, 'Pre-endocrine', 'Delta', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Diff_GRN_Pre_to_Delta.pdf')
    sp.pl.diff_network(adata, 'Pre-endocrine', 'Epsilon', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Diff_GRN_Pre_to_Epsilon.pdf')

    # top differential interactions
    sp.pl.diff_interactions(adata, 'Ductal', 'Ngn3 low EP', figsize=(4, 6),
                            title='Top 10 Differential Interactions', legend_font=8,
                            legend_col=2, loc='upper center', figname='results/pancreas/Ductal_to_lowEP_change.pdf')
    sp.pl.diff_interactions(adata, 'Ngn3 low EP', 'Ngn3 high EP', figsize=(4, 6),
                            title='Top 10 Differential Interactions', legend_font=8,
                            legend_col=2, loc='upper center', figname='results/pancreas/lowEP_to_highEP_change.pdf')
    sp.pl.diff_interactions(adata, 'Ngn3 high EP', 'Pre-endocrine', figsize=(4, 6),
                            title='Top 10 Differential Interactions', legend_font=8,
                            legend_col=2, loc='upper center', figname='results/pancreas/highEP_to_Pre_change.pdf')
    sp.pl.diff_interactions(adata, 'Pre-endocrine', 'Alpha', figsize=(4, 6),
                            title='Top 10 Differential Interactions', legend_font=8,
                            legend_col=2, loc='upper center', figname='results/pancreas/Pre_to_Alpha_change.pdf')
    sp.pl.diff_interactions(adata, 'Pre-endocrine', 'Beta', figsize=(4, 6),
                            title='Top 10 Differential Interactions', legend_font=8,
                            legend_col=2, loc='upper center', figname='results/pancreas/Pre_to_Beta_change.pdf')
    sp.pl.diff_interactions(adata, 'Pre-endocrine', 'Delta', figsize=(4, 6),
                            title='Top 10 Differential Interactions', legend_font=8,
                            legend_col=2, loc='upper center', figname='results/pancreas/Pre_to_Delta_change.pdf')
    sp.pl.diff_interactions(adata, 'Pre-endocrine', 'Epsilon', figsize=(4, 6),
                            title='Top 10 Differential Interactions', legend_font=8,
                            legend_col=2, loc='upper center', figname='results/pancreas/Pre_to_Epsilon_change.pdf')

    # conserved GRN
    sp.pl.conserved_grn(adata, 'Ductal', 'Ngn3 low EP', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Cons_GRN_Ductal_to_lowEP.pdf')
    sp.pl.conserved_grn(adata, 'Ngn3 low EP', 'Ngn3 high EP', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Cons_GRN_lowEP_to_highEP.pdf')
    sp.pl.conserved_grn(adata, 'Ngn3 high EP', 'Pre-endocrine', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Cons_GRN_highEP_to_Pre.pdf')
    sp.pl.conserved_grn(adata, 'Pre-endocrine', 'Alpha', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Cons_GRN_Pre_to_Alpha.pdf')
    sp.pl.conserved_grn(adata, 'Pre-endocrine', 'Beta', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Cons_GRN_Pre_to_Beta.pdf')
    sp.pl.conserved_grn(adata, 'Pre-endocrine', 'Delta', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Cons_GRN_Pre_to_Delta.pdf')
    sp.pl.conserved_grn(adata, 'Pre-endocrine', 'Epsilon', pos_style='circle', figsize=(8, 6),
                       weight_quantile=0.975, figname='results/pancreas/Cons_GRN_Pre_to_Epsilon.pdf')

    # top conserved interactions
    sp.pl.top_conserved_int(adata, 'Ductal', 'Ngn3 low EP', figsize=(4, 6),
                            title='Top 10 Conserved Interactions',
                            figname='results/pancreas/Ductal_to_lowEP_conserved.pdf')
    sp.pl.top_conserved_int(adata, 'Ngn3 low EP', 'Ngn3 high EP', figsize=(4, 6),
                            title='Top 10 Conserved Interactions',
                            figname='results/pancreas/lowEP_to_highEP_conserved.pdf')
    sp.pl.top_conserved_int(adata, 'Ngn3 high EP', 'Pre-endocrine', figsize=(4, 6),
                            title='Top 10 Conserved Interactions',
                            figname='results/pancreas/highEP_to_Pre_conserved.pdf')
    sp.pl.top_conserved_int(adata, 'Pre-endocrine', 'Alpha', figsize=(4, 6),
                            title='Top 10 Conserved Interactions',
                            figname='results/pancreas/Pre_to_Alpha_conserved.pdf')
    sp.pl.top_conserved_int(adata, 'Pre-endocrine', 'Beta', figsize=(4, 6),
                            title='Top 10 Conserved Interactions',
                            figname='results/pancreas/Pre_to_Beta_conserved.pdf')
    sp.pl.top_conserved_int(adata, 'Pre-endocrine', 'Delta', figsize=(4, 6),
                            title='Top 10 Conserved Interactions',
                            figname='results/pancreas/Pre_to_Delta_conserved.pdf')
    sp.pl.top_conserved_int(adata, 'Pre-endocrine', 'Epsilon', figsize=(4, 6),
                            title='Top 10 Conserved Interactions',
                            figname='results/pancreas/Pre_to_Epsilon_conserved.pdf')

    # testing how the GRN of starting transition states predict the GRN of final transition states
    sp.pl.plot_grn_comparison(adata, figname='results/pancreas/grn_scores.pdf')
    # tracing the different transitions from Ductal to Alpha, Beta, Delta, or Epsilon
    sp.pl.adjecent_grn_score(adata, ['Ductal', 'Ngn3 low EP', 'Ngn3 high EP', 'Pre-endocrine', 'Alpha'],
                             figname='results/pancreas/adjacent_grn_score_Alpha.pdf')
    sp.pl.adjecent_grn_score(adata, ['Ductal', 'Ngn3 low EP', 'Ngn3 high EP', 'Pre-endocrine', 'Beta'],
                             figname='results/pancreas/adjacent_grn_score_Beta.pdf')
    sp.pl.adjecent_grn_score(adata, ['Ductal', 'Ngn3 low EP', 'Ngn3 high EP', 'Pre-endocrine', 'Delta'],
                             figname='results/pancreas/adjacent_grn_score_Delta.pdf')
    sp.pl.adjecent_grn_score(adata, ['Ductal', 'Ngn3 low EP', 'Ngn3 high EP', 'Pre-endocrine', 'Epsilon'],
                             figname='results/pancreas/adjacent_grn_score_Epsilon.pdf')


def scvelo_scores(adata):
    '''
    scatter plots to compare spliceJAC's instability scores with scVelo's cluster-wise likelihood scores

    Parameters
    ----------
    adata: the anndata count matrix for the pancreas dataset

    Returns
    -------
    None

    '''
    scv.tl.recover_dynamics(adata)
    scv.tl.rank_dynamical_genes(adata, groupby='clusters')
    sp.pl.compare_scvelo_scores(adata, figname='results/pancreas/scvelo_scores.pdf')

def differentiation_plots(adata):
    '''
    Generate figures to compare gene role and transition genes in differentiating branches

    Parameters
    ----------
    adata: the anndata count matrix for the pancreas dataset

    Returns
    -------
    None

    '''
    sp.pl.compare_standout_genes(adata, cluster_list=['Alpha', 'Beta', 'Delta', 'Epsilon'],
                                 figname='results/pancreas/bif_between.pdf')
    sp.pl.tg_bif_sankey(adata, 'Pre-endocrine', ['Alpha', 'Beta', 'Delta', 'Epsilon'],
                        figname='results/pancreas/sankey.pdf', showfig=False)
    sp.pl.bif_GRN(adata, 'Pre-endocrine', ['Alpha', 'Beta', 'Delta', 'Epsilon'], figname='results/pancreas/bif_GRN.pdf')

def robustness(adata):
    '''
    Generate figures related to robustness and stability

    Parameters
    ----------
    adata: the anndata count matrix for the pancreas dataset

    Returns
    -------
    None

    '''
    # test state stability and consistency of inference
    sp.pl.subsample_stability(adata, figname='results/pancreas/stability.pdf')

    # test sensitivity to regression method
    sp.tl.regr_method_sens(adata)
    sp.pl.regression_sens(adata, figname='results/pancreas/inference_method_sens.pdf')

    # test inference robustness to subsampling of cells
    sp.tl.subsampling_sens(adata)
    sp.pl.sampling_sens(adata, figname='results/pancreas/subsampling_sens.pdf')


def main():
    if not os.path.exists('results/pancreas'):
        os.mkdir('results')
        os.mkdir('results/pancreas')

    panc_data = load_and_preprocess()
    run_splicejac(panc_data)
    state_GRN_figures(panc_data)
    transition_figures(panc_data)
    scvelo_scores(panc_data)
    differentiation_plots(panc_data)
    GRN_similarity(panc_data)
    robustness(panc_data)


if __name__=='__main__':
    main()