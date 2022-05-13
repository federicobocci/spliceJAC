import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import networkx as nx
from scipy.sparse.csgraph import connected_components
from pyvis.network import Network

from . import plots

def tg_bif_sankey(adata, start, end, alpha_cluster=0.5, show_node_name=True, label_columns=True,
                  showfig=True, savefig=True, figname='sankey.pdf', format='pdf'):

    assert 'transitions' in adata.uns.keys(), "Please run transition analysis before calling tg_bif_sankey()"

    if 'colors' not in adata.uns.keys():
        plots.plot_setup(adata)

    # get TG of each cluster
    tg, cluster = [], []
    for c in end:
        deg_list, tg_list, both_list = adata.uns['transitions'][start + '-' + c]['gene_lists']
        tg = tg + tg_list + both_list
        cluster = cluster + [c for i in range(len(tg_list)+len(both_list))]

    # construct source and target arrays
    i, i_c = 0, 0
    source, label = [], []
    for t in tg:
        if t not in label:
            source.append(i)
            # target.append(cluster[i_c])
            label.append(t)
            i = i + 1
        else:
            j = label.index(t)
            source.append(j)
            # target.append(cluster[i_c])
        i_c = i_c + 1

    target = []
    i = len(set(tg))
    type = cluster[0]
    label.append(type)
    for c in cluster:
        if c!=type:
            type=c
            label.append(type)
            i = i + 1
        target.append(i)
    value = [1 for s in source]


    genes=list(set(tg))

    # define genes colors
    col_scale = list(plt.cm.Set3.colors)
    if len(col_scale)>len(set(source)):
        colors = list(plt.cm.Set3.colors)[0:len(set(source))]
    else:
        colors = []
        for i in range( len(set(source))//len(col_scale) ):
            for c in col_scale:
                colors.append(c)
        # colors.append( col_scale[0:len(set(source))%len(col_scale)] )
        for c in col_scale[0:len(set(source))%len(col_scale)]:
            colors.append(c)


    gene_colors = []
    for i in range(len(genes)):
        rgb_scaled = tuple([255 * c for c in colors[i]])
        gene_colors.append('rgb' + str(rgb_scaled))

    # define receiving node colors and link colors
    cluster_colors = []
    for e in end:
        rgb_scaled = tuple([ 255*c for c in adata.uns['colors'][e] ])
        cluster_colors.append('rgb' + str(rgb_scaled))

    n_s = len(set(source))
    link_color = []
    for t in target:
        link_color.append( cluster_colors[t-n_s] )


    ### about static export og images in python: https://plotly.com/python/static-image-export/
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Sankey(
        # to cancel plotly labels
        textfont=dict(color="rgba(0,0,0,0)", size=1),
        node=dict(
            pad=0,  # vertical gap between nodes
            thickness=40,  # width of nodes
            line=dict(color="black", width=0.5),
            label=label, #genes+clusters,
            color= gene_colors + cluster_colors
        ),
        link=dict(
            source= source,
            target= target,
            value=value,
            color=link_color
        ))])

    # clst = ['Alpha', 'Beta', 'Delta', 'Epsilon']
    # y_pos = [0.125, 0.375, 0.625, 0.875]
    # for c, y in zip(clst, y_pos):
    #     fig.add_annotation(text=c, xref="paper", yref="paper", x=1., y=y, showarrow=False, align='center', textangle=-90)

    if label_columns:
        fig.add_annotation(text='TG', xref="paper", yref="paper", x=0., y=1.1, showarrow=False, align='center')
        fig.add_annotation(text='Final State', xref="paper", yref="paper", x=1.05, y=1.1, showarrow=False, align='center')
    fig.update_layout(autosize=False, width=400, height=600, font_size=15)

    if savefig:
        fig.write_image(figname, format=format)
    if showfig:
        fig.show()