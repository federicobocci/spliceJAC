'''
functions for Sankey diagrams
'''
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from . import plotting_util

def tg_bif_sankey(adata,
                  start,
                  end,
                  gene_colormap = plt.cm.Set3.colors,
                  width=400,
                  height=600,
                  font_size=15,
                  pad=0,
                  thickness=40,
                  label_columns=True,
                  showfig=True,
                  savefig=None,
                  format='pdf'
                  ):
    '''Plot a Sankey diagram of the top transition genes involved in different cell state transitions

    More details about static export of images in python can be found at: https://plotly.com/python/static-image-export/
    More details about the arguments of plotly objects can be fount at: https://plotly.com/python/graph-objects/

    Parameters
    ----------
    adata: `~anndata.AnnData`
        count matrix
    start: `str`
        starting cell state
    end: `list`
        list of final cell states
    gene_colormap: `pyplot colormap` (default: plt.cm.Set3.colors)
        Colormap for transition genes on the left side of the Sankey diagram. To use another colormap, provide argument
        following the same syntax: plt.cm. + chosen_colormap + .colors. A list of accepted colormaps can be found at:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    width: `int` (default=400)
        width of plotly figure
    height: `int` (default=600)
        height of plotly figure
    font_size: `int` (default: 15)
        font size of figure
    pad: `float` (default: 0)
        vertical gap between nodes of the Sankey plot
    thickness: `float` (default: 40)
        line thickness of the Sankey plot
    label_columns: `Bool` (default=True)
        if True, label the diagram columns
    showfig: `Bool` (default=True)
        if True, show the figure
    savefig: `Bool` or `None` (default=None)
        if True, save the figure using the savefig path
    format: `str` (default='pdf')
        format of saved figure

    Returns
    -------
    None

    '''
    assert 'transitions' in adata.uns.keys(), "Please run transition analysis before calling tg_bif_sankey()"

    if 'colors' not in adata.uns.keys():
        plotting_util.plot_setup(adata)

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
    col_scale = list(gene_colormap)
    if len(col_scale)>len(set(source)):
        colors = list(gene_colormap)[0:len(set(source))]
    else:
        colors = []
        for i in range( len(set(source))//len(col_scale) ):
            for c in col_scale:
                colors.append(c)
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

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=pad,  # vertical gap between nodes
            thickness=thickness,  # width of nodes
            line=dict(color="black", width=0.5),
            label=label,
            color= gene_colors + cluster_colors
        ),
        link=dict(
            source= source,
            target= target,
            value=value,
            color=link_color
        ))])

    if label_columns:
        fig.add_annotation(text='TG', xref="paper", yref="paper", x=0., y=1.1, showarrow=False, align='center')
        fig.add_annotation(text='Final State', xref="paper", yref="paper", x=1.05, y=1.1, showarrow=False, align='center')
    fig.update_layout(autosize=False, width=width, height=height, font_size=font_size)

    if savefig:
        fig.write_image(savefig, format=format)
    if showfig:
        fig.show()