# tools to ease plotting


import os, sys
import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def make_point_centric_profile_heatmap(point_file, bigwig_files, prefix, sort=False, kval=4, referencepoint='TSS', extend_dist=1000):
    '''
    Uses deeptools to make a profile heatmap
    '''
    
    # do the same with TSS
    point_matrix = '{}.point.mat.gz'.format(prefix)
    deeptools_compute_matrix = ("computeMatrix reference-point "
                                "--referencePoint {0} "
                                "-b {1} -a {1} "
                                "-R {2} "
                                "-S {3} "
                                #"--skipZeros "
                                "-o {4} ").format(referencepoint,
                                                  extend_dist,
                                                  point_file,
                                                  ' '.join(bigwig_files),
                                                  point_matrix)
    if not os.path.isfile(point_matrix):
        print deeptools_compute_matrix
        os.system(deeptools_compute_matrix)

    # set up sample labels as needed
    sample_labels = []
    for bigwig_file in bigwig_files:
        fields = os.path.basename(bigwig_file).split('.')
        sample_labels.append('{0}_{1}_{2}'.format(fields[3].split('-')[0],
                                                  fields[0].split('-')[1],
                                                  fields[4]))
    
    # make plot
    point_plot = '{}.heatmap.profile.png'.format(prefix)
    point_sorted_file = '{}.point.sorted.bed'.format(prefix)
    if sort == False:
        sorting = '--sortRegions=no'
    elif kval == 1:
        sorting = ''
    else:
        sorting = '--kmeans {0} --regionsLabel {1}'.format(kval, ' '.join([str(i) for i in range(kval)]))
    deeptools_plot_heatmap = ("plotHeatmap -m {0} "
                                  "-out {1} "
                                  "--outFileSortedRegions {2} "
                                  "--colorMap Blues "
                                  "{3} "
                                  "--samplesLabel {4} "
                                  "--xAxisLabel '' "
                                  "--refPointLabel Summit "
                                  "--legendLocation none "
                                  "--heatmapHeight 50").format(point_matrix,
                                                               point_plot,
                                                               point_sorted_file,
                                                               sorting,
                                                               ' '.join(sample_labels))
    if not os.path.isfile(point_plot):
        print deeptools_plot_heatmap
        os.system(deeptools_plot_heatmap)
        

    return None


# Avanti Shrikumar deeplift visualization code


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(ax, array,
                 height_padding_factor,
                 length_padding,
                 subticks_frequency,
                 highlight,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
            
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)


def plot_weights(array,
  fig_name, 
                 figsize=(150,2), # 20,2
                 height_padding_factor=0.2,
                 length_padding=0.1, #1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)
    plt.savefig(fig_name)
    plt.close()
