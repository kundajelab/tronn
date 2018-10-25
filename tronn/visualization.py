# tools to ease plotting

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import logging

import numpy as np

from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys


# h/t Avanti Shrikumar - importance score visualization code
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
        ax.add_patch(matplotlib.patches.Polygon(
            (np.array([1,height])[None,:]*polygon_coords
             + np.array([left_edge,base])[None,:]),
            facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
        facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
        facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+1, base], width=1.0, height=height,
        facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
        facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
        facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+1, base], width=1.0, height=height,
        facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
        facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
        facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+0.4, base],
        width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge, base+0.8*height],
        width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(
        ax, array,
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

        if np.sum(array[i,:]) == 0:
            continue

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

# TODO - give height axis settings
def plot_weights(
        array,
        fig_name, 
        figsize=(150,2), # 20,2
        height_padding_factor=0.2,
        length_padding=0.1, #1.0,
        subticks_frequency=1.0,
        colors=default_colors,
        plot_funcs=default_plot_funcs,
        highlight={}):
    """Plot weights
    """
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


def _plot_pwm_x_pos_matrix(pwm_x_pos, agg_vector):
    """take given arrays, save out to matrix, plot in R
    """
    # extract and save out to tmp file


    # then plot in R

    
    return


    

def visualize_debug(example_dict, prefix):
    """given an example dict, parse it for useful keys and plot
    if they exist
    """
    region_string = example_dict["example_metadata"][0].strip("\x00").split(";")[0].split("=")[1].replace(":", "-")
    region_prefix = "{}.{}".format(prefix, region_string)
    
    for key in example_dict.keys():

        # visualize importances
        if "importances" in key:
            plot_name = "{}.{}.pdf".format(region_prefix, key)
            print plot_name
            plot_weights(np.squeeze(example_dict[key]), plot_name)

        # visualize pwm scores
        if "pwm-scores" in key:
            pass
            

    return None


def visualize_clustered_h5_dataset_full(
        h5_file,
        data_key,
        cluster_key,
        cluster_ids_attr_key,
        colnames_attr_key,
        three_dims=False,
        cluster_columns=False,
        row_normalize=False,
        signal_normalize=False,
        large_view=False,
        use_raster=False,
        indices=[],
        viz_type="full"):
    """wrapper on nice heatmap2 plotting 
    """

    args = [
        h5_file,
        data_key,
        cluster_key,
        cluster_ids_attr_key,
        colnames_attr_key,
        1 if three_dims else 0,
        1 if cluster_columns else 0,
        1 if row_normalize else 0,
        1 if signal_normalize else 0,
        1 if large_view else 0,
        1 if use_raster else 0,
        ",".join(str(val) for val in indices)
    ]

    if viz_type == "full":
        script = "plot-h5.example_x_key.v2.R"
    elif viz_type == "cluster_map":
        script = "plot-h5.cluster_x_key.v2.R"
    elif viz_type == "multi_key":
        script = "plot-h5.keys_x_task.v2.R"
        
    r_cmd = "{} {}".format(
        script, " ".join(str(val) for val in args))
    logging.info(r_cmd)
    os.system(r_cmd)
    
    return None


# TODO deprecate
def visualize_h5_dataset(
        h5_file,
        dataset_key):
    """prduces a plot per cluster when given
    an aggreeated dataset where the cluster 
    is one of the dimensions

    this is specifically for the pwm x task
    """
    r_cmd= (
        "plot-h5.dataset.R {0} {1}").format(
            h5_file,
            dataset_key)
    print r_cmd
    os.system(r_cmd)
    
    return None


# TODO deprecate?
def visualize_agg_pwm_results(
        h5_file,
        pwm_scores_key,
        master_pwm_vector_key,
        pwm_names_attribute):
    """
    """
    r_cmd = (
        "plot-h5.cluster_x_aggkey.R {} {} {} {}").format(
            h5_file,
            pwm_scores_key,
            master_pwm_vector_key,
            pwm_names_attribute)
    print r_cmd
    os.system(r_cmd)

    return None


def visualize_agg_delta_logit_results(
        h5_file,
        delta_logits_key,
        motif_filter_key,
        task_indices,
        mut_pwm_names_attribute):
    """
    """
    r_cmd = (
        "plot-h5.cluster_x_deltalogits.R {} {} {} {} {}").format(
            h5_file,
            delta_logits_key,
            motif_filter_key,
            mut_pwm_names_attribute,
            ",".join(str(val) for val in task_indices))
    print r_cmd
    os.system(r_cmd)
    
    return None


def visualize_agg_dmim_adjacency_results(
        h5_file,
        dmim_adjacency_key,
        filter_vector_key,
        mut_pwm_names_attribute):
    """
    """
    r_cmd = (
        "plot-h5.cluster-task-pwm_x_pwm.R {} {} {} {}").format(
            h5_file,
            dmim_adjacency_key,
            filter_vector_key,
            mut_pwm_names_attribute)
    print r_cmd
    os.system(r_cmd)

    return None
