# tools to ease plotting

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

import numpy as np

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



def visualize_clustered_h5_dataset_full(
        h5_file,
        cluster_key,
        dataset_key,
        cluster_col=0,
        remove_final_cluster=True,
        row_normalize=False,
        signal_normalize=False,
        cluster_columns=False,
        use_raster=True,
        indices=[]):
    """wrapper on nice heatmap2 plotting 
    """
    r_cmd = (
        "plot-h5.example_x_key.R {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}").format(
            h5_file,
            cluster_key,
            cluster_col,
            1 if remove_final_cluster else 0,
            1 if row_normalize else 0,
            1 if signal_normalize else 0,
            1 if cluster_columns else 0,
            1 if use_raster else 0,
            dataset_key,
            ",".join(str(val) for val in indices))
    print r_cmd
    os.system(r_cmd)
    
    return None


def visualize_aggregated_h5_datasets(
        h5_file,
        cluster_key,
        dataset_keys,
        index_sets,
        cluster_col=0,
        remove_final_cluster=True,
        normalize=False):
    """wrapper on nice heatmap2 plotting
    """
    dataset_and_indices = zip(dataset_keys, index_sets)
    data_strings = [
        "{}={}".format(
            key,
            ",".join(str(val)for val in vals))
        for key, vals in dataset_and_indices]
    
    r_cmd = (
        "plot-h5.keys_x_task.R {0} {1} {2} {3} {4}").format(
            h5_file,
            cluster_key,
            cluster_col,
            1 if remove_final_cluster else 0,
            " ".join(data_strings))
    print r_cmd
    os.system(r_cmd)
    
    return None


def visualize_datasets_by_cluster_map(
        h5_file,
        cluster_key,
        dataset_key,
        cluster_col=0,
        remove_final_cluster=True,
        cluster_rows=True,
        normalize=False,
        indices=[]):
    """wrapper on heatmap2 plotting
    produces a heatmap of aggregated vals per cluster
    """
    r_cmd = (
        "plot-h5.cluster_x_key.R {0} {1} {2} {3} {4} {5} {6} {7}").format(
            h5_file,
            cluster_key,
            cluster_col,
            1 if remove_final_cluster else 0,
            1 if normalize else 0,
            1 if cluster_rows else 0,
            dataset_key,
            ",".join(str(val) for val in indices))
    print r_cmd
    os.system(r_cmd)

    return None


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


def visualize_h5_dataset_by_cluster(
        h5_file,
        dataset_key):
    """prduces a plot per cluster when given
    an aggreeated dataset where the cluster 
    is one of the dimensions

    this is specifically for the pwm x task
    """
    r_cmd= (
        "plot-h5.key_x_task.R {0} {1} pwm_names").format(
            h5_file,
            dataset_key)
    print r_cmd
    os.system(r_cmd)
    
    return None




def visualize_clustering_results(
        h5_file,
        cluster_key,
        inference_task_indices,
        visualize_task_indices,
        visualize_signals,
        cluster_col=0,
        soft_cluster_key=None,
        remove_final_cluster=True):
    """visualize the results
    """
    dataset_keys = [
        "pwm-scores.taskidx-{}".format(i)
        for i in inference_task_indices]
    visualize_task_indices = [inference_task_indices] + visualize_task_indices
    num_vis_sets = len(visualize_task_indices)

    # (1) visualize example x key
    # do this for inference tasks and visualize tasks
    # set up keys, indices, etc for visualizing full example space
    keys = list(dataset_keys)
    keys += ["probs" for i in xrange(num_vis_sets)]
    keys += ["labels" for i in xrange(num_vis_sets)]
    
    indices = [[] for i in xrange(len(dataset_keys))]
    indices += [visualize_task_indices[i] for i in xrange(num_vis_sets)] * 2

    row_normalizations = [True for i in xrange(len(dataset_keys))]
    row_normalizations += [False for i in xrange(num_vis_sets)] * 2

    signal_normalizations = [False for i in xrange(len(dataset_keys))]
    signal_normalizations += [False for i in xrange(num_vis_sets)] * 2

    # add in signals
    for key in visualize_signals.keys():
        keys.append(key)
        indices.append(visualize_signals[key][0])
        row_normalizations.append(False)
        signal_normalizations.append(True)
        
    for i in xrange(len(keys)):

        if "pwm-score" in keys[i]:
            use_raster = True
            cluster_columns = True
        else:
            use_raster = False
            cluster_columns = False
        
        visualize_clustered_h5_dataset_full(
            h5_file,
            cluster_key,
            keys[i],
            cluster_col=cluster_col, # must always be single column
            remove_final_cluster=remove_final_cluster,
            row_normalize=row_normalizations[i],
            signal_normalize=signal_normalizations[i],
            cluster_columns=cluster_columns,
            use_raster=use_raster,
            indices=indices[i])

    # here adjust cluster key after generating example graphs
    if soft_cluster_key is not None:
        cluster_key = soft_cluster_key
        cluster_col = -1
        
    # (2) visualize (per cluster) aggregated x keys (multiple)
    # set up keys
    keys = [["probs", "labels"] for i in xrange(num_vis_sets)]
    indices = [[visualize_task_indices[i], visualize_task_indices[i]]
               for i in xrange(num_vis_sets)]

    # add in signals
    for key in visualize_signals.keys():
        label_key = visualize_signals[key][1].get("label_key", None)
        if label_key is None:
            dataset_keys = ["probs", key]
        else:
            dataset_keys = ["probs", label_key, key]
        key_indices = visualize_signals[key][0]
        key_indices = [key_indices for i in xrange(len(dataset_keys))]

        keys.append(dataset_keys)
        indices.append(key_indices)
    
    for i in xrange(len(keys)):
        visualize_aggregated_h5_datasets(
            h5_file,
            cluster_key,
            keys[i],
            indices[i],
            cluster_col=cluster_col,
            remove_final_cluster=remove_final_cluster)
    
    # (3) set up keys for cluster map
    keys = ["probs" for i in xrange(num_vis_sets)]
    keys += ["labels" for i in xrange(num_vis_sets)]
    
    indices = [visualize_task_indices[i] for i in xrange(num_vis_sets)] * 2
    
    for key in visualize_signals.keys():
        keys.append(key)
        indices.append(visualize_signals[key][0])

    for i in xrange(len(keys)):
        visualize_datasets_by_cluster_map(
            h5_file,
            cluster_key,
            keys[i],
            indices=indices[i],
            cluster_col=cluster_col,
            remove_final_cluster=remove_final_cluster)
        
    return None

