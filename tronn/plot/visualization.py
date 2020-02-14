# tools to ease plotting

# first, adjust params in matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.linewidth'] = 0.1
matplotlib.rcParams['xtick.labelsize'] = 4
matplotlib.rcParams['xtick.major.width'] = 0.1
matplotlib.rcParams['xtick.major.size'] = 1
matplotlib.rcParams['ytick.labelsize'] = 4
matplotlib.rcParams['ytick.major.width'] = 0.1
matplotlib.rcParams['ytick.major.size'] = 1

# imports
import matplotlib.pyplot as plt

import os
import logging

import numpy as np

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys


# heavily guided by aparent (https://github.com/johli/aparent) visualization code
FONTPROP = FontProperties(family="Arial", weight="bold")
FONTPROP = FontProperties(family="DejaVu Sans", weight="bold")


LETTERS = {
    "T" : TextPath((-0.305, 0), "T", size=1, prop=FONTPROP),
    "G" : TextPath((-0.384, 0), "G", size=1, prop=FONTPROP),
    "A" : TextPath((-0.35, 0), "A", size=1, prop=FONTPROP),
    "C" : TextPath((-0.366, 0), "C", size=1, prop=FONTPROP),
}

COLOR_SCHEME = {
    "A": "darkgreen",
    "C": "blue",
    "G": "orange",
    "T": "red"
}

IDX_TO_LETTER = {
    0: "A",
    1: "C",
    2: "G",
    3: "T"
}


def plot_letter(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):
    """plot letters at appropriate positions
    """
    globscale = 1.35
    text = LETTERS[letter]
    chosen_color = COLOR_SCHEME[letter]
    if color is not None :
        chosen_color = color
    
    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax != None:
        ax.add_artist(p)
        
    return p


def plot_pwm(
        array,
        plot_file):
    """plot pwm
    """
    # set up figure
    figsize=(array.shape[0]/20., 10/20.)
    f = plt.figure(figsize=figsize)

    # convert to entropy
    entropy = np.zeros(array.shape)
    entropy[array > 0] = array[array > 0] * -np.log2(array[array > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy
    
    # set up plot area
    height_base = 0.0
    logo_height = 1.0
    logo_ax = plt.gca()

    # go through each position and bp
    for j in range(array.shape[0]) :
        sort_index = np.argsort(array[j, :])
        
        for ii in range(0, 4) :
            i = sort_index[ii]
            nt_prob = array[j, i] * conservation[j]
            
            nt = ''
            if i == 0 :
                nt = 'A'
            elif i == 1 :
                nt = 'C'
            elif i == 2 :
                nt = 'G'
            elif i == 3 :
                nt = 'T'

            if ii == 0 :
                plot_letter(nt, j + 0.5, height_base, nt_prob * logo_height, logo_ax, color=None)
            else :
                prev_prob = np.sum(array[j, sort_index[:ii]] * conservation[j]) * logo_height
                plot_letter(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, logo_ax, color=None)

    plt.xlim((0, array.shape[0]))
    plt.ylim((0, 2))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    logo_ax.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2/10.)

    plt.tight_layout()
    plt.savefig(plot_file, transparent=True)
    
    return


def plot_weights(
        array,
        ax,
        height_min=-1,
        height_max=1,
        x_lab=False,
        y_lab=False,
        sig_array=None):
    """plot weights
    """
    # array is (seqlen, 4)
    height_base = 0.0
    
    # for each position
    # TODO include option to plot base pairs in gray
    for pos_idx in range(array.shape[0]):
        letter_idx = np.argmax(np.abs(array[pos_idx]))
        val = array[pos_idx, letter_idx]
        letter_str = IDX_TO_LETTER[letter_idx]
        if sig_array is not None:
            if np.sum(sig_array[pos_idx]) != 0:
                color = None
            else:
                color="lightgrey"
        else:
            color = None
        plot_letter(letter_str, pos_idx, height_base, val, ax, color=color)

    # adjust plot
    plt.sca(ax)
    plt.xlim((0, array.shape[0]))
    plt.ylim((height_min, height_max))
    if not x_lab:
        plt.xticks([], [])
    ax.yaxis.tick_right()
    if not y_lab:
        plt.yticks([], [])
    #plt.axis('off')
    ax.axhline(y=0.00001 + height_base, color='lightgrey', linestyle='-', linewidth=0.001)

    return None



def plot_weights_group(array, plot_file, sig_array=None):
    """
    """
    # assume array is of form (task, seqlen, 4)
    num_rows = array.shape[0]
    assert len(array.shape) == 3
    
    # calculate max/min height
    max_val = np.max(array)
    min_val = np.min(array)

    # calculate ratio and adjust to fit in page width
    # assumes that 160 bps should be 6in wide
    # maintain same height
    desired_height = 2.25
    width_to_height_factor = 6
    width_height_ratio = array.shape[1] / float(array.shape[0])
    #plot_height = height_to_width_factor * width_height_ratio * desired_width
    #desired_width = 6.0
    desired_width = desired_height * width_height_ratio / width_to_height_factor
    
    # set up plot
    f, ax = plt.subplots(num_rows, 1, figsize=(desired_width, desired_height))
    for row_idx in range(num_rows):
        x_lab = False
        y_lab = False
        if row_idx == 0:
            y_lab = True
        if row_idx == (num_rows - 1):
            x_lab = True
        if sig_array is not None:
            plot_sig = sig_array[row_idx]
        else:
            plot_sig = None
        plot_weights(
            array[row_idx], ax[row_idx],
            height_min=min_val, height_max=max_val,
            x_lab=x_lab, y_lab=y_lab,
            sig_array=None)
            #sig_array=plot_sig)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(plot_file, transparent=True)
    plt.close("all")
    
    return plt


def plot_weights_heatmap(array, plot_file):
    """plot as heatmap with max vals up top?
    """
    


    
    return




def scale_scores(orig_scores, match_scores, scale_axis=0):
    """main use case: you have original importance scores
    that are unnormalized, but want to match the normalized scores.
    assumes that normalization was LINEAR, finds top position and uses
    that score to linearly match scores.
    """
    assert np.all(orig_scores.shape == match_scores.shape), "{} and {}".format(orig_scores.shape, match_scores.shape)

    across_axes = range(len(orig_scores.shape))
    across_axes.remove(scale_axis)
    
    # find max scores
    orig_max_vals = np.max(orig_scores, axis=tuple(across_axes), keepdims=True)
    match_max_vals = np.max(match_scores, axis=tuple(across_axes), keepdims=True)
    
    # get ratios
    ratios = np.divide(match_max_vals, orig_max_vals)

    # apply
    final_scores = np.multiply(ratios, orig_scores)

    return final_scores


