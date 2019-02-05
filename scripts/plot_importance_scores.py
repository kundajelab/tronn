#!/usr/bin/env python

"""
description: script to visualize importance scores

"""

import os
import sys
import h5py
import logging
import argparse

import numpy as np
import pandas as pd

from tronn.visualization import plot_weights
from tronn.util.scripts import setup_run_logs
from tronn.util.utils import DataKeys


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    parser.add_argument(
        "--data_file",
        help="h5 file with importance scores")
    parser.add_argument(
        "--keys", nargs="+",
        default=[
            DataKeys.WEIGHTED_SEQ_ACTIVE, #],
            DataKeys.MUT_MOTIF_WEIGHTED_SEQ,
            DataKeys.DFIM_SCORES],
        help="keys of importance scores to plot out")
    parser.add_argument(
        "--indices", nargs="+", default=[0], type=int,
        help="indices of examples to plot")
    parser.add_argument(
        "--regions", nargs="+", default=[],
        help="region names to plot (supersedes indices if used)")

    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    parser.add_argument(
        "--prefix",
        help="file prefix")
    
    # parse args
    args = parser.parse_args()

    return args


def _plot_importance_scores(
        h5_file, key, idx, prefix, user_pos_height=None, user_neg_height=None):
    """wrapper to handle importance scores
    """
    with h5py.File(h5_file, "r") as hf:
        data = hf[key][idx]

    # check max/min vals
    if user_pos_height is None:
        max_height = np.max(data)
    else:
        max_height = user_pos_height
        
    if user_neg_height is None:
        min_height = np.min(data)
    else:
        min_height = user_neg_height
        
    # check shape
    num_dims = len(data.shape)
    if num_dims == 3:
        # raw importance scores across tasks
        for task_idx in range(data.shape[0]):
            plot_name = "{}.{}.inferidx-{}.pdf".format(
                prefix, key.replace(".", "_"), task_idx)
            print plot_name
            plot_weights(
                data[task_idx], plot_name,
                user_pos_height=max_height, user_neg_height=min_height)
    if num_dims == 4:
        # importance scores for mut or other
        for mut_idx in range(data.shape[0]):
            for task_idx in range(data.shape[1]):
                plot_name = "{}.{}.mut_idx-{}.inferidx-{}.pdf".format(
                    prefix, key.replace(".", "_"), mut_idx, task_idx)
                print plot_name
                plot_weights(
                    data[mut_idx, task_idx], plot_name,
                    user_pos_height=max_height, user_neg_height=min_height)
        
    return None


def _get_max_min_heights_across_keys(h5_file, keys, idx):
    """get max and min across keys
    """
    neg_val = 0
    pos_val = 0
    for key in keys:
        with h5py.File(h5_file, "r") as hf:
            data = hf[key][idx]
        data_max = np.max(data)
        if data_max > pos_val:
            pos_val = data_max
        data_min = np.min(data)
        if data_min < neg_val:
            neg_val = data_min
            
    return neg_val, pos_val


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    main_prefix = "{}/{}".format(args.out_dir, args.prefix)

    # adjust if using region name
    if len(args.regions) > 0:
        raise ValueError, "not yet implemented!"
    
    # and plot results
    for idx in args.indices:
        prefix = "{}.regionidx-{}".format(main_prefix, idx)
        neg_val, pos_val = _get_max_min_heights_across_keys(
            args.data_file, args.keys, idx)
        for key in args.keys:
            _plot_importance_scores(
                args.data_file, key, idx, prefix,
                user_pos_height=pos_val,
                user_neg_height=neg_val)
    
    return None



if __name__ == "__main__":
    main()
