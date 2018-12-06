#!/usr/bin/env python

"""
description: script to call differential motifs
between a background set and foreground set

Key params used here:

- qval thresh (FDR)
- reduce type (any, all, or min sig across tasks)

Think carefully about

- scores used (default is hits)
- background set used

"""

import os
import sys
import h5py
import json
import argparse
import logging

import numpy as np

from multiprocessing import Pool

from tronn.datalayer import H5DataLoader
from tronn.interpretation.motifs import test_differential_motifs
from tronn.stats.nonparametric import threshold_by_qvalues
from tronn.util.h5_utils import AttrKeys
from tronn.util.scripts import setup_run_logs
from tronn.util.scripts import parse_multi_target_selection_strings
from tronn.util.scripts import load_selected_targets
from tronn.util.utils import DataKeys


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="call differential motifs")

    # required args
    parser.add_argument(
        "--foreground_files", nargs="+",
        required=True,
        help="data files for foreground calculations")
    parser.add_argument(
        "--background_files", nargs="+",
        required=True,
        help="data files for background calculations")
    parser.add_argument(
        "--foregrounds", nargs="+",
        required=True,
        help="foregrounds to test")
    parser.add_argument(
        "--background", nargs="+",
        required=True,
        help="background set (single background for ALL foregrounds)")
    parser.add_argument(
        "--inference_json",
        required=True,
        help="the json produced from scanmotifs (contains inference_targets and model_json)")
    parser.add_argument(
        "--targets", nargs="+",
        help="if this is defined, it overrides the targets in the inference json")
    
    # other settings
    parser.add_argument(
        "--scores_key", default=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        help="scores dataset key in h5 file")
    parser.add_argument(
        "--gc_key", default=DataKeys.GC_CONTENT,
        help="GC content dataset key in h5 file")
    parser.add_argument(
        "--qval_thresh", default=0.10, type=float,
        help="qvalue threshold")
    parser.add_argument(
        "--reduce_type", nargs="+", default=["any"],
        help="type of reduction across the inference tasks (any, all, or min)")
    parser.add_argument(
        "--num_threads", default=1, type=int,
        help="num_threads for parallelization")
    
    # outputs
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="outputs directory")
    parser.add_argument(
        "--out_file", default="pvals.h5",
        help="out file for pvals")

    # parse
    args = parser.parse_args()
    
    return args


def _run_differential_test_parallel(args):
    """wrapper for test
    """
    foreground_data = args[0]
    background_data = args[1]
    inference_targets = args[2]
    task_idx = args[3]
    scores_key = args[4]
    gc_key = args[5]
    
    # get foreground for task
    task_foreground_scores = foreground_data[scores_key][:,task_idx]

    # select specific background for task
    task_background_indices = np.where(
        inference_targets[:,task_idx] > 0)[0]
    task_background_scores = background_data[scores_key][
        task_background_indices][:,task_idx]
    task_background_gc = background_data[gc_key][task_background_indices]
    
    # run hits version
    pvals = test_differential_motifs(
        task_foreground_scores != 0,
        task_background_scores != 0,
        foreground_data[gc_key],
        task_background_gc)

    return pvals


def main():
    """call differential motifs with foreground and background
    """
    args = parse_args()
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    # load inference information and model
    with open(args.inference_json, "r") as fp:
        args.inference = json.load(fp)
    
    # make background data loader
    logging.info("setting up background data")
    background_targets = parse_multi_target_selection_strings(args.background)
    background_data_loader = H5DataLoader(data_files=args.background_files)

    # get background subset indices. if multiple selection strings,
    # background will check for ANY (instead of ALL)
    logging.info("loading background targets")
    selected_targets = []
    for targets, params in background_targets:
        logging.info("loading {} with {}".format(targets, params))
        reduced_targets = load_selected_targets(background_data_loader, targets, params)
        selected_targets.append(reduced_targets)
    selected_targets = np.concatenate(selected_targets, axis=1)
    background_targets = np.any(selected_targets, axis=1)
    background_indices = np.where(background_targets)[0]
    
    # load background data - gc_key, scores
    logging.info("loading background data")
    background_data = background_data_loader.load_datasets(
        [args.scores_key, args.gc_key])
    for key in background_data.keys():
        background_data[key] = background_data[key][background_indices]
    with h5py.File(background_data_loader.data_files[0], "r") as hf:
        background_pwm_names = hf[args.scores_key].attrs["pwm_names"]

    # load background inference targets (that were used in scanmotifs)
    if args.targets is not None:
        inference_targets = parse_multi_target_selection_strings(args.targets)
    else:
        inference_targets = [(target, {}) for target in args.inference["targets"]]
    selected_targets = []
    for targets, params in inference_targets:
        found_targets = load_selected_targets(
            background_data_loader, targets, {"reduce_type": "none"})
        selected_targets.append(found_targets)
    inference_targets = np.concatenate(selected_targets, axis=1)
    inference_target_indices = args.inference["inference_targets"]
    if len(inference_target_indices) > 0:
        inference_targets = inference_targets[:,inference_target_indices]
    inference_targets = inference_targets[background_indices]
    
    # iterate through foregrounds
    foregrounds = parse_multi_target_selection_strings(args.foregrounds)
    hits_pvals = np.ones([len(foregrounds)] + list(background_data[args.scores_key].shape[1:]))
    for foreground_idx in range(len(foregrounds)):
        foreground = foregrounds[foreground_idx]
        logging.info("running foreground {}".format(args.foregrounds[foreground_idx]))

        # setup data loader
        # TODO is it worth saving out a new dataset json? yes - set up
        foreground_data_loader = H5DataLoader(
            data_files=args.foreground_files)
        with h5py.File(foreground_data_loader.data_files[0], "r") as hf:
            foreground_pwm_names = hf[args.scores_key].attrs["pwm_names"]
        assert np.array_equal(background_pwm_names, foreground_pwm_names)
        
        # select targets
        selected_targets = load_selected_targets(
            foreground_data_loader, foreground[0], foreground[1])
        foreground_indices = np.where(selected_targets)[0]
        
        # load in relevant data
        foreground_data = foreground_data_loader.load_datasets(
            [args.scores_key, args.gc_key])
        for key in foreground_data.keys():
            foreground_data[key] = foreground_data[key][foreground_indices] # scores {N, task, ...}

        # set up args to run for each task
        run_args_list = [
            (foreground_data, background_data, inference_targets, task_idx, args.scores_key, args.gc_key)
            for task_idx in range(foreground_data[args.scores_key].shape[1])]

        # set up pool and run
        pool = Pool(args.num_threads)
        pvals = pool.map(_run_differential_test_parallel, run_args_list)
        pvals = np.stack(pvals, axis=0)
        pool.close()
        pool.join()

        # save out
        hits_pvals[foreground_idx] = pvals

    # calc qval thresholded
    hits_qval_thresholded = threshold_by_qvalues(hits_pvals, qval_thresh=args.qval_thresh)
    overall_thresholded = hits_qval_thresholded

    # reduce
    if args.reduce_type[0] == "any":
        overall_thresholded = np.any(overall_thresholded, axis=1)
    elif args.reduce_type[0] == "all":
        overall_thresholded = np.all(overall_thresholded, axis=1)
    elif args.reduce_type[0] == "min":
        num_sig = np.sum(overall_thresholded == True, axis=1)
        overall_thresholded = num_sig >= eval(args.reduce_type[1])
    else:
        raise ValueError, "reduce type not recognized!"
        
    logging.info("Significant motifs per foreground: {}".format(
        " ".join([str(i) for i in np.sum(overall_thresholded, axis=1).tolist()])))
    
    # and save out
    pvals_file = "{}/{}".format(args.out_dir, args.out_file)
    _RAW_PVALS_GROUP = "pvals"
    hits_pvals_key = "pvals"
    sig_key = "sig"
    
    results = {
        hits_pvals_key: hits_pvals,
        sig_key: overall_thresholded}

    # save out each foreground separately - easier to handle downstream
    for foreground_idx in range(len(foregrounds)):
        foreground = args.foregrounds[foreground_idx]
        foreground_string = foreground.replace("=", "-")
        foreground_string = foreground_string.replace("::", ".")
        foreground_string = foreground_string.replace(",", "-")
        logging.info("saving foreground results as {}".format(foreground_string))

        for key in results.keys():
            with h5py.File(pvals_file, "a") as hf:
                out_key = "{}/{}/{}".format(
                    _RAW_PVALS_GROUP, foreground_string, key)
                if hf.get(out_key) is not None:
                    del hf[out_key]
                hf.create_dataset(out_key, data=results[key][foreground_idx])
                hf[out_key].attrs[AttrKeys.TASK_INDICES] = inference_target_indices
                hf[out_key].attrs[AttrKeys.PWM_NAMES] = background_pwm_names

        # debug
        sig_pwms = [background_pwm_names[i] for i in np.where(results[sig_key][foreground_idx]!=0)[0]]
        logging.debug(sig_pwms)

    with h5py.File(pvals_file, "a") as hf:
        hf[_RAW_PVALS_GROUP].attrs["foregrounds"] = args.foregrounds
    
    return



if __name__ == "__main__":
    main()
