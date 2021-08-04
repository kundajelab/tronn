# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import json
import h5py
import glob
import logging

import networkx as nx
import numpy as np
import pandas as pd

from itertools import permutations
from tronn.datalayer import H5DataLoader
from tronn.interpretation.inference import run_inference
from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.h5_utils import copy_h5_datasets
from tronn.util.formats import write_to_json
from tronn.util.utils import DataKeys


def _run_calculations(h5_file, args):
    """do post process calculations
    """
    logging.info("NOTE: these calculations rely on an ORDERED queue (for the reshape)")
    with h5py.File(h5_file, "r") as hf:
        grammar_strings = hf["grammar.string"][:,0]
        logits = hf[DataKeys.LOGITS][:]
        dists = hf["simul.pwm.dist"][:]

    # calculate num syntaxes and num positions
    num_syntax = 2**len(args.grammar_pwms) * len(
        [i for i in permutations(range(len(args.grammar_pwms)))])
    start_grammar_range = max(args.grammar_range[0], args.min_spacing)
    num_positions = len(
        range(start_grammar_range, args.grammar_range[1], args.pwm_stride))

    # reshape logits: {N, logit} -> {syntax, pos, sample, 2, logit}
    num_positions = 108 # debug!
    logits = np.reshape(
        logits, (num_syntax, num_positions, args.num_samples, 2, logits.shape[1]))
    logFC = logits[:,:,:,0] - logits[:,:,:,1]
    logFC_mean = np.mean(logFC, axis=2) # {syntax, pos, logit}

    # reshape distances: {N, dist} -> {syntax, pos, sample*2} and check!
    dists = np.reshape(
        dists, (num_syntax, num_positions, -1))
    check_dists = np.equal(dists, np.expand_dims(dists[:,:,0], axis=-1))
    assert np.all(check_dists)
    dists = dists[0,:,0] # {pos}

    # reshape groups: {N} -> {syntax, pos*sample*2}
    grammar_strings = np.reshape(
        grammar_strings, (num_syntax, -1))
    check_strings = np.equal(grammar_strings, np.expand_dims(grammar_strings[:,0], axis=-1))
    assert np.all(check_strings)
    grammar_strings = grammar_strings[:,0] # {syntax}
    
    # now calculate for each syntax, for each logit, a smoothed result
    window_len = 10
    logFC_mean_extended = np.concatenate([
        logFC_mean[:,window_len-1:0:-1],
        logFC_mean,
        logFC_mean[:,-2:-window_len-1:-1]], axis=1)

    #filt = np.ones(window)
    #filt = np.bartlett(window) # l1 like
    filt = np.blackman(window_len) # more l2 like
    logFC_mean_smooth = np.apply_along_axis(
        lambda m: np.convolve(m, filt/filt.sum(), mode="valid"),
        axis=1,
        arr=logFC_mean_extended)
    logFC_mean_smooth = logFC_mean_smooth[:,(window_len/2-1):-(window_len/2)]

    # for this get a super smoothed version (use to access best syntax and position)
    window_len = 24
    logFC_mean_extended = np.concatenate([
        logFC_mean[:,window_len-1:0:-1],
        logFC_mean,
        logFC_mean[:,-2:-window_len-1:-1]], axis=1)

    filt = np.hamming(window_len) # l1 like
    logFC_mean_supersmooth = np.apply_along_axis(
        lambda m: np.convolve(m, filt/filt.sum(), mode="valid"),
        axis=1,
        arr=logFC_mean_extended)
    logFC_mean_supersmooth = logFC_mean_supersmooth[:,(window_len/2-1):-(window_len/2)]
    
    # want to save the results as {task, syntax, dist} with logFC (over background) in the cell
    group_key = "simul.calcs"
    with h5py.File(h5_file, "a") as hf:
        # save raw scores
        score_key = "{}/simul.scores".format(group_key)
        if hf.get(score_key) is not None:
            del hf[score_key]
        hf.create_dataset(score_key, data=logFC_mean)
        hf[score_key].attrs["grammar.string"] = grammar_strings
        hf[score_key].attrs["dists"] = dists
        
        # save smoothed scores
        score_key = "{}/simul.scores.smooth".format(group_key)
        if hf.get(score_key) is not None:
            del hf[score_key]
        hf.create_dataset(score_key, data=logFC_mean_smooth)
        hf[score_key].attrs["grammar.string"] = grammar_strings
        hf[score_key].attrs["dists"] = dists

        # save super smoothed scores
        score_key = "{}/simul.scores.smooth.high".format(group_key)
        if hf.get(score_key) is not None:
            del hf[score_key]
        hf.create_dataset(score_key, data=logFC_mean_supersmooth)
        hf[score_key].attrs["grammar.string"] = grammar_strings
        hf[score_key].attrs["dists"] = dists
        
    return


def run(args):
    """Scan motifs from a PWM file
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running motif scan")
    
    # if ensemble, require a prediction sample
    if args.model["name"] == "ensemble":
        args.model["params"]["prediction_sample"] = args.prediction_sample
        args.inference_params["prediction_sample"] = args.prediction_sample
        
    # run inference
    inference_files = run_inference(args)

    # build smoothed signal on the results
    for inference_file in inference_files:
        _run_calculations(inference_file, args)
    
    # add in PWM names to the datasets
    for inference_file in inference_files:
        add_pwm_names_to_h5(
            inference_file,
            [pwm.name for pwm in args.pwm_list],
            other_keys=[DataKeys.FEATURES])
    
    # put the files into a dataloader
    results_data_log = "{}/dataset.{}.json".format(args.out_dir, args.subcommand_name)
    results_data_loader = H5DataLoader(
        data_dir=args.out_dir, data_files=inference_files, fasta=args.fasta)
    dataset = results_data_loader.describe()
    dataset.update({
        "targets": args.targets,
        "target_indices": args.target_indices})
    write_to_json(dataset, results_data_log)

    # save out relevant inference run details for downstream runs
    infer_log = "{}/infer.{}.json".format(args.out_dir, args.subcommand_name)
    infer_vals = {
        "infer_dir": args.out_dir,
        "model_json": args.model_json,
        "targets": args.targets,
        "inference_targets": args.inference_targets,
        "target_indices": args.target_indices,
        "pwm_file": args.pwm_file}
    write_to_json(infer_vals, infer_log)

    # and now plot out the results
    plot_cmd = "plot-h5.simul_results.R {} {} {} {} {} {}".format(
        "{}/{}.{}.h5".format(args.out_dir, args.prefix, args.subcommand_name),
        "simul.pwm.dist",
        DataKeys.LOGITS,
        "grammar.string",
        "\"simul.calcs/simul.scores.smooth.high\"",
        "{}/{}".format(args.out_dir, os.path.basename(args.grammar).split(".gml")[0]))
    print plot_cmd
    os.system(plot_cmd)
    
    return None

