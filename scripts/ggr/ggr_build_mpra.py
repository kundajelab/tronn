#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import re
import sys
import h5py
import argparse

import numpy as np
import pandas as pd
import networkx as nx

from scipy.stats import zscore
from tronn.interpretation.combinatorial import setup_combinations
from tronn.util.scripts import setup_run_logs
from tronn.util.utils import DataKeys

def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # required args
    parser.add_argument(
        "--grammar_summary",
        help="grammar summary file")
    parser.add_argument(
        "--synergy_main_dir",
        help="folder where synergy dirs reside")
    
    # out
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")

    # parse args
    args = parser.parse_args()

    return args


def save_sequences(h5_file, indices, df, left_clip=420, right_clip=580, prefix="id"):
    """save sequences to a df
    """
    keys = [
        DataKeys.SYNERGY_DIFF_SIG,
        DataKeys.SYNERGY_DIST,
        DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT,
        DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT,
        DataKeys.MUT_MOTIF_LOGITS,
        DataKeys.SEQ_METADATA,
        "ATAC_SIGNALS.NORM",
        "H3K27ac_SIGNALS.NORM",
        DataKeys.LOGITS_NORM,
        DataKeys.GC_CONTENT,
        "{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)]

    # first get reshape params
    num_sequences = indices.shape[0]
    with h5py.File(h5_file, "r") as hf:
        num_combos = hf[DataKeys.MUT_MOTIF_ORIG_SEQ].shape[1]
        num_muts = int(np.sqrt(num_combos))

    # go through keys and extract relevant results
    with h5py.File(h5_file, "r") as hf:
        for key_idx in range(len(keys)):
            key = keys[key_idx]
            key_results = hf[key][:][indices]

            # check dims and flatten accordingly
            if len(key_results.shape) <= 2:
                end_dims = list(key_results.shape)[1:]
                key_results = np.stack([key_results]*num_combos, axis=1)
            elif len(key_results.shape) == 3:
                end_dims = list(key_results.shape)[2:]
            else:
                raise ValueError, "key does not match shape, is {}".format(key_results.shape)
            key_results = np.reshape(key_results, [-1]+end_dims)

            # if bool convert to int
            if key_results.dtype == np.bool:
                key_results = key_results.astype(int)

            # other adjustments
            if key == DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT:
                key_results = key_results.astype(int)
            if key == DataKeys.SYNERGY_DIST:
                key_results = key_results.astype(int)
                
            # create dataframe
            key_df = pd.DataFrame(key_results)
            if key_df.shape[1] > 1:
                key_df = pd.DataFrame({key:key_df.astype(str).apply(",".join, axis=1)})
            else:
                key_df.columns = [key]
                
            # append
            if key_idx == 0:
                all_results = key_df
            else:
                all_results[key] = key_df[key]

    # adjust sequence
    all_results.insert(
        0, "sequence.mpra",
        all_results["{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)].str[left_clip:right_clip])
    all_results = all_results.drop(columns=["{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)])
    
    # set up combos
    combinations = setup_combinations(num_muts)
    combinations = np.swapaxes(combinations, 0, 1)
    combinations = np.stack([combinations]*num_sequences, axis=0)
    combinations = np.reshape(combinations, [-1, num_muts])
    combo_df = pd.DataFrame(combinations).astype(int)
    combo_df["combos"] = combo_df.astype(str).apply(",".join, axis=1)
    all_results.insert(0, "combos", combo_df["combos"])

    # set up indices
    rep_indices = np.stack([indices]*num_combos, axis=1)
    rep_indices = np.reshape(rep_indices, [-1] )
    all_results.insert(0, "example_fileidx", rep_indices)

    # set up a unique id
    all_results.insert(
        0, "unique_id",
        prefix + "." + all_results["example_fileidx"].astype(str) + "." + all_results["combos"].str.replace(",", ""))

    return all_results


def extract_sequences(args):
    """sample sequences
    """
    diff_sample_num = 20
    nondiff_proximal_sample_num = 10
    nondiff_distal_sample_num = 10
    
    grammar_summary = pd.read_table(args.grammar_summary, index_col=0)

    for grammar_idx in range(grammar_summary.shape[0]):
        # set seed each time you run a new grammar
        np_seed = np.random.seed(1337)
        grammar_file = grammar_summary.iloc[grammar_idx]["filename"]
        synergy_dir = os.path.basename(grammar_file).split(".gml")[0]
        
        synergy_file = "{}/{}/ggr.synergy.h5".format(args.synergy_main_dir, synergy_dir)
        synergy_pwm_names_file = "{}/{}/ggr.synergy.pwms.order.txt".format(
            args.synergy_main_dir, synergy_dir)
        print synergy_file

        # open synergy file to subsample sequences
        try:
            with h5py.File(synergy_file, "r") as hf:
                # extract
                sequences = hf["{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)][:] # {N, combos, 1}
                distances = hf[DataKeys.SYNERGY_DIST][:] # {N}
                diffs_sig = hf[DataKeys.SYNERGY_DIFF_SIG][:] # {N, logit}
                max_dist = hf[DataKeys.SYNERGY_MAX_DIST][()]
        except IOError:
            continue
            
        # get diff
        min_diff = diffs_sig.shape[1]
        num_tasks_diff = np.sum(diffs_sig, axis=1)
        while True:
            print min_diff
            diffs_sig = np.greater_equal(num_tasks_diff, min_diff) # {N}
            diff_indices = np.where(diffs_sig)[0]
            try:
                diff_sample_indices = diff_indices[
                    np.random.choice(diff_indices.shape[0], diff_sample_num, replace=False)]
                break
            except ValueError:
                min_diff -= 1
                
        diff_bool = np.zeros(distances.shape)
        diff_bool[diff_sample_indices] = 1
        #diff_df = save_sequences(synergy_file, diff_sample_indices, args.out_dir, prefix=synergy_dir)
        
        # get nondiff, less than dist
        nondiff = np.logical_not(diffs_sig) # {N}
        nondiff_proximal_indices = np.where(np.logical_and(nondiff, distances < max_dist))[0]
        nondiff_proximal_sample_indices = nondiff_proximal_indices[
            np.random.choice(nondiff_proximal_indices.shape[0], nondiff_proximal_sample_num, replace=False)]
        nondiff_proximal_bool = np.zeros(distances.shape)
        nondiff_proximal_bool[nondiff_proximal_sample_indices] = 1
        #nondiff_proximal_df = save_sequences(synergy_file, nondiff_proximal_sample_indices, args.out_dir, prefix=synergy_dir)
        
        # get nondiff, greater than dist
        nondiff_distal_indices = np.where(np.logical_and(nondiff, distances >= max_dist))[0]
        nondiff_distal_sample_indices = nondiff_distal_indices[
            np.random.choice(nondiff_distal_indices.shape[0], nondiff_distal_sample_num, replace=False)]
        nondiff_distal_bool = np.zeros(distances.shape)
        nondiff_distal_bool[nondiff_distal_sample_indices] = 1
        #nondiff_distal_df = save_sequences(synergy_file, nondiff_distal_sample_indices, args.out_dir, prefix=synergy_dir)

        # and mark out the ones chosen in the synergy file
        all_sample_indices = np.concatenate(
            [diff_sample_indices,
             nondiff_proximal_sample_indices,
             nondiff_distal_sample_indices], axis=0)
        all_bool = np.zeros(distances.shape)
        all_bool[all_sample_indices] = 1
        mpra_sample_df = save_sequences(synergy_file, all_sample_indices, args.out_dir, prefix=synergy_dir)
        #mpra_sample_df = mpra_sample_df.sort_values(DataKeys.SYNERGY_DIFF_SIG)
        mpra_sample_df.to_csv("{}/test.results.txt".format(args.out_dir), sep="\t")
        quit()
        
        
        # open synergy file to subsample sequences
        with h5py.File(synergy_file, "a") as hf:
            if hf.get("mpra.sample.diff") is not None:
                del hf["mpra.sample.diff"]
            hf.create_dataset("mpra.sample.diff", data=diff_bool)
            
            if hf.get("mpra.sample.nondiff.proximal") is not None:
                del hf["mpra.sample.nondiff.proximal"]
            hf.create_dataset("mpra.sample.nondiff.proximal", data=diff_bool)
            
            if hf.get("mpra.sample.nondiff.distal") is not None:
                del hf["mpra.sample.nondiff.distal"]
            hf.create_dataset("mpra.sample.nondiff.distal", data=diff_bool)
            
            if hf.get("mpra.sample.all") is not None:
                del hf["mpra.sample.all"]
            hf.create_dataset("mpra.sample.all", data=all_bool)
            
        # and can plot out here
        out_prefix = "{}/{}".format(args.out_dir, synergy_dir)
        plot_cmd = "plot-h5.synergy_results.2.R {} {} {} {} {} {} {} {}".format(
            synergy_file,
            DataKeys.SYNERGY_SCORES,
            DataKeys.SYNERGY_DIFF,
            DataKeys.SYNERGY_DIFF_SIG,
            DataKeys.SYNERGY_DIST,
            DataKeys.SYNERGY_MAX_DIST,
            out_prefix,
            "mpra.sample.all")
        print plot_cmd
        os.system(plot_cmd)

        # debug
        if grammar_idx >= 3:
            quit()
            
    return None


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    # make reproducible

    
    # read out sequences from synergy files
    # for each grammar file, go back to synergy file and read out sequences {N, mutM, 1} string
    # sample randomly <- split diff and non diff, and sample half and half from each side
    # for the non diff side, look at distance cutoff and sample half and half from each side
    extract_sequences(args)

    # would be good to plot out the subsamples...
    
    return None



if __name__ == "__main__":
    main()
