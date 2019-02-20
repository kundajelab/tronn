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


def save_sequences(sequences, indices, df, left_clip=420, right_clip=580):
    """save sequences to a df
    """
    # make combinations into 1 col (won't use for downstream anyways)
    # index positions of mutations, make into 1 col
    # dists, make into 1 col
    # region metadata
    
    # TODO: also keep: index positions of mutations, dist between them? region metadata
    # need to read out pwm score strength also?
    # make a unique prefix
    
    
    # get sequences
    sequences = sequences[indices]
    
    # get combos
    num_muts = int(np.sqrt(sequences.shape[1]))
    combinations = setup_combinations(num_muts)
    assert combinations.shape[1] == sequences.shape[1]
    combinations = np.swapaxes(combinations, 0, 1)
    combinations = np.stack([combinations]*sequences.shape[0], axis=0)

    # set up indices
    indices = np.stack([indices]*combinations.shape[1], axis=1)
    
    # flatten {N, comb, 1} -> {N*comb}
    sequences_flattened = np.reshape(sequences, (-1, 1))
    combinations_flattened = np.reshape(combinations, (-1, num_muts))
    indices_flattened = np.reshape(indices, (-1, 1))
    
    # convert to pandas df and conct
    sequences_df = pd.DataFrame(sequences_flattened, columns=["sequence_string"])
    sequences_df["sequence_string_active"] = sequences_df["sequence_string"].str[left_clip:right_clip]
    sequences_df["index"] = indices_flattened
    for col_idx in range(combinations_flattened.shape[1]):
        sequences_df["motif_{}".format(col_idx)] = combinations_flattened[:,col_idx].astype(int)

    if df is None:
        df = sequences_df
    else:
        # join?
        pass
        
    import ipdb
    ipdb.set_trace()

    return


def extract_sequences(args):
    """sample sequences
    """
    diff_sample_num = 20
    nondiff_proximal_sample_num = 10
    nondiff_distal_sample_num = 10
    
    grammar_summary = pd.read_table(args.grammar_summary, index_col=0)

    for grammar_idx in range(grammar_summary.shape[0]):
        grammar_file = grammar_summary.iloc[grammar_idx]["filename"]
        synergy_dir = os.path.basename(grammar_file).split(".gml")[0]
        
        synergy_file = "{}/{}/ggr.synergy.h5".format(args.synergy_main_dir, synergy_dir)
        synergy_pwm_names_file = "{}/{}/ggr.synergy.pwms.order.txt".format(
            args.synergy_main_dir, synergy_dir)
        print synergy_file

        # open synergy file to subsample sequences
        try:
            with h5py.File(synergy_file, "r") as hf:
                for key in sorted(hf.keys()): print key, hf[key].shape
                quit()
                
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
        save_sequences(sequences, diff_sample_indices, None)
        
        # get nondiff, less than dist
        nondiff = np.logical_not(diffs_sig) # {N}
        nondiff_proximal_indices = np.where(np.logical_and(nondiff, distances < max_dist))[0]
        nondiff_proximal_sample_indices = nondiff_proximal_indices[
            np.random.choice(nondiff_proximal_indices.shape[0], nondiff_proximal_sample_num, replace=False)]
        nondiff_proximal_bool = np.zeros(distances.shape)
        nondiff_proximal_bool[nondiff_proximal_sample_indices] = 1
        
        
        # get nondiff, greater than dist
        nondiff_distal_indices = np.where(np.logical_and(nondiff, distances >= max_dist))[0]
        nondiff_distal_sample_indices = nondiff_distal_indices[
            np.random.choice(nondiff_distal_indices.shape[0], nondiff_distal_sample_num, replace=False)]
        nondiff_distal_bool = np.zeros(distances.shape)
        nondiff_distal_bool[nondiff_distal_sample_indices] = 1

        # and mark out the ones chosen in the synergy file
        all_sample_indices = np.concatenate(
            [diff_sample_indices,
             nondiff_proximal_sample_indices,
             nondiff_distal_sample_indices], axis=0)
        all_bool = np.zeros(distances.shape)
        all_bool[all_sample_indices] = 1
        
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
    np_seed = np.random.seed(1337)
    
    # read out sequences from synergy files
    # for each grammar file, go back to synergy file and read out sequences {N, mutM, 1} string
    # sample randomly <- split diff and non diff, and sample half and half from each side
    # for the non diff side, look at distance cutoff and sample half and half from each side
    extract_sequences(args)

    # would be good to plot out the subsamples...
    
    return None



if __name__ == "__main__":
    main()
