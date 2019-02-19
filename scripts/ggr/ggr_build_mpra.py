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


def save_sequences(sequences, df):
    """save sequences to a df
    """
    
    

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
        print synergy_file

        # open synergy file to subsample sequences
        with h5py.File(synergy_file, "r") as hf:

            # extract
            sequences = hf["{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)][:] # {N, combos, 1}
            distances = hf[DataKeys.SYNERGY_DIST][:] # {N}
            diffs_sig = hf[DataKeys.SYNERGY_DIFF_SIG][:] # {N, logit}
            max_dist = hf[DataKeys.SYNERGY_MAX_DIST][()]
            
        # get diff
        diffs_sig = np.greater_equal(np.sum(diffs_sig, axis=1), 2) # {N}
        diff_indices = np.where(diffs_sig)[0]
        diff_sample_indices = diff_indices[
            np.random.choice(diff_indices.shape[0], diff_sample_num, replace=False)]

        # get nondiff, less than dist
        nondiff = np.logical_not(diffs_sig) # {N}
        nondiff_proximal_indices = np.where(np.logical_and(nondiff, distances < max_dist))[0]
        nondiff_proximal_sample_indices = nondiff_proximal_indices[
            np.random.choice(nondiff_proximal_indices.shape[0], nondiff_proximal_sample_num, replace=False)]

        # get nondiff, greater than dist
        nondiff_distal_indices = np.where(np.logical_and(nondiff, distances >= max_dist))[0]
        nondiff_distal_sample_indices = nondiff_distal_indices[
            np.random.choice(nondiff_distal_indices.shape[0], nondiff_distal_sample_num, replace=False)]

        # and mark out the ones chosen in the synergy file
        all_sample_indices = np.concatenate(
            [diff_sample_indices,
             nondiff_proximal_sample_indices,
             nondiff_distal_sample_indices], axis=0)
        
        # set up as sample
        sample_bool = np.zeros(distances.shape)
        sample_bool[all_sample_indices] = 1
        
        # open synergy file to subsample sequences
        with h5py.File(synergy_file, "a") as hf:
            if hf.get("mpra.sample") is not None:
                del hf["mpra.sample"]
            hf.create_dataset("mpra.sample", data=sample_bool)

        # and can plot out here


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
