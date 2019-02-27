#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import sys
import glob
import logging
import argparse

import pandas as pd

from tronn.util.ggr_utils import compile_grammars
from tronn.util.ggr_utils import expand_pwms_by_rna
from tronn.util.ggr_utils import merge_duplicates
from tronn.util.ggr_utils import plot_results
from tronn.util.scripts import setup_run_logs


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # required args
    parser.add_argument(
        "--grammar_summaries", nargs="+",
        help="all grammar summaries to be annotated and compiled")
    parser.add_argument(
        "--filter", default="GO_terms",
        help="filter column")
    parser.add_argument(
        "--synergy_dirs", nargs="+",
        help="directories of synergy files")
    parser.add_argument(
        "--tss",
        help="BED file of TSS positions with gene IDs in the name column")
    parser.add_argument(
        "--foreground_rna",
        help="gene set for linking grammars")
    parser.add_argument(
        "--background_rna",
        help="all genes expressed (as background for GO enrichment)")

    # pwm useful stuff
    parser.add_argument(
        "--pwms",
        help="pwm file to remove long pwms from analysis")
    parser.add_argument(
        "--pwm_metadata",
        help="pwm metadata to get gene IDs of interest")
    parser.add_argument(
        "--max_pwm_length", default=20, type=int,
        help="cutoff for removing long pwms")
    
    # out
    parser.add_argument(
        "--merged_synergy_dir", dest="merged_synergy_dir",
        help="out directory")

    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    
    # parse args
    args = parser.parse_args()

    return args


def main():
    """merge synergy runs
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    os.system("mkdir -p {}".format(args.merged_synergy_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    # first compile all summaries
    summary_file = compile_grammars(args)

    # set up auxiliary files
    dynamic_genes = pd.read_table(args.foreground_rna, index_col=0)
    pwm_metadata = pd.read_table(args.pwm_metadata).dropna()
    pwm_to_rna_dict = dict(zip(
        pwm_metadata["hclust_model_name"].values.tolist(),
        pwm_metadata["expressed_hgnc"].values.tolist()))
    split_cols = ["expressed", "expressed_hgnc"]
    pwm_metadata = expand_pwms_by_rna(pwm_metadata, split_cols)
    genes_w_pwms = dict(zip(
        pwm_metadata["expressed"].values.tolist(),
        pwm_metadata["expressed_hgnc"].values.tolist()))
    interesting_genes = genes_w_pwms

    # merge
    merged_summary_file = merge_duplicates(
        summary_file,
        pwm_to_rna_dict,
        args.tss,
        dynamic_genes,
        interesting_genes,
        args.background_rna,
        args.out_dir,
        max_dist=500000,
        synergy_dirs=args.synergy_dirs,
        merged_synergy_dir=args.merged_synergy_dir)
    
    # generate matrices for plotting
    # NOTE removed solo motif grammars
    plot_results(merged_summary_file, args.out_dir)
    
    return None



if __name__ == "__main__":
    main()
