#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import sys
import logging
import argparse

from tronn.util.epithelia_utils import annotate_grammars
from tronn.util.ggr_utils import plot_results
from tronn.util.scripts import setup_run_logs


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # required args
    parser.add_argument(
        "--grammars", nargs="+",
        help="all gml files to be annotated and compiled")
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
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    
    # parse args
    args = parser.parse_args()

    return args


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    # annotate grammars
    #filt_summary_file = "{}/grammar_summary.filt.txt".format(args.out_dir)
    #if not os.path.isfile(filt_summary_file):
    filt_summary_file = annotate_grammars(args)

    # generate matrices for plotting
    # NOTE removed solo motif grammars
    plot_results(filt_summary_file, args.out_dir)
    
    return None



if __name__ == "__main__":
    main()
