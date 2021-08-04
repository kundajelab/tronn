#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import sys
import logging
import argparse

from tronn.util.ggr_utils import annotate_grammars
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

    # relevant annotation info
    parser.add_argument(
        "--region_signal_mat_file",
        help="file with region signal matrix")
    parser.add_argument(
        "--rna_signal_mat_file",
        help="file with region signal matrix")
    parser.add_argument(
        "--links_file",
        help="links file")
    parser.add_argument(
        "--tss_file",
        help="BED file of TSS positions with gene ids")
    parser.add_argument(
        "--foreground_rna",
        help="gene set for linking grammars")
    parser.add_argument(
        "--background_rna",
        help="all genes expressed (as background for GO enrichment)")
    parser.add_argument(
        "--no_go_terms", action="store_true",
        help="run without GO annotations")
    parser.add_argument(
        "--pwm_metadata",
        help="pwm metadata to get gene IDs of interest")
    
    # out
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    parser.add_argument(
        "--tmp_dir", dest="tmp_dir", type=str,
        default="./",
        help="tmp directory")
    
    # parse args
    args = parser.parse_args()

    return args


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])
    args.grammars = sorted(args.grammars)
    logging.info("Starting with {} grammars".format(len(args.grammars)))
    os.system("mkdir -p {}".format(args.tmp_dir))
    
    # annotate grammars
    filt_summary_file = annotate_grammars(args)

    # generate matrices for plotting
    plot_results(filt_summary_file, args.out_dir)
    
    return None



if __name__ == "__main__":
    main()
