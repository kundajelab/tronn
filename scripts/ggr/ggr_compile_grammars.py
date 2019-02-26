#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import sys
import argparse

from tronn.util.scripts import setup_run_logs
from tronn.util.ggr_utils import compile_grammars
from tronn.util.ggr_utils import plot_results

def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # required args
    parser.add_argument(
        "--grammar_summaries", nargs="+",
        help="all grammar summary files to be compiled")
    parser.add_argument(
        "--filter", default="GO_terms",
        help="filter column (1 or 0)")
    
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

    # compile
    filt_summary_file = compile_grammars(args)
    
    # generate matrices for plotting
    # NOTE removed solo motif grammars
    plot_results(filt_summary_file, args.out_dir)
    
    return None



if __name__ == "__main__":
    main()
