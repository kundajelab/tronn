#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import re
import sys
import argparse

import numpy as np
import pandas as pd
import networkx as nx

from scipy.stats import zscore

from tronn.util.scripts import setup_run_logs
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


def compile_grammars(args):
    """compile grammars
    """

    for grammar_summary_idx in range(len(args.grammar_summaries)):
        grammar_summary = args.grammar_summaries[grammar_summary_idx]
        grammar_summary_path = os.path.dirname(grammar_summary)
        print grammar_summary

        # read in
        grammar_summary = pd.read_table(grammar_summary, index_col=0)

        # filter
        grammar_summary = grammar_summary[grammar_summary[args.filter] == 1]

        # copy relevant files to new folder
        for grammar_idx in range(grammar_summary.shape[0]):
            # need gprofiler and gml
            grammar_file = "{}/{}".format(
                grammar_summary_path,
                os.path.basename(grammar_summary.iloc[grammar_idx]["filename"]))
            copy_file = "cp {} {}/".format(grammar_file, args.out_dir)
            print copy_file
            os.system(copy_file)
            functional_file = re.sub(".annot-\d+.gml", "*gprofiler.txt", grammar_file)
            copy_file = "cp {} {}/".format(functional_file, args.out_dir)
            print copy_file
            os.system(copy_file)

            # and adjust file location
            grammar_index = grammar_summary.index[grammar_idx]
            grammar_summary.at[grammar_index, "filename"] = "{}/{}".format(
                args.out_dir,
                os.path.basename(grammar_file))
            
        # concat
        if grammar_summary_idx == 0:
            all_grammars = grammar_summary
        else:
            all_grammars = pd.concat([all_grammars,grammar_summary], axis=0)

    # save out to new dir
    all_grammars = all_grammars.sort_values("filename")
    if "manual_filt" not in all_grammars.columns:
        all_grammars.insert(0, "manual_filt", np.ones(all_grammars.shape[0]))
    new_grammar_summary_file = "{}/grammars_summary.txt".format(args.out_dir)
    all_grammars.to_csv(new_grammar_summary_file, sep="\t")
    
    return new_grammar_summary_file



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
