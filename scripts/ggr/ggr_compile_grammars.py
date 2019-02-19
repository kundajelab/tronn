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


def plot_results(filt_summary_file, out_dir):
    """pull out matrices for plotting. need:
    1) motif presence in each grammar
    2) ATAC pattern for each grammar
    3) RNA pattern for each grammar
    """
    grammars_df = pd.read_table(filt_summary_file)
    print grammars_df.shape

    if False:
        # remove the lowest signals
        vals = grammars_df["ATAC_signal"].values
        print np.min(vals), np.max(vals)
        grammars_df = grammars_df[grammars_df["ATAC_signal"] > 2.85]

        # remove the poorest performers
        vals = grammars_df["delta_logit"].values
        print np.min(vals), np.max(vals)
        grammars_df = grammars_df[grammars_df["delta_logit"] < -0.09] # 0.8

        # don't mess with RNA
        vals = grammars_df["max_rna_vals"].values
        print np.min(vals), np.max(vals)

    # get number of grammars
    num_grammars = grammars_df.shape[0]

    # adjust ordering
    num_to_order_val = {
        0:1,
        1:10,
        2:11,
        3:12,
        4:13,
        5:14,
        7:2,
        8:3,
        9:4,
        10:5,
        11:6,
        12:7,
        13:8,
        14:9
    }
    
    for line_idx in range(num_grammars):
        grammar = nx.read_gml(grammars_df["filename"].iloc[line_idx])
        grammar_traj = int(
            grammars_df["filename"].iloc[line_idx].split(
                "/")[1].split(".")[1].split("-")[1].split("_")[0])
        
        # get motifs and merge into motif df
        motifs = grammars_df["nodes"].iloc[line_idx].split(",")
        motif_presence = pd.DataFrame(
            num_to_order_val[grammar_traj]*np.ones(len(motifs)),
            index=motifs,
            columns=[grammars_df["nodes_rna"].iloc[line_idx]])
        if line_idx == 0:
            motifs_all = motif_presence
        else:
            motifs_all = motifs_all.merge(
                motif_presence,
                left_index=True,
                right_index=True,
                how="outer")
            motifs_all = motifs_all.fillna(0)
            
        # extract RNA vector and append
        rna = [float(val) for val in grammar.graph["RNASIGNALS"].split(",")]
        if line_idx == 0:
            rna_all = np.zeros((num_grammars, len(rna)))
            rna_all[line_idx] = rna
        else:
            rna_all[line_idx] = rna

        # extract ATAC vector and append
        epigenome_signals = np.array([float(val) for val in grammar.graph["ATACSIGNALSNORM"].split(",")])
        atac = epigenome_signals[[0,2,3,4,5,6,9,10,12]]
        if line_idx == 0:
            atac_all = np.zeros((num_grammars, atac.shape[0]))
            atac_all[line_idx] = atac
        else:
            atac_all[line_idx] = atac

    # transpose motifs matrix
    motifs_all["rank"] = np.sum(motifs_all.values, axis=1) / np.sum(motifs_all.values != 0, axis=1)
    motifs_all = motifs_all.sort_values("rank")
    del motifs_all["rank"]
    motifs_all = (motifs_all > 0).astype(int)
    motifs_all = motifs_all.transpose()
    
    # zscore the ATAC data
    atac_all = zscore(atac_all, axis=1)

    # convert others to df
    atac_df = pd.DataFrame(atac_all)
    rna_df = pd.DataFrame(rna_all)
    
    if False:
        # remove solos from motifs and adjust all matrices accordingly
        motif_indices = np.where(np.sum(motifs_all, axis=0) <= 1)[0]
        motifs_all = motifs_all.drop(motifs_all.columns[motif_indices], axis=1)
        orphan_grammar_indices = np.where(np.sum(motifs_all, axis=1) <= 1)[0]
        motifs_all = motifs_all.drop(motifs_all.index[orphan_grammar_indices], axis=0)
        atac_df = atac_df.drop(atac_df.index[orphan_grammar_indices], axis=0)
        rna_df = rna_df.drop(rna_df.index[orphan_grammar_indices], axis=0)
        
    print motifs_all.shape
    print atac_df.shape
    print rna_df.shape

    motifs_file = "{}/grammars.filt.motif_presence.mat.txt".format(out_dir)
    motifs_all.to_csv(motifs_file, sep="\t")

    atac_file = "{}/grammars.filt.atac.mat.txt".format(out_dir)
    atac_df.to_csv(atac_file, sep="\t")

    rna_file = "{}/grammars.filt.rna.mat.txt".format(out_dir)
    rna_df.to_csv(rna_file, sep="\t")
    
    # run R script
    plot_file = "{}/grammars.filt.summary.pdf".format(out_dir)
    plot_summary = "ggr_plot_grammar_summary.R {} {} {} {}".format(
        motifs_file, atac_file, rna_file, plot_file)
    os.system(plot_summary)
    
    return



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
