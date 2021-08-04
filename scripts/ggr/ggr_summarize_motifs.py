#!/usr/bin/env python

import os
import sys
import h5py
import argparse

import pandas as pd
import numpy as np

from tronn.util.scripts import setup_run_logs

BLACKLIST_MOTIFS = [
    "SMARC",
    "NANOG"]


def get_blacklist_indices(df):
    """search for blacklist substrings and return a list of indices
    """
    blacklist_indices = []
    for i in range(df.shape[0]):
        for substring in BLACKLIST_MOTIFS:
            if substring in df.index[i]:
                blacklist_indices.append(df.index[i])

    return blacklist_indices


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    parser.add_argument(
        "--data_file",
        help="pval file produced after running intersect_pwms_and_rna.py")
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    parser.add_argument(
        "--prefix",
        help="prefix to attach to output files")
    
    args = parser.parse_args()

    return args


def main():
    """condense results
    """
    # set up args
    args = parse_args()
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])
    prefix = "{}/{}".format(args.out_dir, args.prefix)
    
    # GGR ordered trajectory indices
    with h5py.File(args.data_file, "r") as hf:
        foregrounds_keys = hf["pvals"].attrs["foregrounds.keys"]
    labels = [val.replace("_LABELS", "") for val in foregrounds_keys]
    days  = ["day {0:.1f}".format(float(val))
             for val in [0,1,1.5,2,2.5,3,4.5,5,6]]

    # go through each index to collect
    for i in range(len(foregrounds_keys)):
        key = foregrounds_keys[i]
        #key = "TRAJ_LABELS-{}".format(index)
        
        with h5py.File(args.data_file, "r") as hf:
            sig = hf["pvals"][key]["sig"][:]
            rna_patterns = hf["pvals"][key]["rna_patterns"][:]
            pwm_patterns = hf["pvals"][key]["pwm_patterns"][:]
            correlations = hf["pvals"][key]["correlations"][:]
            hgnc_ids = hf["pvals"][key].attrs["hgnc_ids"]
            pwm_names = hf["pvals"][key].attrs["pwm_names"]
            
        # TF present
        tf_present = pd.DataFrame(
            correlations,
            index=hgnc_ids)
        tf_present.columns = [key]
        
        # rna pattern
        tf_data = pd.DataFrame(rna_patterns, index=hgnc_ids)
        
        # pwm present
        pwm_present = pd.DataFrame(
            np.arcsinh(np.max(pwm_patterns, axis=1)),
            index=pwm_names)
        pwm_present.columns = [key]
        
        # pwm pattern
        pwm_data = pd.DataFrame(pwm_patterns, index=pwm_names)
        pwm_data["pwm_names"] = pwm_data.index.values
        pwm_data = pwm_data.drop_duplicates()
        pwm_data = pwm_data.drop("pwm_names", axis=1)
        
        if i == 0:
            traj_tfs = tf_present
            traj_pwms = pwm_present
            tf_patterns = tf_data
            motif_patterns = pwm_data
        else:
            traj_tfs = traj_tfs.merge(tf_present, how="outer", left_index=True, right_index=True)
            traj_pwms = traj_pwms.merge(pwm_present, how="outer", left_index=True, right_index=True)
            tf_patterns = pd.concat([tf_patterns, tf_data])
            tf_patterns = tf_patterns.drop_duplicates()
            motif_patterns = pd.concat([motif_patterns, pwm_data])
            #motif_patterns = motif_patterns.drop_duplicates()

    # remove nans/duplicates
    traj_tfs = traj_tfs.fillna(0)
    traj_pwms = traj_pwms.fillna(0).reset_index().drop_duplicates()
    traj_pwms = traj_pwms.set_index("index")
    
    # reindex
    tf_patterns = tf_patterns.reindex(traj_tfs.index)
    motif_patterns = motif_patterns.groupby(motif_patterns.index).mean() # right now, just average across trajectories (though not great)
    motif_patterns = motif_patterns.reindex(traj_pwms.index)

    # fix column names
    traj_pwms.columns = labels
    motif_patterns.columns = days
    traj_tfs.columns = labels
    tf_patterns.columns = days

    # remove blacklist
    motif_indices = get_blacklist_indices(motif_patterns)
    traj_pwms = traj_pwms.drop(index=motif_indices)
    motif_patterns = motif_patterns.drop(index=motif_indices)

    tf_indices = get_blacklist_indices(tf_patterns)
    traj_tfs = traj_tfs.drop(index=tf_indices)
    tf_patterns = tf_patterns.drop(index=tf_indices)
    
    # filtering on specific TFs and motifs to exclude
    traj_tfs_file = "{}.tfs_corr_summary.txt".format(prefix)
    traj_pwms_file = "{}.pwms_present_summary.txt".format(prefix)
    tf_patterns_file = "{}.tfs_patterns_summary.txt".format(prefix)
    motif_patterns_file = "{}.pwms_patterns_summary.txt".format(prefix)
    
    traj_tfs.to_csv(traj_tfs_file, sep="\t")
    traj_pwms.to_csv(traj_pwms_file, sep="\t")
    tf_patterns.to_csv(tf_patterns_file, sep="\t")
    motif_patterns.to_csv(motif_patterns_file, sep="\t")

    # and R script?
    plot_results = "ggr_plot_motif_summary.R {} {} {} {}".format(
        traj_pwms_file, motif_patterns_file, traj_tfs_file, tf_patterns_file)
    print plot_results
    os.system(plot_results)
    
    return


main()
