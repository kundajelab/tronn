#!/usr/bin/env python

import os
import re
import sys
import h5py
import argparse

import pandas as pd
import numpy as np

from tronn.util.scripts import setup_run_logs
from tronn.util.utils import DataKeys

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
        help="h5 files with motif scan results")
    parser.add_argument(
        "--motif_list",
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

    # read in motif file and get indices from names
    motifs = pd.read_csv(args.motif_list, sep="\t", index_col=0)
    motif_list = list(motifs.index)
    #pwm_indices = [re.sub("HCLUST-", "", val) for val in motif_list]
    #pwm_indices = [int(re.sub("_.+", "", val)) for val in pwm_indices]
    
    # read in active motifs from scanmotifs
    with h5py.File(args.data_file, "r") as hf:
        pwm_hits = hf[DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM][:] > 0
        pwm_raw = hf[DataKeys.ORIG_SEQ_PWM_SCORES_SUM][:] > 0
        pwm_names = hf[DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM].attrs["pwm_names"]
        
    # filter
    pwm_indices = np.where(np.isin(pwm_names, motif_list))[0]
    pwm_hits = pwm_hits[:,:,pwm_indices]
    pwm_hits = np.max(pwm_hits, axis=1)
    
    pwm_raw = pwm_raw[:,0,pwm_indices]
    pwm_names = pwm_names[pwm_indices]

    # adjust pwm names
    pwm_names = [re.sub("HCLUST-\d+.", "", val) for val in pwm_names]
    pwm_names = [re.sub(".UNK.+", "", val) for val in pwm_names]
    
    # flatten
    #pwm_hits = np.reshape(pwm_hits, [-1, pwm_hits.shape[2]])

    # marginals
    num_pwms = pwm_hits.shape[1]
    marginals = np.zeros((num_pwms))
    for pwm_i in range(num_pwms):
        marginals[pwm_i] = np.mean(pwm_raw[:,pwm_i])
    
    # and count up co-occurrence
    mat = np.zeros((num_pwms, num_pwms))
    for pwm_i in range(num_pwms):
        for pwm_j in range(num_pwms):

            # multiply or add?
            if False:
                joint_presence = np.multiply(
                    pwm_hits[:,pwm_i],
                    pwm_hits[:,pwm_j])
                active_total = np.sum(joint_presence)



                expected_score = np.multiply(
                    pwm_raw[:,pwm_i],
                    pwm_raw[:,pwm_j])
                expected_total = np.sum(expected_score).astype(float)

                diff = active_total / expected_total

            actual = np.multiply(
                pwm_hits[:,pwm_i],
                pwm_hits[:,pwm_j])
            actual = np.mean(actual)
            
            expected = np.multiply(
                marginals[pwm_i],
                marginals[pwm_j])

            diff = actual - expected

            mat[pwm_i, pwm_j] = diff
            
    # save to df
    out_file = "{}/{}.cooccurrence.mat.txt".format(args.out_dir, args.prefix)
    data = pd.DataFrame(data=mat, index=pwm_names, columns=pwm_names)
    data.to_csv(out_file, sep="\t")

    


    quit()
    
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
        pwm_data = pwm_data.drop_duplicates()
        
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
    traj_pwms = traj_pwms.fillna(0).drop_duplicates()

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
