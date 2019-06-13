#!/usr/bin/env python

import os
import re
import sys
import h5py
import glob
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
        description="given a motif list, get weighted and matched unweighted sets, and footprint")

    parser.add_argument(
        "--motif_list",
        help="text file of motifs to run")
    parser.add_argument(
        "--data_file",
        help="data file with motif positions and scores")
    parser.add_argument(
        "--bam_files", nargs="+",
        help="bam files with reads")
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    parser.add_argument(
        "--prefix",
        help="prefix to attach to output files")
    
    args = parser.parse_args()

    return args


def run_footprinting(data_file, pwm_idx, bam_files, out_dir, prefix):
    """for a given motif, run footprinting and plots
    """
    # extract matched sets of positives and negatives
    get_matched_motif_sites = (
        "python ~/git/tronn/scripts/split_motifs_by_importance_scores.py "
        "--data_files {} "
        "--pwm_idx {} "
        "--out_dir {} "
        "--prefix {}").format(
            data_file, pwm_idx, out_dir, prefix)
    print get_matched_motif_sites
    os.system(get_matched_motif_sites)
    
    # go through bam files
    for bam_file in bam_files:
        bam_prefix = os.path.basename(bam_file).split(".bam")[0]
        bam_prefix = bam_prefix.split(".")[0].split("-")[1]
        bam_dir = "{}/{}".format(out_dir, bam_prefix)

        # get bias corrected positive sites footprints
        match_file="{}/{}.impt_positive.HINT.bed".format(out_dir, prefix)
        build_footprint = (
            "rgt-hint differential "
            "--organism hg19 "
            "--bc "
            "--nc 24 "
            "--mpbs-file1={0} "
            "--mpbs-file2={0} "
            "--reads-file1={1} "
            "--reads-file2={1} "
            "--condition1=c1 "
            "--condition2=c2 "
            "--output-location={2}").format(
                match_file, bam_file, bam_dir)
        print build_footprint
        os.system(build_footprint)
        
        # get bias corrected negative sites footprint
        match_file="{}/{}.impt_negative.HINT.bed".format(out_dir, prefix)
        build_footprint = (
            "rgt-hint differential "
            "--organism hg19 "
            "--bc "
            "--nc 24 "
            "--mpbs-file1={0} "
            "--mpbs-file2={0} "
            "--reads-file1={1} "
            "--reads-file2={1} "
            "--condition1=c1 "
            "--condition2=c2 "
            "--output-location={2}").format(
                match_file, bam_file, bam_dir)
        print build_footprint
        os.system(build_footprint)

        # plot pair together
        paired_file = "{}/pos_w_neg.footprints.txt".format(bam_dir)
        pos_data = pd.read_csv("{}/Lineplots/pos.txt".format(bam_dir), sep="\t")
        neg_data = pd.read_csv("{}/Lineplots/neg.txt".format(bam_dir), sep="\t")
        joint_data = pd.DataFrame(
            {"pos": pos_data["c1"],
             "neg": neg_data["c1"]})
        joint_data.to_csv(paired_file, sep="\t")
        plot_file = "{}/{}.{}.footprints.pdf".format(out_dir, prefix, bam_prefix)
        plot_cmd = "Rscript /srv/scratch/dskim89/ggr/ggr.tronn.2019-06-13.footprinting/plot_footprints.R {} {}".format(
            paired_file, plot_file)
        os.system(plot_cmd)
        
    # pull together files
    all_pos_file = "{}/{}.footprints.timepoints.txt".format(out_dir, prefix)
    positive_files = sorted(glob.glob("{}/*/Lineplots/pos.txt".format(out_dir)))
    pos_data = {}
    for i in range(len(positive_files)):
        positive_file = positive_files[i]
        prefix = positive_file.split("/")[-3]
        data = pd.read_csv(positive_file, sep="\t")
        pos_data[prefix] = data["c1"]
    joint_data = pd.DataFrame(pos_data)
    joint_data.to_csv(all_pos_file, sep="\t")
    plot_file = "{}.pdf".format(all_pos_file.split(".txt")[0])
    plot_cmd = "Rscript /srv/scratch/dskim89/ggr/ggr.tronn.2019-06-13.footprinting/plot_footprints.R {} {}".format(
            all_pos_file, plot_file)
    os.system(plot_cmd)
    
    return


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
    pwm_indices = [re.sub("HCLUST-", "", val) for val in motif_list]
    pwm_indices = [int(re.sub("_.+", "", val)) for val in pwm_indices]

    # debug
    if False:
        motif_list = ["TP63"]
        pwm_indices = [170]
    
    for i in range(len(motif_list)):
        motif_name = motif_list[i]
        pwm_idx = pwm_indices[i]
        pwm_dir = "{}/{}".format(args.out_dir, motif_name)
        
        print motif_name
        run_footprinting(
            args.data_file,
            pwm_idx,
            args.bam_files,
            pwm_dir,
            "{}.{}".format(args.prefix, motif_name))
        
    return


main()
