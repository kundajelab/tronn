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
        "--motif_string",
        help="string to choose motifs to test")
    parser.add_argument(
        "--bigwig_files", nargs="+",
        help="bigwig files with signal")
    parser.add_argument(
        "--mapchain",
        help="mapchain file if need to liftover")
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    parser.add_argument(
        "--prefix",
        help="prefix to attach to output files")
    
    args = parser.parse_args()

    return args


def get_signal_from_bigwig(
        bed_file,
        bigwig_file,
        out_file,
        extend_dist=1000,
        bin_size=20,
        mapchain=None):
    """
    """
    # if liftover, first convert the bed file to correct coords
    if mapchain is not None:
        out_dir = os.path.dirname(bed_file)
        tmp_bed = "{}/liftover.tmp.bed".format(out_dir)
        unmapped_file = "{}/unmapped.txt".format(out_dir)
        liftover = "liftOver {} {} {} {}".format(
            bed_file, mapchain, tmp_bed, unmapped_file)
        os.system(liftover)
    else:
        tmp_bed = bed_file

    # give each region a name (necessary?)
    new_bed = "{}/signal.bed".format(out_dir)
    name_regions = (
        "cat {} | "
        "awk -F '\t' '{{ print $0\"\tregion\"NR }}' "
        "> {}").format(tmp_bed, new_bed)
    os.system(name_regions)

    # then go to deeptools
    mat_file = "{}.coverage.mat.tmp.txt.gz".format(new_bed.split(".bed")[0])
    #bin_total = 100
    #extend_dist = 1000
    #bin_size = extend_dist * 2/ bin_total
    compute_matrix_cmd = (
        "computeMatrix reference-point "
        "--referencePoint {0} "
        "-b {1} -a {1} -bs {2} "
        "-R {3} "
        "-S {4} "
        "-o {5}").format(
            "TSS",
            extend_dist,
            bin_size,
            new_bed,
            bigwig_file,
            mat_file)
    print compute_matrix_cmd
    os.system(compute_matrix_cmd)

    # read in and aggregate
    mat_data = pd.read_csv(mat_file, sep="\t", header=None, comment="@")
    mat_data = mat_data.iloc[:,6:]
    agg = pd.DataFrame({"c1": np.mean(mat_data, axis=0)})
    agg.to_csv(out_file, sep="\t")
    
    return 




def run_chipseq_agg(
        data_file,
        pwm_idx,
        bigwig_files,
        out_dir,
        prefix,
        extend_dist=500,
        bin_size=20,
        mapchain=None,
        label_indices=""):
    """for a given motif, run footprinting and plots
    """
    if label_indices != "":
        label_indices = "--labels_indices {}".format(label_indices)
    label_indices = ""
        
    # extract matched sets of positives and negatives
    get_matched_motif_sites = (
        "python ~/git/tronn/scripts/split_motifs_by_importance_scores.py "
        "--data_files {} "
        "--labels_key ATAC_LABELS " # TRAJ_LABELS
        "{} "
        "--pwm_idx {} "
        "--out_dir {} "
        "--prefix {}").format(
            data_file, label_indices, pwm_idx, out_dir, prefix)
    print get_matched_motif_sites
    os.system(get_matched_motif_sites)

    # go through bigwig files
    for bigwig_file in bigwig_files:
        bigwig_prefix = os.path.basename(bigwig_file).split(".bw")[0].split(".bigwig")[0]

        # get positive sites
        match_file="{}/{}.impt_positive.HINT.bed".format(out_dir, prefix)
        pos_file = "{}/{}.positive_vals.txt".format(out_dir, prefix)
        get_signal_from_bigwig(match_file, bigwig_file, pos_file,
                               extend_dist=extend_dist, bin_size=bin_size, mapchain=mapchain)

        # get negative sites
        match_file="{}/{}.impt_negative.HINT.bed".format(out_dir, prefix)
        neg_file = "{}/{}.negative_vals.txt".format(out_dir, prefix)
        get_signal_from_bigwig(match_file, bigwig_file, neg_file,
                               extend_dist=extend_dist, bin_size=bin_size, mapchain=mapchain)

        # plot pair together
        paired_file = "{}/pos_w_neg.chipseq_agg.txt".format(out_dir)
        pos_data = pd.read_csv(pos_file, sep="\t")
        neg_data = pd.read_csv(neg_file, sep="\t")
        joint_data = pd.DataFrame(
            {"pos": pos_data["c1"],
             "neg": neg_data["c1"]})
        joint_data.index = np.arange(-extend_dist, extend_dist, bin_size) + (bin_size / 2)
        joint_data.to_csv(paired_file, sep="\t")
        plot_file = "{}/{}.{}.agg_chipseq.pdf".format(out_dir, prefix, bigwig_prefix)
        plot_cmd = "/users/dskim89/git/ggr-project/figs/fig_3.motifs_and_tfs/fig_3-f.0.plot.agg_chipseq.R {} {}".format(
            paired_file, plot_file)
        print plot_cmd
        os.system(plot_cmd)

    # pull together files
    if False:
        all_pos_file = "{}/{}.agg_chipseq.timepoints.txt".format(out_dir, prefix)
        positive_files = sorted(glob.glob("{}/*/pos_w_neg.footprints.txt".format(out_dir)))
        pos_data = {}
        for i in range(len(positive_files)):
            positive_file = positive_files[i]
            prefix = positive_file.split("/")[-2]
            prefix = re.sub("0$", ".0", prefix)
            data = pd.read_csv(positive_file, sep="\t")
            neg_data = data["neg"]
            norm_factor = (np.mean(neg_data[0:10]) + np.mean(neg_data[190:200])) / 2.
            pos_data[prefix] = data["pos"] / norm_factor
        joint_data = pd.DataFrame(pos_data)
        joint_data.to_csv(all_pos_file, sep="\t")
        plot_file = "{}.pdf".format(all_pos_file.split(".txt")[0])
        plot_cmd = "/users/dskim89/git/ggr-project/figs/fig_3.motifs_and_tfs/fig_3-e.0.plot.footprints.R {} {}".format(
            all_pos_file, plot_file)
        print plot_cmd
        os.system(plot_cmd)
    
    return


def main():
    """get chipseq signal
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
    
    for i in range(len(motif_list)):
        motif_name = motif_list[i]
        pwm_idx = pwm_indices[i]
        pwm_dir = "{}/{}".format(args.out_dir, motif_name)

        # only look at specific trajectories
        # NOTE CURRENTLY NOT USED
        traj_indices = np.where(motifs.iloc[i].values!=0)[0]
        trajectories = list(motifs.columns[traj_indices])
        traj_indices = []
        for traj in trajectories:
            indices = traj.split("-")[1:]
            traj_indices += traj.split("-")[1:]
        traj_indices = " ".join(traj_indices)

        if args.motif_string not in motif_name:
            continue
        
        print motif_name
        run_chipseq_agg(
            args.data_file,
            pwm_idx,
            args.bigwig_files,
            pwm_dir,
            "{}.{}".format(args.prefix, motif_name),
            mapchain=args.mapchain,
            label_indices=traj_indices)
        
    return


main()
