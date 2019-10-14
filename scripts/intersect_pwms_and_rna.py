#!/usr/bin/env python

"""
description: script to use for those who also
have joint RNA data to help filter their results
from `tronn scanmotifs`

Key params used here:

- correlation

"""

import os
import sys
import glob
import h5py
import logging
import argparse

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from tronn.datalayer import H5DataLoader
from tronn.stats.nonparametric import threshold_by_qvalues

from tronn.util.h5_utils import AttrKeys
from tronn.util.pwms import MotifSetManager
from tronn.util.utils import DataKeys
from tronn.util.scripts import setup_run_logs
from tronn.util.scripts import parse_multi_target_selection_strings
from tronn.util.scripts import load_selected_targets


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="add in RNA data to filter pwms")

    # required args
    parser.add_argument(
        "--dataset_files", nargs="+",
        required=True,
        help="dataset with motif scans")
    parser.add_argument(
        "--pvals_file",
        required=True,
        help="h5 file with pvals for pwms")
    parser.add_argument(
        "--pwm_file",
        required=True,
        help="pwm file to filter")
    parser.add_argument(
        "--pwm_metadata_file",
        required=True,
        help="metadata file, requires 1 column with PWM names and the other with gene ids OR genes present")

    # options
    parser.add_argument(
        "--pvals_key", default="pvals",
        help="pvals key")
    parser.add_argument(
        "--pwm_scores_key", default=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        help="scores key in the hdf5 file")
    parser.add_argument(
        "--pwm_metadata_expr_col_key", default="expressed",
        help="which column has the TF presence information")
    parser.add_argument(
        "--pwm_metadata_hgnc_col_key", default="expressed_hgnc",
        help="which column has the TF presence information")
    parser.add_argument(
        "--cor_thresh", type=float, default=0.75,
        help="pwm score/rna expression correlation threshold")
    parser.add_argument(
        "--other_targets",
        nargs="+",
        default=[DataKeys.LOGITS],
        help="other keys and indices that are relevant")
    
    # for second stage
    parser.add_argument(
        "--rna_expression_file",
        help="RNA file if adding in RNA expression information")

    # other filtering
    parser.add_argument(
        "--max_pwm_length", default=50, type=int,
        help="max pwm length allowed as significant (weed out long non-informative pwms)")
    
    # other
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")

    # parse args
    args = parser.parse_args()
    
    return args

# TODO move this to util/scripts
def _parse_to_key_and_indices(key_strings):
    """given an arg string list, parse out into a dict
    assumes a format of: key=indices
    
    NOTE: this is for smart indexing into tensor dicts (mostly label sets)

    Returns:
      list of tuples, where tuple is (dataset_key, indices_list)
    """
    ordered_keys = []
    for key_string in key_strings:
        key_and_indices = key_string.split("=")
        key = key_and_indices[0]
        if len(key_and_indices) > 1:
            indices = [int(i) for i in key_and_indices[1].split(",")]
        else:
            indices = []
        ordered_keys.append((key, indices))
    
    return ordered_keys


def _expand_pwms_by_rna(pwms_df, split_cols, split_char=";"):
    """given column(s) with multiple values in a row, expand so that
    each row has a single value

    assumes that multiple split_cols are ordered same way
    """
    # make the duplicate values into lists
    for split_col in split_cols:
        pwms_df = pwms_df.assign(
            **{split_col: pwms_df[split_col].str.split(split_char)})

    # repeat rows based on list lengths
    pwms_df_expanded = pd.DataFrame({
        col: np.repeat(pwms_df[col].values, pwms_df[split_cols[0]].str.len())
        for col in pwms_df.columns})    

    # put a unique value into each row
    for split_col in split_cols:
        pwms_df_expanded = pwms_df_expanded.assign(
            **{split_col: np.concatenate(pwms_df[split_col].values)})[
                pwms_df.columns.tolist()]

    return pwms_df_expanded



def main():
    """run the intersect
    """
    # set up args
    args = parse_args()
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    # set up other targets to load
    args.other_targets = _parse_to_key_and_indices(args.other_targets)

    # set up dataloader
    data_loader = H5DataLoader(data_files=args.dataset_files)

    # read in pwms and metadata file, get expression info from column
    pwm_list = MotifSetManager.read_pwm_file(args.pwm_file)
    pwm_metadata = pd.read_table(args.pwm_metadata_file, sep="\t")
    tf_expressed = pwm_metadata[args.pwm_metadata_expr_col_key].notnull().values

    # set up blacklist motifs filter (ignore long pwms)
    blacklisted_pwms = np.ones(len(pwm_list))
    for pwm_idx in range(len(pwm_list)):
        pwm = pwm_list[pwm_idx]
        if pwm.weights.shape[1] > args.max_pwm_length:
            blacklisted_pwms[pwm_idx] = 0
    
    # STAGE 1 - filter for expressed
    expr_pvals_file = "{}/pvals.rna_filt.h5".format(args.out_dir)

    # get groups
    with h5py.File(args.pvals_file, "r") as hf:
        foregrounds_strings = hf[args.pvals_key].attrs["foregrounds"]
        foreground_keys = hf[args.pvals_key].attrs["foregrounds.keys"]
    foregrounds = parse_multi_target_selection_strings(foregrounds_strings)
        
    # check each foreground
    for foreground_idx in xrange(len(foreground_keys)):
        foreground_key = foreground_keys[foreground_idx]
        
        # get sig pwms
        sig_pwms_key = "{}/{}/sig".format(
            args.pvals_key, foreground_key)
        with h5py.File(args.pvals_file, "r") as hf:
            old_sig_pwms = hf[sig_pwms_key][:]
            pwm_names = hf[sig_pwms_key].attrs[AttrKeys.PWM_NAMES]

        # filter
        sig_pwms_filt = np.multiply(
            old_sig_pwms,
            tf_expressed)

        # and filter blacklist
        sig_pwms_filt = np.multiply(
            sig_pwms_filt,
            blacklisted_pwms)

        # save out with attributes
        with h5py.File(expr_pvals_file, "a") as out:
            out.create_dataset(sig_pwms_key, data=sig_pwms_filt)
            out[sig_pwms_key].attrs[AttrKeys.PWM_NAMES] = pwm_names
            out[sig_pwms_key].attrs["ensembl_ids"] = pwm_metadata[
                args.pwm_metadata_expr_col_key].values.astype(str)
            out[sig_pwms_key].attrs["hgnc_ids"] = pwm_metadata[
                args.pwm_metadata_hgnc_col_key].values.astype(str)
            out[args.pvals_key].attrs["foregrounds"] = foregrounds_strings
            out[args.pvals_key].attrs["foregrounds.keys"] = foreground_keys

        logging.info(
            "{}: After filtering for expressed TFs, got {} motifs (from {})".format(
                sig_pwms_key, np.sum(sig_pwms_filt), np.sum(old_sig_pwms)))

    # STAGE 2 - correlation information
    corr_pvals_file = "{}/pvals.rna_filt.corr_filt.h5".format(args.out_dir)
    
    # read in RNA matrix
    rna_patterns = pd.read_table(args.rna_expression_file, sep="\t", index_col=0)
    rna_patterns["ensembl_ids"] = rna_patterns.index
    
    # get foregrounds
    with h5py.File(expr_pvals_file, "r") as hf:
        foregrounds_strings = hf[args.pvals_key].attrs["foregrounds"]
        foreground_keys = hf[args.pvals_key].attrs["foregrounds.keys"]
    foregrounds = parse_multi_target_selection_strings(foregrounds_strings)
    
    # extract scores for that target to save into patterns
    other_targets = {}
    for target, target_indices in args.other_targets:
        other_targets[target] = data_loader.load_dataset(target)[:,target_indices]
    pwm_scores = data_loader.load_dataset(args.pwm_scores_key)
    
    # adjust pwm scores IF rc pwms included
    # this is indicated by the len of pwm names relative to background (2x scores)
    scores_tmp = np.reshape(
        pwm_scores, list(pwm_scores.shape)[:2] + [2, -1])
    pwm_scores = np.sum(scores_tmp, axis=-2)
        
    for foreground_idx in xrange(len(foreground_keys)):
        foreground_key = foreground_keys[foreground_idx]
        
        # set up targets
        foreground = foregrounds[foreground_idx]
        selected_targets = load_selected_targets(
            data_loader, foreground[0], foreground[1])
        foreground_indices = np.where(selected_targets)[0]
        
        # get sig pwms
        sig_pwms_key = "{}/{}/sig".format(
            args.pvals_key, foreground_key)
        with h5py.File(expr_pvals_file, "r") as hf:
            old_sig_pwms = hf[sig_pwms_key][:]
            pwm_names = hf[sig_pwms_key].attrs[AttrKeys.PWM_NAMES]
            ensembl_ids = hf[sig_pwms_key].attrs["ensembl_ids"]
            hgnc_ids = hf[sig_pwms_key].attrs["hgnc_ids"]

        # set up pwm score patterns
        example_scores = pwm_scores[foreground_indices]
        pwm_patterns = np.sum(example_scores, axis=0).transpose()
        # REMOVE LATER
        logging.info("WARNING GGR SPECIFIC ADJUSTMENT HERE")
        pwm_patterns = pwm_patterns[:,[0,2,3,4,5,6,7,8,9]]
        # add in all necessary information to pwm patterns (convert to dataframe)
        pwm_patterns = pd.DataFrame(pwm_patterns, index=pwm_names)
        pwm_patterns["ensembl_ids"] = ensembl_ids
        pwm_patterns["hgnc_ids"] = hgnc_ids
        pwm_patterns["original_indices"] = range(pwm_patterns.shape[0])
        pwm_patterns["pwm_names"] = pwm_patterns.index

        # filter out non sig
        sig_indices = np.where(old_sig_pwms > 0)[0]
        pwm_patterns = pwm_patterns.iloc[sig_indices]

        # expand on RNA column
        split_cols = ["ensembl_ids", "hgnc_ids"]
        pwm_patterns = _expand_pwms_by_rna(pwm_patterns, split_cols)

        # move hgnc names to index and remove extraneous data
        pwm_patterns = pwm_patterns.set_index("hgnc_ids")
        pwm_patterns_vals = pwm_patterns[
            pwm_patterns.columns.difference(
                ["ensembl_ids", "original_indices", "pwm_names"])]

        # set up rna patterns
        rna_patterns_matched = rna_patterns[rna_patterns["ensembl_ids"].isin(pwm_patterns["ensembl_ids"])]
        rna_patterns_matched = rna_patterns_matched.set_index("ensembl_ids")
        rna_patterns_matched = rna_patterns_matched.loc[pwm_patterns["ensembl_ids"]]

        # calculate the row-wise correlations
        pwm_rna_correlations = np.zeros((pwm_patterns.shape[0]))
        for i in xrange(pwm_rna_correlations.shape[0]):
            corr_coef, pval = pearsonr(
                pwm_patterns_vals.values[i], rna_patterns_matched.values[i])
            pwm_rna_correlations[i] = corr_coef

        # filter for correlation
        if args.cor_thresh is not None:
            good_cor = pwm_rna_correlations >= args.cor_thresh
            pwm_patterns = pwm_patterns.iloc[good_cor]
            pwm_patterns_vals = pwm_patterns_vals.iloc[good_cor]
            rna_patterns_matched = rna_patterns_matched.iloc[good_cor]
            pwm_rna_correlations = pwm_rna_correlations[good_cor]

        # set up new sig pwms
        new_sig_pwms = np.zeros((pwm_scores.shape[2])).astype(int)
        new_sig_pwms[pwm_patterns["original_indices"].values] = 1

        # extract other keys
        subset_targets = {}
        for key in other_targets.keys():
            subset_targets[key] = np.mean(
                other_targets[key][foreground_indices], axis=0)
            
        # finally, all of this needs to be saved out
        subgroup_key = "{}/{}".format(
            args.pvals_key, foreground_key)
        new_sig_pwms_key = "{}/sig".format(
            subgroup_key)
        pwm_patterns_key = "{}/pwm_patterns".format(subgroup_key)
        rna_patterns_key = "{}/rna_patterns".format(subgroup_key)
        cor_key = "{}/correlations".format(subgroup_key)

        with h5py.File(corr_pvals_file, "a") as hf:
            # datasets
            hf.create_dataset(new_sig_pwms_key, data=new_sig_pwms)
            hf.create_dataset(pwm_patterns_key, data=pwm_patterns_vals.values)
            hf.create_dataset(rna_patterns_key, data=rna_patterns_matched.values)
            hf.create_dataset(cor_key, data=pwm_rna_correlations)

            # attributes
            hf[subgroup_key].attrs["ensembl_ids"] = pwm_patterns["ensembl_ids"].values.astype(str)
            hf[subgroup_key].attrs["hgnc_ids"] = pwm_patterns.index.values.astype(str)
            hf[subgroup_key].attrs["pwm_names"] = pwm_patterns["pwm_names"].values.astype(str)
            hf[args.pvals_key].attrs["foregrounds"] = foregrounds_strings
            hf[args.pvals_key].attrs["foregrounds.keys"] = foreground_keys

            # other keys - keep to a separate subgroup
            for key in subset_targets.keys():
                out_key = "{}/other/{}".format(subgroup_key, key)
                hf.create_dataset(out_key, data=subset_targets[key])

        logging.info(
            "{}: After filtering for correlated TFs, got {} motifs (from {})".format(
                new_sig_pwms_key, np.sum(new_sig_pwms), np.sum(old_sig_pwms)))

    # and plot
    plot_cmd = "plot-h5.pwm_x_rna.R {} {}".format(corr_pvals_file, args.pvals_key)
    print plot_cmd
    os.system(plot_cmd)
    
    return



if __name__ == '__main__':
    main()
