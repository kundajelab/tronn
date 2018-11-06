#!/usr/bin/env python

'''
description: script to use for those who also
have joint RNA data to help filter their results
from `tronn scanmotifs`



'''

import os
import sys
import glob
import h5py
import logging
import argparse

import numpy as np
import pandas as pd

from scipy.stats import pearsonr

from tronn.stats.nonparametric import threshold_by_qvalues

from tronn.util.h5_utils import AttrKeys
from tronn.util.pwms import MotifSetManager
from tronn.util.utils import DataKeys


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="add in RNA data to filter pwms")

    # required args
    parser.add_argument(
        "--pvals_h5_file", required=True,
        help="h5 file with pvals for pwms")
    parser.add_argument(
        "--pwm_file", required=True,
        help="pwm file to filter")
    parser.add_argument(
        "--pwm_metadata_file", required=True,
        help="metadata file, requires 1 column with PWM names and the other with gene ids OR genes present")

    # options
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
        "--qval_thresh", default=0.05,
        help="qval threshold")
    
    # for second stage
    parser.add_argument(
        "--rna_expression_file",
        help="RNA file if adding in RNA expression information")

    # other things to visualize
    parser.add_argument(
        "--other_targets", nargs="+",
        help="other keys and indices that are relevant")
    
    # other
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")

    # parse args
    args = parser.parse_args()
    
    return args


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


def track_runs(args):
    """track command and github commit
    """
    # keeps track of restores (or different commands) in folder
    subcommand_name = "intersect_pwms_and_rna"
    num_restores = len(glob.glob('{0}/{1}.command*'.format(args.out_dir, subcommand_name)))
    logging_file = '{0}/{1}.command_{2}.log'.format(args.out_dir, subcommand_name, num_restores)
    
    # track github commit
    git_repo_path = os.path.dirname(os.path.realpath(__file__))
    os.system('echo "commit:" > {0}'.format(logging_file))
    os.system('git --git-dir={0}/.git rev-parse HEAD >> {1}'.format(
        git_repo_path.split("/scripts")[0], logging_file))
    os.system('echo "" >> {0}'.format(logging_file))
    
    # write out the command
    with open(logging_file, 'a') as f:
        f.write(' '.join(sys.argv)+'\n\n')
    
    return logging_file


def _setup_logs(args):
    """set up logging
    """
    logging_file = track_runs(args)
    reload(logging)
    logging.basicConfig(
        filename=logging_file,
        level=logging.DEBUG, # TODO ADJUST BEFORE RELEASE
        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    for arg in sorted(vars(args)):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    logging.info("")

    return


def main():
    """run the intersect
    """
    # set up args
    args = parse_args()
    args.other_targets = _parse_to_key_and_indices(args.other_targets)
    _setup_logs(args)
    
    # set up out file
    out_file = "{}/{}.pwms_filt.h5".format(
        args.out_dir, os.path.basename(args.pvals_h5_file).split(".h5")[0])
    
    # read in pwms
    pwm_list = MotifSetManager.read_pwm_file(args.pwm_file)

    # read in pwm_metadata file
    pwm_metadata = pd.read_table(args.pwm_metadata_file, sep="\t")

    # look at expressed info (bool vector)
    tf_expressed = pwm_metadata[args.pwm_metadata_expr_col_key].notnull().values
    tf_expressed = np.reshape(
        tf_expressed, (1, 1, tf_expressed.shape[0]))
    
    # get the pval keys
    with h5py.File(args.pvals_h5_file, "r") as hf:
        pval_keys = hf[DataKeys.PWM_PVALS].keys()
    
    # go through pval keys and produce the sig pwms
    for pval_key in pval_keys:

        # load in pvals
        with h5py.File(args.pvals_h5_file, "r") as hf:
            pvals = hf[DataKeys.PWM_PVALS][pval_key][:]
            foreground_indices = hf[DataKeys.PWM_PVALS][pval_key].attrs[AttrKeys.TASK_INDICES]
            pwm_names = hf[args.pwm_scores_key].attrs[AttrKeys.PWM_NAMES]

        # convert to qvals
        pass_qval_thresh = threshold_by_qvalues(
            pvals, qval_thresh=args.qval_thresh, num_bins=50)

        # now filter by expression
        pass_expr_filter = np.multiply(
            pass_qval_thresh,
            tf_expressed)

        # save out
        with h5py.File(out_file, "w") as hf:
            out_key = "{}/{}".format(DataKeys.PWM_SIG_ROOT, pval_key)
            hf.create_dataset(out_key, data=pass_expr_filter)
            hf[DataKeys.PWM_SIG_ROOT].attrs["ensembl_ids"] = pwm_metadata[
                args.pwm_metadata_expr_col_key].values.astype(str)
            hf[DataKeys.PWM_SIG_ROOT].attrs["hgnc_ids"] = pwm_metadata[
                args.pwm_metadata_hgnc_col_key].values.astype(str)

    # STAGE 2 - correlation information
    _CORR_GROUP_KEY = "cor_filt"
    
    # read in RNA matrix
    rna_patterns = pd.read_table(args.rna_expression_file)
    rna_patterns["ensembl_ids"] = rna_patterns.index
    
    # now go through each foreground
    # remember the pval key is the target key
    for pval_key in pval_keys:

        # extract from h5 file
        other_targets = {}
        with h5py.File(args.pvals_h5_file, "r") as hf:
            targets = hf[pval_key][:]
            pwm_scores = hf[args.pwm_scores_key][:]
            foreground_indices = hf[DataKeys.PWM_PVALS][pval_key].attrs[AttrKeys.TASK_INDICES]
            for target, indices in args.other_targets:
                if len(indices) != 0:
                    other_targets[target] = hf[target][:,indices]
                else:
                    other_targets[target] = hf[target][:]

        # go through each foreground
        for foreground_idx in xrange(len(foreground_indices)):
            prefix = "{}-{}".format(pval_key, foreground_indices[foreground_idx])

            # extract subset
            example_indices = np.where(targets[:,foreground_indices[foreground_idx]] > 0)[0]
            example_scores = pwm_scores[example_indices]
            pwm_patterns = np.sum(example_scores, axis=0).transpose()

            # extract other keys
            subset_targets = {}
            for key in other_targets.keys():
                subset_targets[key] = np.mean(other_targets[key][example_indices], axis=0)
            
            # REMOVE LATER
            pwm_patterns = pwm_patterns[:,[0,2,3,4,5,6,7,8,9]]

            # set up df for matching to RNA
            pwms_for_matching = pd.DataFrame(pwm_patterns, index=pwm_names)
            pwms_for_matching["ensembl_ids"] = pwm_metadata[args.pwm_metadata_expr_col_key].values
            pwms_for_matching["hgnc_ids"] = pwm_metadata[args.pwm_metadata_hgnc_col_key].values
            pwms_for_matching["original_indices"] = range(pwm_patterns.shape[0])
            pwms_for_matching["pwm_names"] = pwms_for_matching.index

            # filter for those that are sig
            passes_sig_filter = np.all(pass_expr_filter[foreground_idx], axis=0)
            pass_indices = np.where(passes_sig_filter)[0]
            pwms_for_matching = pwms_for_matching.iloc[pass_indices]
            
            # filter for those that have expression (redundant, but make sure)
            pwms_for_matching = pwms_for_matching[pwms_for_matching["ensembl_ids"].notnull()]
            
            # and expand on RNA, matching ensembl expansion with hgnc expansion
            split_cols = ["ensembl_ids", "hgnc_ids"]
            split_char = ";"
            for split_col in split_cols:
                pwms_for_matching = pwms_for_matching.assign(
                    **{split_col: pwms_for_matching[split_col].str.split(split_char)})
            
            pwms_for_matching_expanded = pd.DataFrame({
                col: np.repeat(pwms_for_matching[col].values, pwms_for_matching[split_cols[0]].str.len())
                for col in pwms_for_matching.columns})    

            for split_col in split_cols:
                pwms_for_matching_expanded = pwms_for_matching_expanded.assign(
                    **{split_col: np.concatenate(pwms_for_matching[split_col].values)})[
                        pwms_for_matching.columns.tolist()]

            # move hgnc names to index
            pwms_for_matching_expanded = pwms_for_matching_expanded.set_index("hgnc_ids")

            # remove extraneous columns for the correlation analysis
            pwm_patterns = pwms_for_matching_expanded[
                pwms_for_matching_expanded.columns.difference(
                    ["ensembl_ids", "original_indices", "pwm_names"])]
            
            # now match with RNA
            matched_rna = rna_patterns[rna_patterns["ensembl_ids"].isin(
                pwms_for_matching_expanded["ensembl_ids"])]
            matched_rna = matched_rna.set_index("ensembl_ids")
            matched_rna = matched_rna.loc[pwms_for_matching_expanded["ensembl_ids"]]
            
            # calculate row-wise correlations
            pwm_rna_correlations = np.zeros((pwms_for_matching_expanded.shape[0]))
            for i in xrange(pwm_rna_correlations.shape[0]):
                corr_coef, pval = pearsonr(
                    pwm_patterns.values[i], matched_rna.values[i])
                pwm_rna_correlations[i] = corr_coef

            # and make a sig pwms vector
            sig_pwms = np.zeros((pwm_scores.shape[2])).astype(int)
            sig_pwms[pwms_for_matching_expanded["original_indices"].values] = 1
                
            # finally, all of this needs to be saved out
            _PREFIX_KEY = "{}/{}".format(_CORR_GROUP_KEY, prefix)
            _PWM_SIG_KEY = "{}/pwms.sig".format(_PREFIX_KEY)
            _PWM_PATTERN_KEY = "{}/pwm_patterns".format(_PREFIX_KEY)
            _RNA_PATTERN_KEY = "{}/rna_patterns".format(_PREFIX_KEY)
            _COR_KEY = "{}/correlations".format(_PREFIX_KEY)
            
            with h5py.File(out_file, "a") as hf:
                # pwm pattern
                hf.create_dataset(_PWM_SIG_KEY, data=sig_pwms)
                hf.create_dataset(_PWM_PATTERN_KEY, data=pwm_patterns.values)
                hf.create_dataset(_RNA_PATTERN_KEY, data=matched_rna.values)
                hf.create_dataset(_COR_KEY, data=pwm_rna_correlations)

                # TODO also save out non-expanded?

                # save out attributes
                hf[_PREFIX_KEY].attrs["ensembl_ids"] = pwms_for_matching_expanded["ensembl_ids"].values.astype(str)
                hf[_PREFIX_KEY].attrs["hgnc_ids"] = pwms_for_matching_expanded.index.values.astype(str)
                hf[_PREFIX_KEY].attrs["pwm_names"] = pwms_for_matching_expanded["pwm_names"].values.astype(str)
                
                # other keys
                for key in subset_targets.keys():
                    out_key = "{}/{}.mean".format(_PREFIX_KEY, key)
                    hf.create_dataset(out_key, data=subset_targets[key])

    # and plot
    plot_cmd = "plot-h5.pwm_x_rna.R {}".format(out_file)
    #os.system(plot_cmd)
    
    return



if __name__ == '__main__':
    main()
