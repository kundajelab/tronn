#!/usr/bin/env python

"""
description: script to evaluate performance
given specific features in dataset and
external (or internal) annotation set (BED file)

"""

import os
import sys
import h5py
import logging
import argparse

import networkx as nx
import numpy as np
import pandas as pd

from numpy.random import RandomState

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from tronn.util.scripts import setup_run_logs
from tronn.util.utils import DataKeys


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # dataset file, features to compare, bed file
    parser.add_argument(
        "--data_files", nargs="+",
        help="h5 files with information desired")
    parser.add_argument(
        "--features", nargs="+",
        help="which features to use in analyses")
    parser.add_argument(
        "--positives",
        help="bed file of regions that are positive")
    parser.add_argument(
        "--translate_positives", action="store_true",
        help="set if you need to change genome reference")
    parser.add_argument(
        "--map_chain_file",
        help="map chain file for conversion")
    parser.add_argument(
        "--out_dir", "-o", dest="out_dir", type=str,
        default="./",
        help="out dir")
    parser.add_argument(
        "--prefix",
        help="file prefix")
    
    # parse args
    args = parser.parse_args()

    return args


def get_bed_from_metadata_list(examples, bed_file, interval_key="active"):
    """make a bed file from list of metadata info
    """
    with open(bed_file, "w") as fp:
        for region_metadata in examples:
            interval_types = region_metadata.split(";")
            interval_types = dict([
                interval_type.split("=")[0:2]
                for interval_type in interval_types])
            interval_string = interval_types[interval_key]

            chrom = interval_string.split(":")[0]
            start = interval_string.split(":")[1].split("-")[0]
            stop = interval_string.split("-")[1]
            fp.write("{}\t{}\t{}\n".format(chrom, start, stop))

    return


def _get_data_from_h5_files(h5_files, key):
    """get data concatenated
    """
    data = []
    for file_idx in range(len(h5_files)):
        h5_file = h5_files[file_idx]
        with h5py.File(h5_file, "r") as hf:
            data.append(hf[key][:])
    data = np.concatenate(data)
    
    return data


def auprc(labels, probs):
    """Wrapper for sklearn AUPRC

    Args:
      labels: 1D vector of labels
      probs: 1D vector of probs

    Returns:
      auprc: area under precision-recall curve
    """
    pr_curve = precision_recall_curve(labels, probs)
    precision, recall = pr_curve[:2]
    
    return auc(recall, precision)


def make_recall_at_fdr(fdr):
    """Construct function to get recall at FDR
    
    Args:
      fdr: FDR value for precision cutoff

    Returns:
      recall_at_fdr: Function to generate the recall at
        fdr fraction (number of positives
        correctly called as positives out of total 
        positives, at a certain FDR value)
    """
    def recall_at_fdr(labels, probs):
        pr_curve = precision_recall_curve(labels, probs)
        precision, recall = pr_curve[:2]
        return recall[np.searchsorted(precision - fdr, 0)]
        
    return recall_at_fdr



def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])
    prefix = "{}/{}".format(args.out_dir, args.prefix)

    # convert postives as needed
    if args.translate_positives:
        positives_bed_file = "{}/positives.translated.tmp.bed".format(
            args.out_dir)
        unmapped_file = "{}/unmapped.tmp.txt".format(args.out_dir)
        assert args.map_chain_file is not None
        liftover = "liftOver {} {} {} {}".format(
            args.positives, args.map_chain_file, positives_bed_file, unmapped_file)
        print liftover
        os.system(liftover)
    else:
        positives_bed_file = args.positives
    
    # get metadata
    metadata = _get_data_from_h5_files(
        args.data_files, DataKeys.SEQ_METADATA)[:,0]

    # get scores
    scores = []
    for feature_info in args.features:
        key, indices = feature_info.split("=")
        indices = [int(val) for val in indices.split(",")]
        feature_scores = _get_data_from_h5_files(
            args.data_files, key)
        for index in indices:
            feature_scores = feature_scores[:,index]
        scores.append(feature_scores)
    scores = np.stack(scores, axis=1)

    # make a df
    scores_df = pd.DataFrame(data=scores, columns=args.features)
    scores_df["metadata"] = metadata
    
    # make a bed file for the metadata
    features_bed_file = "{}/features.bed".format(args.out_dir)
    get_bed_from_metadata_list(
        scores_df["metadata"].values,
        features_bed_file,
        interval_key="active")

    # intersect and get counts
    count_results_file = "{}/counts.txt.gz".format(args.out_dir)
    intersect_cmd = (
        "bedtools intersect -c -a {} -b {} | "
        "gzip -c > {}").format(
            features_bed_file,
            positives_bed_file,
            count_results_file)
    print intersect_cmd
    os.system(intersect_cmd)

    counts = pd.read_csv(count_results_file, sep="\t", header=None)
    scores_df["labels"] = (counts.iloc[:,3] > 0).astype(int)
    print np.sum(scores_df["labels"].values) / float(scores_df.shape[0])
    
    # save out the df
    results_file = "{}/scores.txt.gz".format(args.out_dir)
    scores_df.to_csv(results_file, sep="\t", compression="gzip")
    
    # look at stats
    for key in args.features:
        print key
        scores = scores_df[key].values
        labels = scores_df["labels"].values

        #print np.sum(labels) / float(labels.shape[0])
        
        print "AUPRC", auprc(labels, scores)
        print "AUROC", roc_auc_score(labels, scores)
        #recall_at_fdr_fn = make_recall_at_fdr(0.50)
        #print "recall at fdr", recall_at_fdr_fn(labels, scores)
        
        
    return None



if __name__ == "__main__":
    main()
