#!/usr/bin/env python

"""
description: script to take a feature in the 
dataset, and provide back the positives and a matched
negative split (based on GC content, etc) in BED format

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
        "--labels_key", default="ATAC_LABELS",
        help="which label set to use for initial filtering")
    parser.add_argument(
        "--pwm_idx", type=int, default=0,
        help="index of pwm")
    parser.add_argument(
        "--raw_pwm_key", default=DataKeys.ORIG_SEQ_PWM_SCORES_SUM,
        help="key of PWM scores")
    parser.add_argument(
        "--weighted_pwm_key", default=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        help="key of weighted PWM scores")
    parser.add_argument(
        "--raw_pwm_pos_key", default=DataKeys.ORIG_SEQ_PWM_SCORES_THRESH,
        help="key of PWM scores with position info")
    parser.add_argument(
        "--weighted_pwm_pos_key", default=DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX,
        help="key of PWM scores with position info")
    parser.add_argument(
        "--gc_key", default=DataKeys.GC_CONTENT,
        help="key with GC content info")
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


def get_bed_from_metadata_list(
        examples,
        bed_file,
        offsets=None,
        #window=60,
        interval_key="active"):
    """make a bed file from list of metadata info
    """
    with open(bed_file, "w") as fp:
        for example_idx in range(examples.shape[0]):
            region_metadata = examples[example_idx]
            interval_types = region_metadata.split(";")
            interval_types = dict([
                interval_type.split("=")[0:2]
                for interval_type in interval_types])
            interval_string = interval_types[interval_key]

            chrom = interval_string.split(":")[0]
            start = int(interval_string.split(":")[1].split("-")[0])
            stop = int(interval_string.split("-")[1])

            # if using offsets, get point location
            if offsets is not None:
                start += offsets[example_idx] + 24
                stop = start + 1
                #start += (offsets[example_idx] - window/2)
                #stop = start + window
            
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


def build_matched_null_set(
        target_scores,
        background_scores,
        target_gc_scores,
        background_gc_scores,
        num_increments=5,
        rand_seed=1):
    """given a set of target scores and all scores, give back a matched
    set
    """
    # build target score intervals
    sorted_scores = np.sort(target_scores)
    intervals = np.interp(
        np.linspace(0, len(sorted_scores), num=num_increments),
        range(len(sorted_scores)),
        sorted_scores)
    print intervals.shape

    # build gc score intervals
    sorted_gc = np.sort(target_gc_scores)
    gc_intervals = np.interp(
        np.linspace(0, len(sorted_gc), num=num_increments),
        range(len(sorted_gc)),
        sorted_gc)
    print gc_intervals.shape

    # for each score interval
    target_counts_per_bin = []
    indices_per_bin = []
    for score_interval_idx in range(len(intervals)-1):
        score_min = intervals[score_interval_idx]
        score_max = intervals[score_interval_idx+1]
        #print score_min, score_max
        
        # get target examples in score range
        target_score_range_indices = np.where(
            (target_scores >= score_min) &
            (target_scores < score_max))[0]
        
        # get examples in score range
        score_range_indices = np.where(
            (background_scores >= score_min) &
            (background_scores < score_max))[0]
        #print background_scores.shape
        #print score_range_indices.shape
        
        # for each gc interval
        for gc_interval_idx in range(len(gc_intervals)-1):
            gc_min = gc_intervals[gc_interval_idx]
            gc_max = gc_intervals[gc_interval_idx+1]
            #print "  ", gc_min, gc_max
            
            # get examples in gc range
            target_gc_range_indices = np.where(
                (target_gc_scores >= gc_min) &
                (target_gc_scores < gc_max))[0]
            
            # get examples in gc range
            gc_range_indices = np.where(
                (background_gc_scores >= gc_min) &
                (background_gc_scores < gc_max))[0]

            # intersect to get those in score and GC range
            target_indices_in_bin = set(target_score_range_indices).intersection(
                set(target_gc_range_indices))
            target_counts_per_bin.append(len(target_indices_in_bin))
            #print len(target_indices_in_bin)
            
            background_indices_in_bin = set(score_range_indices).intersection(
                set(gc_range_indices))
            #print len(background_indices_in_bin)
            #background_indices_in_bin = background_indices_in_bin.difference(
            #    target_indices_in_bin)
            #print len(background_indices_in_bin)
            indices_per_bin.append(list(background_indices_in_bin))
            
    # for each bin, get scores from all scores
    print target_counts_per_bin
    print [len(val) for val in indices_per_bin]
    rand_state = RandomState(rand_seed)
    keep_indices = []
    print "NOTE REPLACE IS TRUE"
    for bin_idx in range(len(target_counts_per_bin)):
        target_count = target_counts_per_bin[bin_idx]
        if target_count == 0:
            continue
        indices = indices_per_bin[bin_idx]
        selected = rand_state.choice(indices, size=target_count, replace=True)
        keep_indices.append(selected)
    keep_indices = np.concatenate(keep_indices)
    
    return keep_indices



def get_matched_neg_set(data, to_match_df):
    """match GC content and PWM distribution strength
    """
    target_scores = to_match_df["raw_scores"]
    target_gc_scores = to_match_df["gc"]

    background_scores = data["raw_scores"]
    background_gc_scores = data["gc"]

    
    keep_indices = build_matched_null_set(
        target_scores,
        background_scores,
        target_gc_scores,
        background_gc_scores,
        num_increments=4,
        rand_seed=1)
    data_matched = data.iloc[keep_indices]
    
    return data_matched


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])
    prefix = "{}/{}".format(args.out_dir, args.prefix)

    # get labels
    labels = _get_data_from_h5_files(
        args.data_files, args.labels_key)
    if True:
        labels = labels[:,0]
    
    # confirm correct pwm
    with h5py.File(args.data_files[0], "r") as hf:
        print "testing", hf[args.raw_pwm_key].attrs["pwm_names"][args.pwm_idx]
    
    # get PWM raw scores and max index positions
    raw_scores = _get_data_from_h5_files(
        args.data_files, args.raw_pwm_key)[:,0,args.pwm_idx]
    raw_positional_scores = _get_data_from_h5_files(
        args.data_files, args.raw_pwm_pos_key)[:,0,:,args.pwm_idx]
    raw_max_positions = np.argmax(raw_positional_scores, axis=-1) + 12
    #if True:
    #    raw_max_positions += 420 + 12
    
    # get PWM weighted scores and index positions
    weighted_scores = _get_data_from_h5_files(
        args.data_files, args.weighted_pwm_key)[:,:,args.pwm_idx]
    if True:
        weighted_scores = weighted_scores[:,0] # just day 0 for now
    weighted_max_positions = _get_data_from_h5_files(
        args.data_files, args.weighted_pwm_pos_key)[:,args.pwm_idx,0].astype(int)
    weighted_max_positions -= 420
    
    # get GC content
    gc_content = _get_data_from_h5_files(
        args.data_files, args.gc_key)

    # get metadata
    metadata = _get_data_from_h5_files(
        args.data_files, DataKeys.SEQ_METADATA)[:,0]
    
    # make a df for easier handling and filter
    data = pd.DataFrame({
        "labels": labels,
        "raw_scores": raw_scores,
        "weighted_scores": weighted_scores,
        "raw_offset": raw_max_positions,
        "weighted_offset": weighted_max_positions,
        "gc": gc_content,
        "metadata": metadata})

    # filter for accessibility and also places where raw score exists
    data_filt = data[data["labels"] != 0]
    data_filt = data_filt[data_filt["raw_scores"] != 0]

    #print data_filt
    #quit()
    print data_filt[data_filt["weighted_scores"] > 0.05].shape
    #quit()
    
    # and split to positive and negative set
    data_filt_pos = data_filt[data_filt["weighted_scores"] > 0.01] # 0

    # save out to bed files and also build match files (for HINT)
    positives_bed_file = "{}/{}.impt_positive.bed".format(args.out_dir, args.prefix)
    get_bed_from_metadata_list(
        data_filt_pos["metadata"].values,
        positives_bed_file,
        offsets=data_filt_pos["weighted_offset"].values,
        interval_key="active")
    positives_match_file = "{}/{}.impt_positive.HINT.bed".format(args.out_dir, args.prefix)
    make_match_file = (
        "cat {} | "
        "sort -k1,1 -k2,2n | "
        "uniq | "
        "awk -F '\t' '{{ print $0\"\tCTCF\t10\t+\"}}' > "
        "{}").format(positives_bed_file, positives_match_file)
    os.system(make_match_file)

    data_filt_neg_all = data_filt[data_filt["weighted_scores"] == 0]
    data_filt_neg = get_matched_neg_set(data_filt_neg_all, data_filt_pos)
    
    negatives_bed_file = "{}/{}.impt_negative.bed".format(args.out_dir, args.prefix)
    get_bed_from_metadata_list(
        data_filt_neg["metadata"].values,
        negatives_bed_file,
        offsets=data_filt_neg["raw_offset"].values,
        interval_key="active")
    negatives_match_file = "{}/{}.impt_negative.HINT.bed".format(args.out_dir, args.prefix)
    make_match_file = (
        "cat {} | "
        "sort -k1,1 -k2,2n | "
        "uniq | "
        "awk -F '\t' '{{ print $0\"\tCTCF\t10\t+\"}}' > "
        "{}").format(negatives_bed_file, negatives_match_file)
    os.system(make_match_file)

    
    return None



if __name__ == "__main__":
    main()
