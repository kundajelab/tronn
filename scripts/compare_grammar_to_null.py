#!/usr/bin/env python

"""
description: script to compare pwm hits
in grammar regions vs non-grammar regions

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

from tronn.util.scripts import setup_run_logs
from tronn.util.utils import DataKeys


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    parser.add_argument(
        "--grammar",
        help="grammar gml file")
    parser.add_argument(
        "--score_files", nargs="+",
        help="h5 file with pwm scores (raw sequence)")
    parser.add_argument(
        "--synergy_file",
        help="h5 file with synergy scores")
    parser.add_argument(
        "--compare_keys", nargs="+", default=["ATAC_SIGNALS.NORM"],
        help="which score keys to compare")
    parser.add_argument(
        "--compare_bigwigs", nargs="+",
        help="other bigwigs for extracting signals for comparison")
    parser.add_argument(
        "--eqtl_bed_file",
        help="eqtl BED file to see if any eQTLs in grammar/null regions (rsid must be in name col)")
    parser.add_argument(
        "--eqtl_effects_file",
        help="table with effect sizes")
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    parser.add_argument(
        "--prefix",
        help="file prefix")
    
    # parse args
    args = parser.parse_args()

    return args


def get_bed_from_metadata_list(examples, bed_file, interval_key="active", merge=True):
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

    if merge:
        tmp_bed_file = "{}.tmp.bed".format(bed_file.split(".bed")[0])
        os.system("mv {} {}".format(bed_file, tmp_bed_file))
        os.system((
            "cat {} | "
            "sort -k1,1 -k2,2n | "
            "bedtools merge -i stdin | "
            "gzip -c > {}").format(
                tmp_bed_file, bed_file))
        os.system("rm {}".format(tmp_bed_file))

    return


def get_overlapping_variants(bed_file, variant_bed_file, lookup_table):
    """get variants that are in BED regions as well as effect sizes
    """
    tmp_overlap_file = "overlap.tmp.txt.gz"
    overlap_cmd = (
        "bedtools intersect -wa -a {} -b {} | "
        "awk -F '\t' '{{ print $4 }}' | "
        "gzip -c > {}").format(
            variant_bed_file, bed_file, tmp_overlap_file)
    print overlap_cmd
    os.system(overlap_cmd)

    # read in file
    variant_ids = pd.read_csv(tmp_overlap_file, header=None).values[:,0]

    # get filtered set from lookup
    if False:
        variant_results = lookup_table[lookup_table["variant_id"].isin(variant_ids)]
    else:
        variant_results = lookup_table[lookup_table["CausalSNP"].isin(variant_ids)]

    if False:
        col_key = "slope"
    else:
        col_key = "P VALUE"
    #print lookup_table.columns
    #print list(variant_results["slope"].values)
    print variant_results.shape, np.median(np.absolute(variant_results[col_key].values))
    
    return None


def build_score_matched_bins(target_scores, full_scores, num_increments=21):
    """build bins
    """
    # TODO ideally build intervals for both GC and scores...
    # then to build bins, do a nested loop?
    # for score interval, for gc interval:
    # collect all target that match <- track count as num_target
    # collect all that match that interval <- select num_target from that set
    # trick part is then you need to match the num target in that set
    # so then you want to do min/max scores and divide equally
    
    # build intervals
    sorted_scores = np.sort(target_scores)
    intervals = np.interp(
        np.linspace(0, len(sorted_scores), num=num_increments),
        range(len(sorted_scores)),
        sorted_scores)

    # go through bins and collect sequences in bins
    binned_indices = []
    for i in range(len(intervals)-1):
        range_min = intervals[i]
        range_max = intervals[i+1]
        full_indices = np.where(
            (full_scores >= range_min) &
            (full_scores < range_max))[0]
        binned_indices.append(full_indices)
    
    return binned_indices


def build_matched_null_set(
        target_scores,
        background_scores,
        target_gc_scores,
        background_gc_scores,
        num_increments=11,
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

    # build gc score intervals
    sorted_gc = np.sort(target_gc_scores)
    gc_intervals = np.interp(
        np.linspace(0, len(sorted_gc), num=num_increments),
        range(len(sorted_gc)),
        sorted_gc)

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
    rand_state = RandomState(rand_seed)
    keep_indices = []
    for bin_idx in range(len(target_counts_per_bin)):
        target_count = target_counts_per_bin[bin_idx]
        if target_count == 0:
            continue
        indices = indices_per_bin[bin_idx]
        selected = rand_state.choice(indices, size=target_count, replace=False)
        keep_indices.append(selected)
    keep_indices = np.concatenate(keep_indices)
    
    return keep_indices


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


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])
    prefix = "{}/{}".format(args.out_dir, args.prefix)

    # read in grammar
    grammar = nx.read_gml(args.grammar)

    # get grammar regions
    grammar_instances = grammar.graph["examples"].split(",")

    # to consider, subset those that are diff?
    if True:
        with h5py.File(args.synergy_file, "r") as hf:
            synergy_diff = hf["{}.0".format(DataKeys.SYNERGY_DIFF_SIG)][:]
            diff_indices = np.where(
                np.any(synergy_diff!=0, axis=1))[0]
            synergy_metadata = hf[DataKeys.SEQ_METADATA][:,0][diff_indices]
        synergy_metadata = synergy_metadata[np.isin(synergy_metadata, grammar_instances)]
        grammar_instances = synergy_metadata
    
    # get metadata and get grammar indices
    metadata = _get_data_from_h5_files(
        args.score_files,
        DataKeys.SEQ_METADATA)[:,0]
    grammar_indices = np.where(
        np.isin(metadata, grammar_instances))[0]
    
    # get pwm scores and GC content
    pwm_raw_scores = _get_data_from_h5_files(
        args.score_files,
        DataKeys.ORIG_SEQ_PWM_SCORES_SUM)
    gc_scores = _get_data_from_h5_files(
        args.score_files,
        DataKeys.GC_CONTENT)

    # load eqtl table
    eqtl_lookup = pd.read_csv(args.eqtl_effects_file, sep="\t")
    
    # TODO generate grammar BED file
    # TODO note right now considering all regions
    grammar_tmp_bed = "grammar.instances.bed.gz"
    get_bed_from_metadata_list(
        metadata[grammar_indices],
        grammar_tmp_bed,
        interval_key="active",
        merge=True)
    get_overlapping_variants(
        grammar_tmp_bed, args.eqtl_bed_file, eqtl_lookup)
    
    # for each pwm in grammar:
    pwms = grammar.nodes
    plot_data = None
    for pwm in pwms:
        pwm_idx = int(grammar.nodes[pwm]["pwmidx"])

        # get target scores (GC and pwms)
        target_gc_scores = gc_scores[grammar_indices]
        target_pwm_scores = pwm_raw_scores[grammar_indices,0,pwm_idx]
        background_pwm_scores = pwm_raw_scores[:,0,pwm_idx]
        
        # get matched null set
        matched_background_indices = build_matched_null_set(
            target_pwm_scores,
            background_pwm_scores,
            target_gc_scores,
            gc_scores,
            rand_seed=1)
        
        # TODO generate matched null BED file(s)?
        # TODO should i consider bootstrapped null
        
        # look at relevant keys
        key = "ATAC_SIGNALS.NORM"
        #key = "H3K27ac_SIGNALS.NORM"
        #key = "H3K4me1_SIGNALS.NORM"
        #key = "ZNF750_LABELS"
        signal = _get_data_from_h5_files(
            args.score_files, key)
        target_signal = signal[grammar_indices]
        background_signal = signal[matched_background_indices]

        # quick test save for plot
        test_idx = 10 # 10
        plot_background = background_signal[:,test_idx]
        plot_background_data = pd.DataFrame(data=plot_background, columns=["signal"])
        plot_background_data["variable"] = pwm

        if plot_data is None:
            plot_data = plot_background_data.copy()
        else:
            plot_data = pd.concat([plot_data, plot_background_data], axis=0)
        
        print np.mean(target_signal, axis=0)
        print np.mean(background_signal, axis=0)
        
        print ""

        # test variants
        null_tmp_bed = "grammar.non_instances.bed.gz"
        get_bed_from_metadata_list(
            metadata[matched_background_indices],
            null_tmp_bed,
            interval_key="active",
            merge=True)
        get_overlapping_variants(
            null_tmp_bed, args.eqtl_bed_file, eqtl_lookup)

        
    plot_target = target_signal[:,test_idx]
    plot_target = pd.DataFrame(data=plot_target, columns=["signal"])
    plot_target["variable"] = "target"
    plot_data = pd.concat([plot_data, plot_target], axis=0)

    plot_data.to_csv("test_atac.txt", index=False, sep="\t")

    #with h5py.File(args.score_files[0], "r") as hf:a
    #    for key in sorted(hf.keys()): print key, hf[key].shape
    
    
    
    return None



if __name__ == "__main__":
    main()
