#!/usr/bin/env python

"""
description: script to compare pwm hits
in grammar regions vs non-grammar regions

"""

import os
import re
import sys
import h5py
import logging
import argparse

import networkx as nx
import numpy as np
import pandas as pd

from numpy.random import RandomState
from scipy.stats import ranksums

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
        "--raw_pwm_key", default=DataKeys.ORIG_SEQ_PWM_SCORES_SUM,
        help="key of PWM scores")
    parser.add_argument(
        "--weighted_pwm_key", default=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        help="key of weighted PWM scores")
    parser.add_argument(
        "--num_impt_bps_key", default="positive_importance_bp_sum")
    parser.add_argument(
        "--gc_key", default=DataKeys.GC_CONTENT,
        help="key with GC content info")
    parser.add_argument(
        "--compare_keys", nargs="+", default=[
            "ATAC_SIGNALS.NORM", "H3K27ac_SIGNALS.NORM"], #, "H3K4me1_SIGNALS.NORM"],
        help="which score keys to compare")
    parser.add_argument(
        "--synergy_file", # default None
        help="h5 file with synergy scores (if want to filter using synergy scores)")
    parser.add_argument(
        "--n_null", type=int, default=100,
        help="number of null distributions to try")
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
        rand_seed=1,
        replace=False):
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
        selected = rand_state.choice(indices, size=target_count, replace=replace)
        keep_indices.append(selected)
    keep_indices = np.concatenate(keep_indices)
    
    return keep_indices


def get_matched_neg_set(data, to_match_df, score_key="raw_scores", gc_key="gc", replace=False):
    """match GC content and PWM distribution strength
    """
    target_scores = to_match_df[score_key]
    target_gc_scores = to_match_df[gc_key]

    background_scores = data[score_key]
    background_gc_scores = data[gc_key]

    
    keep_indices = build_matched_null_set(
        target_scores,
        background_scores,
        target_gc_scores,
        background_gc_scores,
        num_increments=4,
        rand_seed=1,
        replace=replace)
    data_matched = data.iloc[keep_indices]
    
    return data_matched


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


def get_sig(results, null_results):
    """sort nulls, figure out where results fit in
    """
    null_results_sorted = np.sort(null_results)

    idx = np.searchsorted(null_results, results)

    pval = 1 - float(idx) / len(null_results)
    
    return pval


def save_results(
        filename,
        results,
        null_results,
        pwm_names,
        save_str):
    """save out to file with a comment line at front
    """
    data = pd.DataFrame(data=results, columns=["signal"])
    data["variable"] = "target"

    for i in range(len(null_results)):
        null_data = pd.DataFrame(data=null_results[i], columns=["signal"])
        null_data["variable"] = pwm_names[i]
        data = pd.concat([data, null_data], axis=0)
        
    with open(filename, "w") as fp:
        fp.write("#{}\n".format(save_str))
        data.to_csv(fp, index=False, sep="\t")

    # just make the plot each time?
    plot_file = "{}.pdf".format(filename.split(".txt")[0])
    plot_cmd = "Rscript /datasets/inference.2019-03-12/dmim.shuffle/quick_plot.R {} {}".format(
        filename, plot_file)
    os.system(plot_cmd)
    
    return


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])
    prefix = "{}/{}".format(args.out_dir, args.prefix)

    # set up summary files
    summary_files = []
    for i in range(len(args.compare_keys)):
        summary_file = "{}.{}.summary.txt".format(prefix, args.compare_keys[i])
        if os.path.isfile(summary_file):
            os.system("rm {}".format(summary_file))
        summary_files.append(summary_file)
    
    # read in grammar
    grammar = nx.read_gml(args.grammar)
    grammar_instances = grammar.graph["examples"].split(",")
    logging.info("grammar: {}".format(grammar.nodes))

    # get metadata and get grammar indices
    metadata = _get_data_from_h5_files(
        args.score_files,
        DataKeys.SEQ_METADATA)[:,0]
    grammar_indices = np.where(
        np.isin(metadata, grammar_instances))[0]
    logging.info("Grammar has {} examples".format(grammar_indices.shape[0]))
    
    # with scores, first select best task idx on GRAMMARS to pull out all scores
    logging.info("WARNING GGR SPECIFIC")
    task_selection_key = "ATAC_SIGNALS.NORM"
    interpretation_indices = [0,1,2,3,4,5,6,9,10,12]
    task_selection_scores = _get_data_from_h5_files(
        args.score_files, task_selection_key)[grammar_indices]
    task_selection_scores = task_selection_scores[:,interpretation_indices]
    task_selection_scores = np.median(task_selection_scores, axis=0)
    best_idx = np.argmax(task_selection_scores)

    # use dict for results
    results = {}

    # relevant info
    results["seq_metadata"] = metadata
    results["grammar"] = np.isin(metadata, grammar_instances).astype(int)
    results["gc"] = _get_data_from_h5_files(
        args.score_files, args.gc_key)
    results["num_weighted_bp"] = _get_data_from_h5_files(
        args.score_files, args.num_impt_bps_key)

    # scores to put in
    compare_indices = []
    for key in args.compare_keys:
        # for each, choose maximal task idx and use that one
        scores = _get_data_from_h5_files(
            args.score_files, key)
        score_best_idx = np.argmax(
            np.mean(scores[grammar_indices], axis=0))
        compare_indices.append(score_best_idx)
        save_key = "{}.taskidx-{}".format(key, score_best_idx)
        print save_key
        results[key] = scores[:,score_best_idx]
    
    # raw pwm scores at best idx
    pwms = list(grammar.nodes)
    raw_keys = [] # debug
    weighted_keys = [] # debug
    for pwm_i in range(len(pwms)):
        pwm_name = pwms[pwm_i]
        pwm_idx = int(grammar.nodes[pwm_name]["pwmidx"])

        # pull raw scores
        results_key = "RAW.{}".format(pwm_name)
        results[results_key] = _get_data_from_h5_files(
            args.score_files, args.raw_pwm_key)[:,0,pwm_idx]
        raw_keys.append(results_key)
        
        # pull weighted scores
        results_key = "WEIGHTED.{}.taskidx-{}".format(pwm_name, best_idx)
        results[results_key] = _get_data_from_h5_files(
            args.score_files, args.weighted_pwm_key)[:,best_idx,pwm_idx]
        weighted_keys.append(results_key)
        
    # set up as df
    data = pd.DataFrame(results)

    # count key motifs (0,1,2)
    data["num_grammar_motifs"] = np.sum(data[raw_keys].values > 0, axis=1)
        
    # get positive set
    data_grammar = data[data["grammar"] == 1]
    data_grammar = data_grammar[
        np.all(data_grammar[weighted_keys].values > 0, axis=1)]

    # set up negative set
    data_neg_all = data[data["grammar"] != 1]
    data_neg_all = data[data["num_grammar_motifs"] == 1]

    # try select top 1000 by weighted impt
    # TODO check this
    if True:
        for pwm_i in range(len(pwms)):
            pwm_name = pwms[pwm_i]
            score_key = "WEIGHTED.{}.taskidx-{}".format(pwm_name, best_idx)
            cutoff = np.percentile(data_grammar[score_key].values, 10)
            print pwm_name, score_key, cutoff
            data_grammar = data_grammar[data_grammar[score_key] > cutoff]
        
    # for each pwm, get a null distribution from negative set that matches positive set
    for i in range(len(args.compare_keys)):
        compare_key = args.compare_keys[i]
        for pwm_i in range(len(pwms)):
            pwm_name = pwms[pwm_i]
            score_key = raw_keys[pwm_i]

            # only take regions with active motifs
            data_neg_all_pwm = data_neg_all[data_neg_all[score_key] > 0]
            try:
                data_neg_matched = get_matched_neg_set(
                    data_neg_all_pwm,
                    data_grammar,
                    score_key=score_key,
                    replace=True)
            except:
                print ">>NO NULL"
                return None

            # just output diff for now, and figure out synergy/additive/buffer later
            diff = np.median(data_grammar[compare_key].values) - np.median(data_neg_matched[compare_key].values)
                
            # wilcoxon rank sums (compare distributions)
            _, pval = ranksums(
                data_grammar[compare_key].values,
                data_neg_matched[compare_key].values)
            
            # save out
            if pwm_i == 1:
                print_pwms = pwms[::-1]
            else:
                print_pwms = pwms

            print_pwms = [re.sub("HCLUST-\d+_", "", pwm_name) for pwm_name in print_pwms]
            print_pwms = [re.sub(".UNK.0.A", "", pwm_name) for pwm_name in print_pwms]
                
            # only write out 2s?
            with open(summary_files[i], "a") as fp:
                out_str = "{}\t{}\t{}\t{}\n".format(
                    "\t".join(print_pwms),
                    best_idx,
                    #pwm_name,
                    diff,
                    pval)
                fp.write(out_str)
    
    return None



if __name__ == "__main__":
    main()
