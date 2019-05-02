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
        "--synergy_only", action="store_true",
        help="filter for synergy only")
    parser.add_argument(
        "--synergy_file",
        help="h5 file with synergy scores")
    parser.add_argument(
        "--compare_keys", nargs="+", default=["ATAC_SIGNALS.NORM", "H3K27ac_SIGNALS.NORM"],
        help="which score keys to compare")
    parser.add_argument(
        "--compare_bigwigs", nargs="+", default=[],
        help="other bigwigs for extracting signals for comparison")
    parser.add_argument(
        "--map_chain_file",
        help="liftover file")
    parser.add_argument(
        "--eqtl_bed_file",
        help="eqtl BED file to see if any eQTLs in grammar/null regions (rsid must be in name col)")
    parser.add_argument(
        "--eqtl_effects_file",
        help="table with effect sizes")
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


def get_signal_from_bigwig(bed_file, bigwig_file, map_chain_file=None, liftover=True):
    """
    """
    # if liftover, first convert the bed file to correct coords
    out_dir = os.path.dirname(bed_file)
    tmp_bed = "{}/liftover.tmp.bed".format(out_dir)
    unmapped_file = "{}/unmapped.txt".format(out_dir)
    if liftover:
        liftover = "liftOver {} {} {} {}".format(
            bed_file, map_chain_file, tmp_bed, unmapped_file)
        os.system(liftover)
    else:
        tmp_bed = bed_file

    # give each region a name
    new_bed = "{}/signal.bed".format(out_dir)
    name_regions = (
        "cat {} | "
        "awk -F '\t' '{{ print $0\"\tregion\"NR }}' "
        "> {}").format(tmp_bed, new_bed)
    os.system(name_regions)
    
    # then bigWigAverageOverBed
    avg_signal_file = "{}/signal.avg.tmp.txt".format(out_dir)
    get_signal = "bigWigAverageOverBed {} {} {}".format(
        bigwig_file, new_bed, avg_signal_file)
    os.system(get_signal)

    # then read in data
    avg_signal = pd.read_csv(avg_signal_file, sep="\t", header=None)
    avg_signal = avg_signal.iloc[:,4].values
    #print "{}: median signal {}".format(bigwig_file, np.median(avg_signal))
    
    return avg_signal


def get_overlapping_variants(
        bed_file,
        variant_bed_file,
        lookup_table,
        variant_type="GTEx"):
    """get variants that are in BED regions as well as effect sizes
    """
    tmp_dir = os.path.dirname(bed_file)
    # get overlapping variants
    tmp_overlap_file = "{}/overlap.tmp.txt.gz".format(tmp_dir)
    overlap_cmd = (
        "bedtools intersect -wa -a {} -b {} | "
        "awk -F '\t' '{{ print $4 }}' | "
        "gzip -c > {}").format(
            variant_bed_file, bed_file, tmp_overlap_file)
    os.system(overlap_cmd)
    variant_ids = pd.read_csv(tmp_overlap_file, header=None).values[:,0]

    # get filtered set from lookup
    if variant_type == "GTEx":
        variant_results = lookup_table[lookup_table["variant_id"].isin(variant_ids)]
    else:
        variant_results = lookup_table[lookup_table["CausalSNP"].isin(variant_ids)]

    # get effect sizes
    if variant_type == "GTEx":
        col_key = "slope"
    else:
        col_key = "P VALUE"
    variant_effects = np.absolute(variant_results[col_key].values)
    #print "{}: {} variants, median effect {}".format(
    #    bed_file, variant_results.shape[0], np.median(variant_effects))
    
    return variant_effects


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


def get_results(data_indices, metadata, args, prefix="grammar.instances"):
    """collect all results that cover this bed file
    """
    # make bed file
    bed_file = "{}/{}.bed.gz".format(args.out_dir, prefix)
    get_bed_from_metadata_list(
        metadata[data_indices],
        bed_file,
        interval_key="active",
        merge=True)

    # save results to dict
    results = {}
    
    # get results for compare keys (ATAC, H3K27ac)
    for key in args.compare_keys:
        signal = _get_data_from_h5_files(
            args.score_files, key)
        results[key] = signal[data_indices]
        
    # get grammar variants
    variants = get_overlapping_variants(
        bed_file, args.eqtl_bed_file, args.eqtl_lookup)
    results["variants"] = variants
    
    # for each bigwig, get signal
    for bigwig_file in args.compare_bigwigs:
        bigwig_signal = get_signal_from_bigwig(
            bed_file,
            bigwig_file,
            map_chain_file=args.map_chain_file,
            liftover=True)
        results[os.path.basename(bigwig_file)] = bigwig_signal

    return results


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

    # read in grammar
    grammar = nx.read_gml(args.grammar)
    grammar_instances = grammar.graph["examples"].split(",")

    # to consider, subset those that are diff?
    if args.synergy_only:
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
    args.eqtl_lookup = pd.read_csv(args.eqtl_effects_file, sep="\t")
    
    # get grammar results
    results = get_results(
        grammar_indices, metadata, args, prefix="grammar.instances")
    #for key in sorted(results.keys()):
    #    print key, results[key].shape, np.median(results[key])
        
    # for each pwm in grammar, get null distributions and run tests
    null_results_all = []
    plot_data = None
    for pwm in grammar.nodes:
        print "\nRunning {}\n".format(pwm)
        pwm_idx = int(grammar.nodes[pwm]["pwmidx"])

        # get target scores (GC and pwms)
        target_gc_scores = gc_scores[grammar_indices]
        target_pwm_scores = pwm_raw_scores[grammar_indices,0,pwm_idx]
        background_pwm_scores = pwm_raw_scores[:,0,pwm_idx]

        # set up null store
        nulls = {}
        for key in results.keys():
            nulls[key] = []
            
        # build n null sets        
        for null_idx in range(args.n_null):

            # logging
            if null_idx % 10 == 0:
                print null_idx
            
            # get null set
            matched_background_indices = build_matched_null_set(
                target_pwm_scores,
                background_pwm_scores,
                target_gc_scores,
                gc_scores,
                rand_seed=null_idx)
            
            # get null results
            null_results = get_results(
                matched_background_indices,
                metadata,
                args,
                prefix="grammar.non_instances")
            #for key in sorted(null_results.keys()):
            #    print key, null_results[key].shape, np.median(null_results[key])
            #print ""
                
            # save out null results
            for key in sorted(null_results.keys()):
                nulls[key].append(null_results[key])

        # finally save out to overall list
        null_results_all.append(nulls)
                
        if False:
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

    # get medians, calc significance
    for key in sorted(results.keys()):

        file_prefix = "{}/results.{}".format(args.out_dir, key.replace(".", "_"))
        
        pval_thresh = 0.05
        print key

        grammar_results = results[key]
        grammar_median = np.median(grammar_results, axis=0)

        if len(grammar_median.shape) > 0:
            # split out
            for i in range(grammar_median.shape[0]):
                split_file_prefix = "{}.idx-{}".format(file_prefix, i)
                null_sigs = []
                null_median_medians = []
                save_nulls = []
                split_grammar_results = grammar_results[:,i]
                split_grammar_median = grammar_median[i]
                for null_idx in range(len(null_results_all)):
                    null_results = null_results_all[null_idx][key]
                    split_null_results = [
                        vals[:,i]
                        for vals in null_results]
                    split_null_medians = [
                        np.median(val, axis=0)
                        for val in split_null_results]
                    pval = get_sig(split_grammar_median, split_null_medians)
                    null_sigs.append(pval)
                    # get the median null dist
                    if len(split_null_medians) % 2 == 0:
                        split_null_medians = split_null_medians[1:]
                    null_median_median = np.median(split_null_medians)
                    null_median_medians.append(null_median_median)
                    save_idx = split_null_medians.index(null_median_median)
                    save_nulls.append(split_null_results[save_idx])

                #print split_grammar_median, null_median_medians, null_sigs
                filename = "{}.txt".format(split_file_prefix)
                save_str = "filename={};pwms={};grammar_avg={};null_avgs={},null_sigs={}".format(
                    filename,
                    grammar.nodes,
                    split_grammar_median,
                    null_median_medians,
                    null_sigs)
                print save_str
                save_results(
                    filename,
                    split_grammar_results,
                    save_nulls,
                    list(grammar.nodes),
                    save_str)
                
        else:
            null_sigs = []
            null_median_medians = []
            save_nulls = []
            for null_idx in range(len(null_results_all)):
                null_results = null_results_all[null_idx][key]
                null_medians = [
                    np.median(vals)
                    for vals in null_results]
                pval = get_sig(grammar_median, null_medians)
                null_sigs.append(pval)
                # get the median null dist
                if len(null_medians) % 2 == 0:
                    null_medians = null_medians[1:]
                null_median_median = np.median(null_medians)
                null_median_medians.append(null_median_median)
                save_idx = null_medians.index(null_median_median)
                save_nulls.append(null_results[save_idx])

            #print grammar_median, null_median_medians, null_sigs
            filename = "{}.txt".format(file_prefix)
            save_str = "filename={};pwms={};grammar_avg={};null_avgs={},null_sigs={}".format(
                filename,
                grammar.nodes,
                grammar_median,
                null_median_medians,
                null_sigs)
            print save_str
            save_results(
                filename,
                grammar_results,
                save_nulls,
                list(grammar.nodes),
                save_str)
    
    return None



if __name__ == "__main__":
    main()
