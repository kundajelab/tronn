"""description: code to calculate perturbation effects after synergy
"""

import os
import re
import sys
import glob
import h5py
import logging
import argparse

import numpy as np
import networkx as nx

from scipy.stats import wilcoxon
from scipy.stats import describe

from tronn.interpretation.combinatorial import setup_combinations
from tronn.interpretation.networks import attach_mut_logits
from tronn.interpretation.networks import attach_data_summary
from tronn.interpretation.networks import stringize_nx_graph
from tronn.stats.nonparametric import run_delta_permutation_test
from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys

from tronn.util.scripts import setup_run_logs


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="calculate mutagenesis effects")

    # required args
    parser.add_argument(
        "--synergy_file",
        help="h5 file with synergy results")
    parser.add_argument(
        "--calculations", nargs="+", default=[],
        help="pairwise calculations to perform, in the format of indices to look at (ex 110 to look at first two)")
    parser.add_argument(
        "--interpretation_indices", nargs="+", default=[],
        help="use to reduce tasks being considered")
    parser.add_argument(
        "--refine", action="store_true",
        help="get regions with sig synergy and save out new gml")
    
    # out
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    parser.add_argument(
        "--prefix",
        help="prefix for files")
    
    # parse args
    args = parser.parse_args()

    return args


def parse_calculation_strings_OLD(args):
    """form the strings into arrays
    """
    calculations = []
    for calculation in args.calculations:
        calculation = calculation.split("/")
        foreground = np.fromstring(
            ",".join(calculation[0].replace("x", "0")), sep=",")
        background = np.fromstring(
            ",".join(calculation[1].replace("x", "0")), sep=",")
        calculations.append((foreground, background))
    args.calculations = calculations
    
    return None

def parse_calculation_strings(args):
    """form the strings into arrays
    """
    calculations = []
    for calculation in args.calculations:
        calc_array = np.fromstring(
            ",".join(calculation.replace("x", "0")), sep=",")
        calculations.append(calc_array)
    args.calculations = calculations
    
    return None


def _setup_output_string(pwm_names, presence):
    """set up string to be more human readable
    """
    presence_strings = []
    for pwm_i in xrange(len(presence)):
        if presence[pwm_i] != 0:
            presence_strings.append("{}+".format(pwm_names[pwm_i]))
        else:
            presence_strings.append("{}-".format(pwm_names[pwm_i]))

    return presence_strings

def _make_conditional_string(foreground_strings, background_strings):
    """make string into conditional
    """
    changing = []
    conditioned_on = []
    for i in xrange(len(foreground_strings)):
        foreground_presence = foreground_strings[i]
        background_presence = background_strings[i]

        if foreground_presence == background_presence:
            conditioned_on.append(foreground_presence)
        else:
            changing.append(foreground_presence)

    assert len(changing) == 1

    # make string
    #new_string = "f({} | {})".format(changing[0], ",".join(conditioned_on))
    new_string = "f(seq | do({}), {})".format(changing[0], ",".join(conditioned_on))
    
    return new_string


def build_graph(
        h5_file,
        differential,
        sig_pwms_names_full,
        sig_pwms_names,
        min_sig=3):
    """build nx graph
    """
    # get examples
    differential = np.greater_equal(np.sum(differential != 0, axis=1), min_sig)
    with h5py.File(h5_file, "r") as hf:
        example_metadata = hf[DataKeys.SEQ_METADATA][:,0]
    examples = example_metadata[np.where(differential)[0]]
    num_examples = len(examples)

    # set up graph
    graph = nx.MultiDiGraph()
    graph.graph["examples"] = examples
    graph.graph["numexamples"] = num_examples
    
    # add nodes
    nodes = []
    for node_idx in range(len(sig_pwms_names_full)):
        node = [
            sig_pwms_names_full[node_idx],
            {"examples": examples,
             "numexamples": num_examples,
             "mutidx": node_idx+1}]
        nodes.append(node)
    graph.add_nodes_from(nodes)
        
    # add edges
    edges = []
    for node_idx in range(1, len(sig_pwms_names_full)):
        edge_attrs = {
            "examples": examples,
            "numexamples": num_examples,
            "edgetype": "directed"}
        edge = (
            sig_pwms_names_full[node_idx-1],
            sig_pwms_names_full[node_idx],
            edge_attrs)
        edges.append(edge)
    graph.add_edges_from(edges)

    return graph


def _calculate_expected_v_actual(calculation, indices, combinations, outputs):
    """calculate expected and actual
    """
    # 11
    one_one_combo = calculation
    one_one_idx = np.where(
        (np.transpose(combinations) == one_one_combo).all(axis=1))[0][0]
    one_one_vals = outputs[:,one_one_idx]

    # 01
    zero_one_combo = np.array(calculation)
    zero_one_combo[indices[0]] = 0
    zero_one_idx = np.where(
        (np.transpose(combinations) == zero_one_combo).all(axis=1))[0][0]
    zero_one_vals = outputs[:,zero_one_idx]

    # 10
    one_zero_combo = np.array(calculation)
    one_zero_combo[indices[1]] = 0
    one_zero_idx = np.where(
        (np.transpose(combinations) == one_zero_combo).all(axis=1))[0][0]
    one_zero_vals = outputs[:,one_zero_idx]

    # 00
    zero_zero_combo = np.array(zero_one_combo)
    zero_zero_combo[indices[1]] = 0
    zero_zero_idx = np.where(
        (np.transpose(combinations) == zero_zero_combo).all(axis=1))[0][0]
    zero_zero_vals = outputs[:,zero_zero_idx]

    # get expected: (01 - 00) + (10 - 00)
    expected = (zero_one_vals - zero_zero_vals) + (one_zero_vals - zero_zero_vals)

    # get actual: 11- 00
    actual = one_one_vals - zero_zero_vals
    
    return expected, actual



def main():
    """run calculations
    """
    # set up args
    args = parse_args()
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])
    out_prefix = "{}/{}".format(args.out_dir, args.prefix)

    # TMP: clean up out dir
    #os.system("rm {}/*".format(args.out_dir))

    # params set up
    min_pwm_dist = 12 # ignore PWMs that overlap (likely same importance scores contributing)
    stdev_thresh = 2.0
    interpretation_indices = [int(val) for val in args.interpretation_indices]
    parse_calculation_strings(args)

    # read in data
    with h5py.File(args.synergy_file, "r") as hf:
        outputs = hf[DataKeys.MUT_MOTIF_LOGITS][:] # {N, mutM_combos, logit} NOTE: 0 is endogenous, 3 is double mut
        sig_pwms_names = hf[DataKeys.MUT_MOTIF_LOGITS].attrs[AttrKeys.PWM_NAMES]
        positions = hf[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT][:] # {N, 4, 2}
        distances = np.max(positions[:,-1,:], axis=-1) - np.min(positions[:,-1,:], axis=-1) # -1 is where positions are marked (because they were marked for double mut)
    distance_filt_indices = np.where(distances >= min_pwm_dist)[0]
    
    # adjust interpretation indices if needed
    if len(interpretation_indices) == 0:
        interpretation_indices = range(outputs.shape[2])

    # clean up names
    sig_pwms_names_full = sig_pwms_names
    sig_pwms_names = [
        re.sub(r"HCLUST-\d+_", "", pwm_name)
        for pwm_name in sig_pwms_names]
    sig_pwms_names = [
        re.sub(r".UNK.0.A", "", pwm_name)
        for pwm_name in sig_pwms_names]
    
    # set up combination matrix
    num_mut_motifs = len(sig_pwms_names)
    combinations = setup_combinations(num_mut_motifs)
    combinations = 1 - combinations

    # set up results arrays
    num_calcs = len(args.calculations)
    results = np.zeros((
        outputs.shape[0],
        num_calcs,
        outputs.shape[2], 3)) # {N, calc, task, 3} where 3 is actual, exp, diff
    example_sigs = np.zeros((outputs.shape[0], num_calcs, outputs.shape[2])) # {N, calc, task}
    
    # set up calculation index sets
    index_sets = []
    for i in range(num_calcs):
        indices = np.where(args.calculations[i]==1)[0]
        assert len(indices) == 2
        index_sets.append(indices)

    # set up summary file log
    summary_file = "{}/{}.interactions.txt".format(args.out_dir, args.prefix)
    header_str = "pwm1\tpwm2\tnum_examples\tsig\tbest_task_index\tactual\texpected\tdiff\tpval\tcategory\n"
    with open(summary_file, "w") as fp:
        fp.write(header_str)

    # calculate interaction scores
    for calc_i in range(num_calcs):
        indices = index_sets[calc_i]

        # calculate
        expected, actual = _calculate_expected_v_actual(
            args.calculations[i], indices, combinations, outputs)
        diff = actual - expected

        # put results into array
        results[:,calc_i,:,0] = actual
        results[:,calc_i,:,1] = expected
        results[:,calc_i,:,2] = diff
        
        # run statistical test ONLY on those that pass distance filter
        actual = actual[distance_filt_indices]
        expected = expected[distance_filt_indices]
        diff = diff[distance_filt_indices]
        pvals = np.apply_along_axis(wilcoxon, 0, diff)[1]
            
        # determine if any are sig and adjust label indices accordingly
        # if non sig, look at all label indices
        label_mask = np.array(
            [1 if val in interpretation_indices else 0
             for val in range(diff.shape[1])])
        label_indices = np.arange(diff.shape[1])
        sig_tasks = (pvals < 0.05) * label_mask
        sig_task_indices = np.where(sig_tasks)[0]
        if sig_task_indices.shape[0] != 0:
            label_indices = label_indices[sig_task_indices]
            sig = 1
        else:
            label_indices = label_indices[interpretation_indices]
            sig = 0

        # filter for just label indices now
        actual = actual[:, label_indices]
        expected = expected[:, label_indices]
        diff = diff[:, label_indices]
        pvals_selected = pvals[label_indices]
        
        # determine best idx (highest scoring endogenous among label indices)
        actual_mean = np.mean(actual, axis=0) # {tasks}
        expected_mean = np.mean(expected, axis=0)
        diff_mean = np.mean(diff, axis=0)
        
        best_idx = np.argmax(actual_mean)
        label_idx = label_indices[best_idx]

        # determine whether synergy/additive/buffer
        category = "additive"
        if sig == 1:
            if diff_mean[best_idx] > 0:
                category = "synergy"
            else:
                category = "buffer"

        # generate results string
        results_str = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            "\t".join(sig_pwms_names),
            actual.shape[0],
            sig,
            label_indices[best_idx],
            actual_mean[best_idx],
            expected_mean[best_idx],
            diff_mean[best_idx],
            pvals_selected[best_idx],
            category)
        print results_str
        
        # save out results str with header
        with open(summary_file, "a") as fp:
            fp.write(results_str)

        # get sig examples
        outlier_thresholds = np.percentile(
            np.abs(diff), 75, axis=0, keepdims=True) # outlier filter (stabilize)
        diffs_filt = np.where(
            np.greater(np.abs(diff), outlier_thresholds),
            0, diff) # zero out beyond outliers
        diffs_filt = np.where(
            np.isclose(diffs_filt, 0),
            np.nan, diffs_filt) # nan to ignore in mean/std calcs
        stdevs = np.nanstd(diffs_filt, axis=0, keepdims=True)
        means = np.nanmean(diffs_filt, axis=0, keepdims=True)

        print stdevs.shape
        print means.shape
        quit()
        
        # differential cutoff: one tailed
        # TODO figure out how to adjust for label indices
        if np.mean(diff) >= 0:
            example_sigs[:,calc_i] = results[:,calc_i,2] > stdev_thresh * stdevs
        else:
            example_sigs[:,calc_i] = results[:,calc_i,2] < -stdev_thresh * stdevs
            differential[np.abs(distances) < min_pwm_dist] = 0
        example_sigs[:,]

        

    # after done with all calcs:
    # save results (actual/expected/diff)
    score_key = DataKeys.SYNERGY_SCORES
    with h5py.File(args.synergy_file, "a") as hf:
        if hf.get(score_key) is not None:
            del hf[score_key]
        hf.create_dataset(score_key, data=results)
        # TODO save out plot labels?
        #hf[score_key].attrs[AttrKeys.PLOT_LABELS] = labels

    # save out distances
    dist_key = DataKeys.SYNERGY_DIST
    with h5py.File(args.synergy_file, "a") as hf:
        distances = np.zeros((results.shape[0], num_calcs))
        for i in range(num_calcs):
            distances[:,i] = np.squeeze(distances)
        if hf.get(dist_key) is not None:
            del hf[dist_key]
        hf.create_dataset(dist_key, data=distances)

    # save out differentials
    synergy_sig_key = DataKeys.SYNERGY_DIFF_SIG
    with h5py.File(args.synergy_file, "a") as hf:
        if hf.get(synergy_sig_key) is not None:
            del hf[synergy_sig_key]
        hf.create_dataset(synergy_sig_key, data=example_sigs)
        #hf[synergy_sig_key].attrs[AttrKeys.PLOT_LABELS] = labels

    quit()
    # analyze max distance of synergistic interaction
    diff_indices = np.greater_equal(np.sum(differential!=0, axis=1), 1)
    diff_indices = np.where(diff_indices)[0]
    diff_distances = distances[diff_indices] # {N}
    max_dist = np.percentile(diff_distances, 95)
    max_dist_key = DataKeys.SYNERGY_MAX_DIST
    with h5py.File(args.synergy_file, "a") as hf:
        if hf.get(max_dist_key) is not None:
            del hf[max_dist_key]
        hf.create_dataset(max_dist_key, data=max_dist)

    # analyze pwm score strength
    pwm_strength_key = "pwms.strength"
    with h5py.File(args.synergy_file, "a") as hf:
        if hf.get(pwm_strength_key) is not None:
            del hf[pwm_strength_key]
        pwm_scores = np.amin(
            hf[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT][:,-1], axis=1)
        hf.create_dataset(
            pwm_strength_key,
            data=pwm_scores)

    # plot
    min_dist = 12 # 12
    plot_cmd = "plot-h5.synergy_results.2.R {} {} {} {} {} {} {} \"\" {}".format(
        args.synergy_file,
        score_key,
        synergy_diffs_key,
        synergy_sig_key,
        dist_key,
        max_dist_key,
        out_prefix,
        min_dist)
    print plot_cmd
    os.system(plot_cmd)

    # refine:
    #if args.refine:
    if False:   
        
        # do for each calculation set
        
        # take these new regions and save out gml
        # get logits, atac signals, delta logits, etc
        graph = build_graph(
            args.synergy_file,
            differential,
            sig_pwms_names_full,
            sig_pwms_names)

        # attach delta logits
        [graph] = attach_mut_logits([graph], args.synergy_file)

        # attach other keys
        other_keys = ["logits.norm", "ATAC_SIGNALS.NORM"]
        for key in other_keys:
            [graph] = attach_data_summary([graph], args.synergy_file, key)

        # write out gml (to run downstream with annotate)
        gml_file = "{}.grammar.gml".format(out_prefix) # TODO have a better name!
        nx.write_gml(stringize_nx_graph(graph), gml_file)

    
    
    return None


if __name__ == "__main__":
    main()


