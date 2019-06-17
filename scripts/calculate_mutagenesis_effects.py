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
        help="calculations to perform, in the format {FOREGROUND}/{BACKGROUND} - ex 110/100")
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


def main():
    """run calculations
    """
    # set up args
    args = parse_args()
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])
    out_prefix = "{}/{}".format(args.out_dir, args.prefix)

    min_pwm_dist = 12 # ignore PWMs that overlap (likely same importance scores contributing)
    stdev_thresh = 2.0
        
    # now set up the indices
    parse_calculation_strings(args)

    # read in data
    with h5py.File(args.synergy_file, "r") as hf:
        outputs = hf[DataKeys.MUT_MOTIF_LOGITS][:] # {N, mutM_combos, logit}
        sig_pwms_names = hf[DataKeys.MUT_MOTIF_LOGITS].attrs[AttrKeys.PWM_NAMES]
        
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
    
    # separate calculation which is correct synergy test (with 2 motifs)
    # expected = (01 - 00) + (10 - 00)
    # actual = 11 - 00
    # just calculate expected, and maintain vectors of info {N, tasks}
    # then for each, you can compare {N} for actual vs {N} expected.
    # this is paired, don't assume normal distr, so use wilcoxon rank sum to get a value
    for i in range(len(args.calculations)):

        # 11
        one_one_combo = args.calculations[i]
        one_one_idx = np.where(
            (np.transpose(combinations) == one_one_combo).all(axis=1))[0][0]
        one_one_vals = outputs[:,one_one_idx]
        
        # pull indices
        indices = np.where(args.calculations[i]==1)[0]
        assert len(indices) == 2
        
        # 01
        zero_one_combo = np.array(one_one_combo)
        zero_one_combo[indices[0]] = 0
        zero_one_idx = np.where(
            (np.transpose(combinations) == zero_one_combo).all(axis=1))[0][0]
        zero_one_vals = outputs[:,zero_one_idx]

        # 10
        one_zero_combo = np.array(one_one_combo)
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
        
        # compare the two
        diff = actual - expected
        pvals = np.apply_along_axis(wilcoxon, 0, diff)[1]
        print sig_pwms_names, (pvals < 0.05).astype(int)
        print np.mean(diff[0:13], axis=0)
        
        # save out: actual, expected, pvals
        
    quit()
    
    
    
    # go through calculations
    results = np.zeros((outputs.shape[0], len(args.calculations), outputs.shape[2]))
    labels = []
    for i in xrange(len(args.calculations)):

        # extract foreground idx
        foreground = args.calculations[i][0]
        foreground_idx = np.where(
            (np.transpose(combinations) == foreground).all(axis=1))[0][0]

        # for logging
        foreground_strings = _setup_output_string(sig_pwms_names, foreground)
            
        # extract background idx
        background = args.calculations[i][1]
        background_idx = np.where(
            (np.transpose(combinations) == background).all(axis=1))[0][0]

        # for logging
        background_strings = _setup_output_string(sig_pwms_names, background)

        # and adjust to conditional string
        conditional_string = _make_conditional_string(
            foreground_strings, background_strings)
        labels.append(conditional_string)
        
        # log scale, so subtract
        results[:,i] = outputs[:,foreground_idx] - outputs[:,background_idx]

    # calculate sig for all pairs
    for i in xrange(results.shape[1]):
        for j in xrange(results.shape[1]):
            if i >= j:
                continue
            
            # calculate sig
            delta_results = results[:,i] - results[:,j]
            pvals = run_delta_permutation_test(delta_results)
            # TODO this is currently unused!
            
    # save out into h5 file
    run_idx = 0
    score_key = "{}.{}".format(DataKeys.SYNERGY_SCORES, run_idx)
    while True:
        print score_key
        with h5py.File(args.synergy_file, "a") as hf:
            if hf.get(score_key) is not None:
                # check if matches, you can update that one
                if list(hf[score_key].attrs[AttrKeys.PLOT_LABELS]) == labels:
                    hf[score_key][:] = results
                    break
                run_idx += 1
                score_key = "{}.{}".format(DataKeys.SYNERGY_SCORES, run_idx)
                continue
            else:
                hf.create_dataset(score_key, data=results)
                hf[score_key].attrs[AttrKeys.PLOT_LABELS] = labels
                break
        
    # calculate max index diff (pwm position diff) <- works for all numbers of motifs
    dist_key = "{}.{}".format(DataKeys.SYNERGY_DIST, run_idx)
    with h5py.File(args.synergy_file, "a") as hf:
        indices = hf[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT][:]
        distances = np.max(indices[:,-1], axis=-1) - np.min(indices[:,-1], axis=-1) # lose orientation but ok for now
        if hf.get(dist_key) is not None:
            del hf[dist_key]
        hf.create_dataset(dist_key, data=distances)

    # other calcs which only work with 2 calculations:
    if len(args.calculations) == 2:

        # get diff betwen the two calcs
        synergy_diffs = results[:,0] - results[:,1] # {N, logit}
        synergy_diffs_key = "{}.{}".format(DataKeys.SYNERGY_DIFF, run_idx)
        with h5py.File(args.synergy_file, "a") as hf:
            if hf.get(synergy_diffs_key) is not None:
                del hf[synergy_diffs_key]
            hf.create_dataset(synergy_diffs_key, data=synergy_diffs)

        # run a quick differential test (outlier stabilized gaussian)
        outlier_thresholds = np.percentile(
            np.abs(synergy_diffs), 75, axis=0, keepdims=True)
        synergy_diffs_filt = np.where(
            np.greater(np.abs(synergy_diffs), outlier_thresholds),
            0, synergy_diffs)
        synergy_diffs_filt = np.where(
            np.isclose(synergy_diffs_filt, 0),
            np.nan, synergy_diffs_filt)
        stdevs = np.nanstd(synergy_diffs_filt, axis=0, keepdims=True)
        means = np.nanmean(synergy_diffs_filt, axis=0, keepdims=True)
        
        # differential cutoff
        differential = synergy_diffs > stdev_thresh * stdevs
        differential[np.abs(distances) < min_pwm_dist] = 0
        
        # add in new dataset that marks differential
        synergy_sig_key = "{}.{}".format(DataKeys.SYNERGY_DIFF_SIG, run_idx)
        with h5py.File(args.synergy_file, "a") as hf:
            if hf.get(synergy_sig_key) is not None:
                del hf[synergy_sig_key]
            hf.create_dataset(synergy_sig_key, data=differential)
            hf[synergy_sig_key].attrs[AttrKeys.PLOT_LABELS] = labels

        # analyze max distance of synergistic interaction
        diff_indices = np.greater_equal(np.sum(differential!=0, axis=1), 1)
        diff_indices = np.where(diff_indices)[0]
        diff_distances = distances[diff_indices] # {N}
        max_dist = np.percentile(diff_distances, 95)
        max_dist_key = "{}.{}".format(DataKeys.SYNERGY_MAX_DIST, run_idx)
        with h5py.File(args.synergy_file, "a") as hf:
            if hf.get(max_dist_key) is not None:
                del hf[max_dist_key]
            hf.create_dataset(max_dist_key, data=max_dist)
            #hf.create_dataset(max_dist_key, data=0.)

        # analyze pwm score strength
        pwm_strength_key = "pwms.strength.{}".format(run_idx)
        with h5py.File(args.synergy_file, "a") as hf:
            if hf.get(pwm_strength_key) is not None:
                del hf[pwm_strength_key]
            pwm_scores = np.amin(hf[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT][:,-1], axis=1)
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
    if args.refine:
        assert len(args.calculations) == 2
        
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


