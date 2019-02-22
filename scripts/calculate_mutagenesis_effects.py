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


def parse_calculation_strings(args):
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
    # TODO consider saving out under new keys each time
    # here - use a tag? "{}.1".format(DataKeys.SYNERGY_SCORES)
    # then plot for whichever one you're on
    # don't write out if already done - ie compare label string
    start_idx = 0
    out_key = "{}.{}".format(DataKeys.SYNERGY_SCORES, start_idx)
    while True:
        with h5py.File(args.synergy_file, "a") as hf:
            if hf.get(out_key) is not None:
                # check if matches, you can update that one
                if hf[out_key].attrs[AttrKeys.PLOT_LABELS] == labels:
                    hf[out_key] = results
                start_idx += 1
                out_key = "{}.{}".format(DataKeys.SYNERGY_SCORES, start_idx)
                continue
            else:
                hf.create_dataset(DataKeys.SYNERGY_SCORES, data=results)
                hf[DataKeys.SYNERGY_SCORES].attrs[AttrKeys.PLOT_LABELS] = labels
    
    # refine:
    if args.refine:
        assert len(args.calculations) == 2
        stdev_thresh = 2.0
        
        # while here, calculate index diff (pwm position diff)
        with h5py.File(args.synergy_file, "a") as hf:
            indices = hf[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT][:]
            distances = indices[:,3,0] - indices[:,3,1] # {N}
            if hf.get(DataKeys.SYNERGY_DIST) is not None:
                del hf[DataKeys.SYNERGY_DIST]
            hf.create_dataset(DataKeys.SYNERGY_DIST, data=distances)

        # get diffs and save out
        synergy_diffs = results[:,0] - results[:,1] # {N, logit}
        with h5py.File(args.synergy_file, "a") as hf:
            if hf.get(DataKeys.SYNERGY_DIFF) is not None:
                del hf[DataKeys.SYNERGY_DIFF]
            hf.create_dataset(DataKeys.SYNERGY_DIFF, data=synergy_diffs)
        
        # remove outliers first to stabilize gaussian calc
        outlier_thresholds = np.percentile(
            np.abs(synergy_diffs), 75, axis=0, keepdims=True)
        synergy_diffs_filt = np.where(
            np.greater(np.abs(synergy_diffs), outlier_thresholds),
            0, synergy_diffs)
        synergy_diffs_filt = np.where(
            np.isclose(synergy_diffs_filt, 0),
            np.nan, synergy_diffs_filt)

        # get stds
        stdevs = np.nanstd(synergy_diffs_filt, axis=0, keepdims=True)
        means = np.nanmean(synergy_diffs_filt, axis=0, keepdims=True)
        
        # differential
        #differential = np.abs(synergy_diffs) > stdev_thresh * stdevs
        differential = synergy_diffs > stdev_thresh * stdevs
        differential[np.abs(distances) < 12] = 0
        
        # add in new dataset that marks differential
        with h5py.File(args.synergy_file, "a") as hf:
            if hf.get(DataKeys.SYNERGY_DIFF_SIG) is not None:
                del hf[DataKeys.SYNERGY_DIFF_SIG]
            hf.create_dataset(DataKeys.SYNERGY_DIFF_SIG, data=differential)
            hf[DataKeys.SYNERGY_DIFF_SIG].attrs[AttrKeys.PLOT_LABELS] = labels

        # try analyzing the distance
        diff_indices = np.greater_equal(np.sum(differential!=0, axis=1), 2)
        diff_indices = np.where(diff_indices)[0]
        diff_distances = distances[diff_indices] # {N}
        max_dist = np.percentile(diff_distances, 99)
        
        with h5py.File(args.synergy_file, "a") as hf:
            if hf.get(DataKeys.SYNERGY_MAX_DIST) is not None:
                del hf[DataKeys.SYNERGY_MAX_DIST]
            hf.create_dataset(DataKeys.SYNERGY_MAX_DIST, data=max_dist)
        
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
        other_keys = [
            "logits.norm",
            "ATAC_SIGNALS.NORM"]
        for key in other_keys:
            [graph] = attach_data_summary([graph], args.synergy_file, key)

        # write out gml (to run downstream with annotate)
        gml_file = "{}.grammar.gml".format(out_prefix) # TODO have a better name!
        nx.write_gml(stringize_nx_graph(graph), gml_file)
        
    # and plot:
    # 1) comparison between the two FCs
    # 2) plot with distance and other things
    plot_cmd = "plot-h5.synergy_results.2.R {} {} {} {} {} {} {}".format(
        args.synergy_file,
        DataKeys.SYNERGY_SCORES,
        DataKeys.SYNERGY_DIFF,
        DataKeys.SYNERGY_DIFF_SIG,
        DataKeys.SYNERGY_DIST,
        DataKeys.SYNERGY_MAX_DIST,
        out_prefix)
    print plot_cmd
    os.system(plot_cmd)
    
    
    return None


if __name__ == "__main__":
    main()


