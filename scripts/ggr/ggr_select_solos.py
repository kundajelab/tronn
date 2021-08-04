#!/usr/bin/env python

import os
import sys
import h5py
import argparse

import numpy as np
import pandas as pd

from tronn.util.h5_utils import AttrKeys
from tronn.util.mpra import seq_list_compatible
from tronn.util.scripts import setup_run_logs
from tronn.util.utils import DataKeys


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="select best single sequences to test")

    # inputs
    parser.add_argument(
        "--grammar_summaries", nargs="+",
        help="grammar summary files (one per run), format as key=file")
    parser.add_argument(
        "--synergy_dirs", nargs="+",
        help="folders where synergy dirs reside")
    parser.add_argument(
        "--sim_dir",
        help="folder where simulations reside")
    parser.add_argument(
        "--mpra",
        help="mpra file of selected sequences")
    parser.add_argument(
        "--variants", nargs="+", default=[],
        help="variants to consider (bed files)")
    parser.add_argument(
        "--plot", action="store_true",
        help="make plots of samples")
    
    # out
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")

    # parse args
    args = parser.parse_args()

    return args


def _get_synergy_file(synergy_dirs, grammar_prefix):
    """get synergy file
    """
    synergy_files = []
    for synergy_dir in synergy_dirs:
        synergy_file = "{}/{}/ggr.synergy.h5".format(synergy_dir, grammar_prefix)
        if os.path.isfile(synergy_file):
            synergy_files.append(synergy_file)
    assert len(synergy_files) == 1
    synergy_file = synergy_files[0]

    # check synergy file is complete and readable
    with h5py.File(synergy_file, "r") as hf:
        sig_pwm_names = hf[DataKeys.MUT_MOTIF_LOGITS].attrs[AttrKeys.PWM_NAMES]

    return synergy_file


def build_consensus_file_sets(grammar_summaries, synergy_dirs):
    """take the grammar sets and build consensus sets of synergy files
    """
    # merge the summaries, require that the summaries match EXACTLY
    for summary_idx in range(len(grammar_summaries)):
        run, summary_file = grammar_summaries[summary_idx]
        summary_df = pd.read_table(summary_file)
        summary_df = summary_df[["nodes", "filename"]]
        summary_df.columns = ["nodes", run]
        summary_df = summary_df.set_index("nodes")
        if summary_idx == 0:
            all_summaries = summary_df
            num_grammars = summary_df.shape[0]
        else:
            all_summaries = all_summaries.merge(summary_df, left_index=True, right_index=True)
            assert num_grammars == summary_df.shape[0]
            
    # convert the grammar files to the synergy files
    for run, _ in grammar_summaries:
        for grammar_idx in range(all_summaries.shape[0]):
            df_index = all_summaries.index[grammar_idx]
            grammar_file = all_summaries.iloc[grammar_idx][run]
            grammar_prefix = os.path.basename(grammar_file).split(".gml")[0]
            run_dirs = [synergy_dir for synergy_dir in synergy_dirs
                        if run in synergy_dir]
            synergy_file = _get_synergy_file(run_dirs, grammar_prefix)
            all_summaries.loc[df_index,run] = synergy_file
            
    # make into list
    all_summaries["combined"] = all_summaries.values.tolist()
            
    return all_summaries


def _get_compatible_sequence_indices(sequences):
    """get indices of MPRA compatible sequences
    """
    compatible = []
    for seq_idx in range(sequences.shape[0]):
        if seq_list_compatible(sequences[seq_idx].squeeze().tolist()):
            compatible.append(seq_idx)

    return compatible


def get_consensus_sequences_across_runs(synergy_files):
    """compare synergy files and get examples (by metadata) that
    can be used in an MPRA library (consistent with design params)
    """
    for file_idx in range(len(synergy_files)):
        with h5py.File(synergy_files[file_idx]) as hf:
            sequences = hf["{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)][:] # {N, combos, 1}
            examples = hf[DataKeys.SEQ_METADATA][:,0]

            # TODO only keep those that are sig synergy
            #print examples.shape
            synergy_sig = np.any(
                hf["{}.0".format(DataKeys.SYNERGY_DIFF_SIG)][:] != 0,
                axis=1)
            examples = examples[synergy_sig]
            sequences = sequences[synergy_sig]
            #print examples.shape
            
        # only keep compatible
        compatible_indices = _get_compatible_sequence_indices(sequences)
        examples = set(examples[compatible_indices].tolist())
        if file_idx == 0:
            consensus = examples
        else:
            consensus = consensus.intersection(examples)

    return sorted(list(consensus))



def get_consensus_trimmed_bed(examples_df, synergy_files, out_bed_file):
    """trim using edges of grammar
    """
    examples = list(examples_df["example_metadata"].values)
    for file_idx in range(len(synergy_files)):
        synergy_file = synergy_files[file_idx]
        with h5py.File(synergy_file, "r") as hf:
            # get matched examples
            file_indices = np.where(np.isin(
                hf[DataKeys.SEQ_METADATA][:,0],
                examples))[0]
            assert file_indices.shape[0] == len(examples), "{}, {}".format(
                file_indices.shape[0], len(examples))
            file_metadata = hf[DataKeys.SEQ_METADATA][:,0][file_indices]
            sort_indices = np.argsort(file_metadata, axis=0)

            file_metadata = file_metadata[sort_indices]
            indices = hf[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT][:][file_indices][sort_indices]
            indices = indices[:,-1]

            # adjust positions
            tmp_start_positions = np.min(indices, axis=-1)
            tmp_end_positions = np.max(indices, axis=-1)
            if file_idx == 0:
                start_positions = tmp_start_positions
                end_positions = tmp_end_positions
            else:
                start_positions = np.maximum(start_positions, tmp_start_positions).astype(int)
                end_positions = np.minimum(end_positions, tmp_end_positions).astype(int)
                
    # and use these to set up a BED file
    examples_df_copy = examples_df.copy().sort_values("example_metadata")
    metadata = examples_df_copy["example_metadata"].str.split(";", n=3, expand=True)
    metadata = metadata[2].str.split("=", n=2, expand=True)
    metadata = metadata[1].str.split(":", n=2, expand=True)
    coords = metadata[1].str.split("-", n=2, expand=True)
    metadata[1] = coords[0].astype(int)
    
    # and adjust with coords
    metadata[2] = metadata[1] + end_positions
    metadata[1] = metadata[1] + start_positions
    metadata.index = examples_df_copy["example_metadata"]
    
    # check consistency
    metadata["diff"] = metadata[2] - metadata[1]
    metadata = metadata[metadata["diff"] > 0]
    metadata = metadata.drop("diff", axis=1)

    # print out bed
    metadata.to_csv(out_bed_file, sep="\t", compression="gzip", header=False, index=False)
    
    return metadata


def attach_variant_info(examples_df, synergy_files, variant_bed_files):
    """make a BED file, trim and check for variant presence
    """
    # get trimmed bed
    tmp_bed_file = "tmp.bed.gz"
    metadata = get_consensus_trimmed_bed(
        examples_df, synergy_files, tmp_bed_file)

    # overlap with variants
    tmp_overlap_bed_file = "tmp.overlap.bed.gz"
    for variant_file_idx in range(len(variant_bed_files)):
        variant_bed_file = variant_bed_files[variant_file_idx]
        intersect = "bedtools intersect -c -a {} -b {} | gzip -c > {}".format(
            tmp_bed_file, variant_bed_file, tmp_overlap_bed_file)
        os.system(intersect)

        # read back in and attach
        has_variant_df = pd.read_table(tmp_overlap_bed_file, header=None, index_col=False)
        has_variant_df[3] = has_variant_df[3].astype(int)
        
        if variant_file_idx == 0:
            metadata["num_variants"] = has_variant_df[3].values
        else:
            metadata["num_variants"] += has_variant_df[3].values

    # finally attach back into original df
    metadata = metadata[["num_variants"]]
    metadata["has_variant"] = (metadata["num_variants"] > 0).astype(int)
    metadata = metadata.reset_index()

    # merge
    examples_df = examples_df.merge(metadata, on="example_metadata")
    
    # clean up
    os.system("rm {} {}".format(tmp_bed_file, tmp_overlap_bed_file))
    
    return examples_df


def attach_data(
        examples_df,
        synergy_file,
        keys):
    """attach extra data
    """
    example_metadata = list(examples_df["example_metadata"].values)
    with h5py.File(synergy_file, "r") as hf:
        file_metadata = hf["example_metadata"][:][:,0]
        indices = np.where(np.isin(file_metadata, example_metadata))[0]
        
        # try to order according to examples file?
        # or actually just generate a new one and merge in
        file_metadata = hf["example_metadata"][:,0][indices]
        for key in keys:
            data = hf[key][:][indices]
            assert len(data.shape) <= 2
            max_data_val = np.max(data, axis=1)
            if data.shape[1] > 1:
                data = pd.DataFrame(data).astype(str).apply(",".join, axis=1).values
            data_df = pd.DataFrame({
                "example_metadata": file_metadata,
                key: data,
                "max.{}".format(key): max_data_val})

            # and merge in
            examples_df = examples_df.merge(data_df, on="example_metadata")

    return examples_df


def _get_sim_results(h5_file, results_key="simul.calcs/simul.scores.smooth.high"):
    """get simulation result to add in
    """
    try:
        with h5py.File(h5_file, "r") as hf:
            data = hf[results_key][:]
            best_logFC = np.max(data)
    except:
        best_logFC = 0.0
    
    return best_logFC



def main():
    """get an ordered list of most interesting sequences
    """
    # set up
    args = parse_args()
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    # set up grammar summaries -> get synergy files
    args.grammar_summaries = [tuple(val.split("=")) for val in args.grammar_summaries]
    summary_df = build_consensus_file_sets(args.grammar_summaries, args.synergy_dirs)
    
    # load mpra summary
    mpra_df = pd.read_table(args.mpra, index_col=0)
    mpra_regions = set(mpra_df["example_metadata"].values)
    
    # go through grammars
    all_solos = None
    for grammar_idx in range(summary_df.shape[0]):
        nodes = summary_df.index[grammar_idx]
        print nodes
        
        # get synergy files and pull consensus examples
        synergy_files = summary_df["combined"].iloc[grammar_idx]
        examples = get_consensus_sequences_across_runs(synergy_files)
        examples = pd.DataFrame({"example_metadata": examples})

        # attach simulation result - max logFC (across all tasks)
        grammar_prefix = synergy_files[0].split("/")[-2]
        sim_h5_file = "{}/{}/ggr.simulategrammar.h5".format(args.sim_dir, grammar_prefix)
        best_sim_logFC = _get_sim_results(sim_h5_file)
        examples["simulated_logFC_max"] = best_sim_logFC
        examples["nodes"] = nodes
        examples["synergy_file"] = synergy_files[0]

        if best_sim_logFC < 0.8:
            continue
        
        # attach whether in MPRA
        examples["in_mpra"] = examples["example_metadata"].isin(mpra_regions).astype(int)

        # attach whether matches variant
        examples = attach_variant_info(examples, synergy_files, args.variants)
        
        # keep if in MPRA OR has variant
        examples = examples[(examples["in_mpra"] > 0) | (examples["has_variant"] > 0)]
        
        # attach ATAC/H3K27ac/synergy
        attach_keys = [
            "ATAC_SIGNALS.NORM",
            "H3K27ac_SIGNALS.NORM",
            "{}.0".format(DataKeys.SYNERGY_DIFF)]
        examples = attach_data(examples, synergy_files[0], attach_keys)
        
        # sort by: variant, library, H3K27ac
        #sort_order = ["has_variant", "in_mpra", "max.ATAC_SIGNALS.NORM"]
        #sort_order = ["has_variant", "in_mpra", "max.H3K27ac_SIGNALS.NORM"]
        sort_order = ["in_mpra", "max.H3K27ac_SIGNALS.NORM"]
        examples = examples.sort_values(sort_order, ascending=False)

        # maybe some form of looser thresholding is better?
        
        # take top k
        sample_num = 10
        examples = examples.iloc[0:sample_num]
        #examples.to_csv("testing.txt", sep="\t")

        
        # merge in
        if all_solos is None:
            all_solos = examples.copy()
        else:
            all_solos = pd.concat([all_solos, examples], axis=0)

        #print all_solos.shape
        
    # sort on sim logFC
    sort_order = ["simulated_logFC_max", "max.H3K27ac_SIGNALS.NORM"]
    all_solos = all_solos.sort_values(sort_order, ascending=False)

    # and for each of these, plot importance scores
    file_indices = []
    for example_idx in range(all_solos.shape[0]):
        synergy_file = all_solos["synergy_file"].iloc[example_idx]
        example_metadata = all_solos["example_metadata"].iloc[example_idx]
        with h5py.File(synergy_file, "r") as hf:
            index = np.where(np.isin(
                hf["example_metadata"][:,0],
                [example_metadata]))[0][0]
        file_indices.append(index)
        plot_dir = "{}/viz/{}.idx-{}".format(
            args.out_dir,
            synergy_file.split("/")[-2],
            index)
        plot_cmd = "plot_importance_scores.py --data_file {} --indices {} -o {} --prefix ggr".format(
            synergy_file,
            index,
            plot_dir)
        print plot_cmd
        #os.system(plot_cmd)

    # write out
    all_solos["file_idx"] = file_indices
    all_solos.to_csv("{}/sorted_results.txt".format(args.out_dir), sep="\t")
    
    return


if __name__ == "__main__":
    main()
