#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import re
import sys
import glob
import h5py
import logging
import argparse

import numpy as np
import pandas as pd
import networkx as nx

from numpy.random import RandomState
from scipy.stats import zscore
from tronn.interpretation.combinatorial import setup_combinations
from tronn.util.scripts import setup_run_logs
from tronn.util.utils import DataKeys


class MPRA_PARAMS(object):
    """all mpra design params (Margaret)
    """
    LEN_FILLER = 20
    LEN_BARCODE = 20
    MAX_OLIGO_LENGTH = 230
    FWD_PCR_PRIMER = 'ACTGGCCGCTTCACTG'
    REV_PCR_PRIMER = 'AGATCGGAAGAGCGTCG'
    RS_ECORI = 'GAATTC' # 5'-3'
    RS_BAMHI = 'GGATCC'
    RS_XHOI = 'CTCGAG'
    RS_XBAI = 'TCTAGA'
    RS_NCOI = 'CCATGG'
    RS_XBAI_dam1 = 'GATCTAGA'
    RS_XBAI_dam2 = 'TCTAGATC'
    LETTERS= ['A', 'C', 'G', 'T']
    MAX_FRAG_LEN = MAX_OLIGO_LENGTH - (
        len(FWD_PCR_PRIMER) + len(RS_XHOI) + LEN_FILLER + len(RS_XBAI) + LEN_BARCODE + len(REV_PCR_PRIMER))


GGR_LEFT_CLIP = 420
    

def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # required args
    parser.add_argument(
        "--grammar_summary",
        help="grammar summary file")
    parser.add_argument(
        "--synergy_dirs", nargs="+",
        help="folders where synergy dirs reside")
    parser.add_argument(
        "--barcodes",
        help="file of barcode sequences")
    
    # out
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")

    # parse args
    args = parser.parse_args()

    return args


def is_rs_clean(sequence):
    """check for cut sites, should have NONE
    """
    # cut sites (insert to backbone) should NOT exist
    if sequence.count(MPRA_PARAMS.RS_ECORI) != 0:
        return False
    if sequence.count(MPRA_PARAMS.RS_BAMHI) != 0:
        return False

    # cut sites (insert promoter and luc) should NOT exist
    if sequence.count(MPRA_PARAMS.RS_XHOI) != 0:
        return False
    if sequence.count(MPRA_PARAMS.RS_XBAI) != 0:
        return False
    
    return True


def is_fragment_compatible(sequence):
    """check fragment for compatibility
    """
    # check for NO cut sites
    if not is_rs_clean(sequence):
        return False

    # check again when attaching FWD primer
    if not is_rs_clean(MPRA_PARAMS.FWD_PCR_PRIMER + sequence):
        return False

    # check after attaching XHOI
    fragment_extended = sequence + MPRA_PARAMS.RS_XHOI
    if fragment_extended.count(MPRA_PARAMS.RS_ECORI) != 0:
        #logging.info("ecori")
        return False
    if fragment_extended.count(MPRA_PARAMS.RS_BAMHI) != 0:
        #logging.info("bamhi")
        return False
    
    # check for N
    if sequence.count("N") != 0:
        #logging.info("n")
        return False
    
    return True
    

def is_barcode_compatible(barcode):
    """check barcode for compatibility
    """
    # check when attaching REV primer
    if not is_rs_clean(barcode + MPRA_PARAMS.REV_PCR_PRIMER):
        return False

    # check when attaching XBAI site
    barcode_extended = MPRA_PARAMS.RS_XBAI + barcode
    if barcode_extended.count(MPRA_PARAMS.RS_XBAI_dam2) != 0:
        return False
    
    # check overlaps
    if barcode_extended.count(MPRA_PARAMS.RS_ECORI) != 0:
        return False
    if barcode_extended.count(MPRA_PARAMS.RS_BAMHI) != 0:
        return False
    
    return True


def is_filler_compatible(filler):
    """check if filler is compatible with library design
    """
    # first check cut sites
    if not is_rs_clean(filler):
        return False
    
    # check overlaps
    filler_extended = MPRA_PARAMS.RS_XHOI + filler + MPRA_PARAMS.RS_XBAI
    if filler_extended.count(MPRA_PARAMS.RS_ECORI) != 0:
        return False
    if filler_extended.count(MPRA_PARAMS.RS_BAMHI) != 0:
        return False
    
    # check methylation
    if filler_extended.count(MPRA_PARAMS.RS_XBAI_dam1) != 0:
        return False
        
    return True


def generate_compatible_filler(rand_seed):
    """generate filler sequence and check methylation
    """
    while True:
        rand_state = RandomState(rand_seed)
        # generate random sequence
        random_seq = rand_state.choice(
            MPRA_PARAMS.LETTERS, size=MPRA_PARAMS.LEN_FILLER)
        random_seq = "".join(random_seq)
        
        # if passes checks then break
        if is_filler_compatible(random_seq):
            break

        # otherwise change the seed and keep going
        rand_seed += 1

    # and move up one more
    rand_seed += 1
        
    return random_seq, rand_seed


def is_sequence_mpra_ready(sequence):
    """given a sequence, check that it's compatible 
    with current library generation strategy
    """
    # cut sites (insert to backbone) should NOT exist
    if sequence.count(MPRA_PARAMS.RS_ECORI) != 0:
        logging.info("ecori")
        return False
    if sequence.count(MPRA_PARAMS.RS_BAMHI) != 0:
        logging.info("bamhi")
        return False

    # cut sites (insert promoter and luc) SHOULD exist ONCE
    if sequence.count(MPRA_PARAMS.RS_XHOI) != 1:
        logging.info("xhoi")
        return False
    if sequence.count(MPRA_PARAMS.RS_XBAI) != 1:
        logging.info("xbai")
        return False

    # check length
    if len(sequence) > MPRA_PARAMS.MAX_OLIGO_LENGTH:
        logging.info("length")
        return False

    # check for N
    if sequence.count("N") != 0:
        logging.info("found N")
        return False

    return True


def trim_sequence_for_mpra(sequence, edge_indices):
    """trim sequence to fit in library
    """
    motif_padding_len = 10
    frag_minimal_len = edge_indices[1] - edge_indices[0] + (motif_padding_len * 2)
    extra_padding = int((MPRA_PARAMS.MAX_FRAG_LEN - frag_minimal_len) / 2.)
    start_position = edge_indices[0] - motif_padding_len - extra_padding
    end_position = start_position + MPRA_PARAMS.MAX_FRAG_LEN
    if start_position < 0:
        # if closer to start, trim off the back only
        sequence = sequence[:MPRA_PARAMS.MAX_FRAG_LEN]
    elif end_position > len(sequence):
        # if closer to back, trim off front only
        sequence = sequence[-MPRA_PARAMS.MAX_FRAG_LEN:]
    else:
        # in the middle, trim equally
        sequence = sequence[start_position:]
        sequence = sequence[:MPRA_PARAMS.MAX_FRAG_LEN]

    # check
    assert len(sequence) == MPRA_PARAMS.MAX_FRAG_LEN, edge_indices
        
    return sequence


def build_mpra_sequence(sequence, barcode, rand_seed, log):
    """attach on relevant sequence info
    """
    if not is_fragment_compatible(sequence):
        import ipdb
        ipdb.set_trace()
    
    assert is_fragment_compatible(sequence)
    assert is_barcode_compatible(barcode)
    
    # attach FWD primer to front
    sequence = MPRA_PARAMS.FWD_PCR_PRIMER + sequence
    # attach XHOI
    sequence += MPRA_PARAMS.RS_XHOI
    # attach filler (random 20)
    filler, rand_seed = generate_compatible_filler(rand_seed)
    sequence += filler
    # attach XBAI
    sequence += MPRA_PARAMS.RS_XBAI
    # attach barcode
    sequence += barcode
    # attach reverse primer
    sequence += MPRA_PARAMS.REV_PCR_PRIMER

    #print rand_seed, barcode, filler
    if not is_sequence_mpra_ready(sequence):
        import ipdb
        ipdb.set_trace()
    
    # sanity check
    assert is_sequence_mpra_ready(sequence), log
    
    return sequence, rand_seed


def seq_list_compatible(seq_list, left_clip=420, right_clip=580):
    """helper function when checking sequences from h5 file
    """
    for seq in seq_list:
        if not is_fragment_compatible(seq[420:580]):
            return False
        
    return True


def save_sequences(h5_file, indices, left_clip=420, right_clip=580, prefix="id"):
    """save sequences to a df
    """
    keys = [
        "{}.0".format(DataKeys.SYNERGY_DIFF_SIG),
        "{}.0".format(DataKeys.SYNERGY_DIST),
        DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT,
        DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT,
        DataKeys.MUT_MOTIF_LOGITS,
        DataKeys.SEQ_METADATA,
        "ATAC_SIGNALS.NORM",
        "H3K27ac_SIGNALS.NORM",
        DataKeys.LOGITS_NORM,
        DataKeys.GC_CONTENT,
        "{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)]

    # first get reshape params
    num_sequences = indices.shape[0]
    with h5py.File(h5_file, "r") as hf:
        num_combos = hf[DataKeys.MUT_MOTIF_ORIG_SEQ].shape[1]
        num_muts = int(np.log2(num_combos))

    # go through keys and extract relevant results
    with h5py.File(h5_file, "r") as hf:
        for key_idx in range(len(keys)):
            key = keys[key_idx]
            key_results = hf[key][:][indices]

            # check dims and flatten accordingly
            if len(key_results.shape) <= 2:
                end_dims = list(key_results.shape)[1:]
                key_results = np.stack([key_results]*num_combos, axis=1)
            elif len(key_results.shape) == 3:
                end_dims = list(key_results.shape)[2:]
            else:
                raise ValueError, "key does not match shape, is {}".format(key_results.shape)
            key_results = np.reshape(key_results, [-1]+end_dims)

            # if bool convert to int
            if key_results.dtype == np.bool:
                key_results = key_results.astype(int)

            # other adjustments
            if key == DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT:
                key_results = key_results.astype(int)
            if key == DataKeys.SYNERGY_DIST:
                key_results = key_results.astype(int)
                
            # create dataframe
            key_df = pd.DataFrame(key_results)
            if key_df.shape[1] > 1:
                key_df = pd.DataFrame({key:key_df.astype(str).apply(",".join, axis=1)})
            else:
                key_df.columns = [key]
                
            # append
            if key_idx == 0:
                all_results = key_df
            else:
                all_results[key] = key_df[key]
                
        # also attach the max indices of motifs to each example (for downstream trimming)
        mut_positions = hf[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT][:][indices][:,-1]
        max_pos = np.amax(mut_positions, axis=1)
        max_pos = np.stack([max_pos]*num_combos, axis=1)
        max_pos = np.reshape(max_pos, [-1])
        min_pos = np.amin(mut_positions, axis=1)
        min_pos = np.stack([min_pos]*num_combos, axis=1)
        min_pos = np.reshape(min_pos, [-1])
        all_results["edge_indices"] = pd.DataFrame(
            np.stack([min_pos, max_pos], axis=1) - GGR_LEFT_CLIP).astype(str).apply(",".join, axis=1)
                
    # adjust sequence
    all_results.insert(
        0, "sequence.nn",
        all_results["{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)].str[left_clip:right_clip])
    all_results = all_results.drop(columns=["{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)])
    
    # set up combos
    combinations = setup_combinations(num_muts)
    combinations = np.swapaxes(combinations, 0, 1)
    combinations = np.stack([combinations]*num_sequences, axis=0)
    combinations = np.reshape(combinations, [-1, num_muts])
    combo_df = pd.DataFrame(combinations).astype(int)
    combo_df["combos"] = combo_df.astype(str).apply(",".join, axis=1)
    all_results.insert(0, "combos", combo_df["combos"])

    # set up indices
    rep_indices = np.stack([indices]*num_combos, axis=1)
    rep_indices = np.reshape(rep_indices, [-1] )
    all_results.insert(0, "example_fileidx", rep_indices)

    # set up a group id
    all_results.insert(
        0, "example_id",
        prefix + "." + all_results["example_fileidx"].astype(str))
    
    # set up a unique id
    all_results.insert(
        0, "unique_id",
        prefix + "." + all_results["example_fileidx"].astype(str) + "." + all_results["combos"].str.replace(",", ""))

    return all_results


def extract_sequences(args):
    """sample sequences
    """
    diff_sample_num = 8
    nondiff_proximal_sample_num = 4
    nondiff_distal_sample_num = 4
    
    grammar_summary = pd.read_table(args.grammar_summary, index_col=0)

    for grammar_idx in range(grammar_summary.shape[0]):
        # set seed each time you run a new grammar
        np_seed = np.random.seed(1337)
        grammar_file = grammar_summary.iloc[grammar_idx]["filename"]
        
        # get synergy file
        grammar_prefix = os.path.basename(grammar_file).split(".gml")[0]
        synergy_files = []
        for synergy_dir in args.synergy_dirs:
            synergy_file = "{}/{}/ggr.synergy.h5".format(synergy_dir, grammar_prefix)
            if os.path.isfile(synergy_file):
                synergy_files.append(synergy_file)
        assert len(synergy_files) == 1
        synergy_file = synergy_files[0]
        synergy_pwm_names_file = "{}/ggr.synergy.pwms.order.txt".format(
            os.path.dirname(synergy_file))
        assert os.path.isfile(synergy_pwm_names_file)
        #print grammar_idx, synergy_file
        
        # open synergy file to subsample sequences
        try:
            with h5py.File(synergy_file, "r") as hf:
                # extract
                sequences = hf["{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)][:] # {N, combos, 1}
                distances = hf["{}.0".format(DataKeys.SYNERGY_DIST)][:] # {N}
                diffs_sig = hf["{}.0".format(DataKeys.SYNERGY_DIFF_SIG)][:] # {N, logit}
                max_dist = hf["{}.0".format(DataKeys.SYNERGY_MAX_DIST)][()]
        except IOError:
            continue
            
        # get diff
        min_diff = diffs_sig.shape[1]
        num_tasks_diff = np.sum(diffs_sig, axis=1)
        while True:
            diffs_sig = np.greater_equal(num_tasks_diff, min_diff) # {N}
            diff_indices = np.where(diffs_sig)[0]
            diff_indices = np.array([
                idx for idx in diff_indices
                if seq_list_compatible(sequences[idx].squeeze().tolist())])
            try:
                diff_sample_indices = diff_indices[
                    np.random.choice(diff_indices.shape[0], diff_sample_num, replace=False)]
                break
            except ValueError:
                min_diff -= 1

            if min_diff == 0:
                diff_sample_indices = diff_indices

        if diff_sample_indices.shape[0] == 0:
            # if no diff sequences, continue
            logging.info("skipping because no differential seqs that are compatible")
            continue
        
        diff_bool = np.zeros(distances.shape)
        diff_bool[diff_sample_indices] = 1
        
        # get nondiff, less than dist
        # TODO give a min dist (to prevent choosing ones that overlap)
        nondiff = np.logical_not(diffs_sig) # {N}
        nondiff_proximal_indices = np.where(np.logical_and(nondiff, distances < max_dist))[0]
        nondiff_proximal_indices = np.array([
            idx for idx in nondiff_proximal_indices
            if seq_list_compatible(sequences[idx].squeeze().tolist())])
        try:
            nondiff_proximal_sample_indices = nondiff_proximal_indices[
                np.random.choice(nondiff_proximal_indices.shape[0], nondiff_proximal_sample_num, replace=False)]
        except ValueError:
            nondiff_proximal_sample_indices = nondiff_proximal_indices
        if nondiff_proximal_sample_indices.shape[0] == 0:
            # no nondiff sequences
            logging.info("skipping because no control seqs that are compatible")
            continue
        nondiff_proximal_bool = np.zeros(distances.shape)
        nondiff_proximal_bool[nondiff_proximal_sample_indices] = 1
        
        # get nondiff, greater than dist
        nondiff_distal_indices = np.where(np.logical_and(nondiff, distances >= max_dist))[0]
        nondiff_distal_indices = np.array([
            idx for idx in nondiff_distal_indices
            if seq_list_compatible(sequences[idx].squeeze().tolist())])
        try:
            nondiff_distal_sample_indices = nondiff_distal_indices[
                np.random.choice(nondiff_distal_indices.shape[0], nondiff_distal_sample_num, replace=False)]
        except ValueError:
            nondiff_distal_sample_indices = nondiff_distal_indices
        if nondiff_distal_sample_indices.shape[0] == 0:
            logging.info("skipping because no control seqs that are compatible")
            # no nondiff sequences
            continue
        nondiff_distal_bool = np.zeros(distances.shape)
        nondiff_distal_bool[nondiff_distal_sample_indices] = 1

        # and mark out the ones chosen in the synergy file
        all_sample_indices = np.concatenate(
            [diff_sample_indices,
             nondiff_proximal_sample_indices,
             nondiff_distal_sample_indices], axis=0)
        all_bool = np.zeros(distances.shape)
        all_bool[all_sample_indices] = 1
        if np.sum(all_bool) < 10:
            logging.info("skipping because not enough sequences of interest")
            continue
        
        mpra_sample_df = save_sequences(synergy_file, all_sample_indices, prefix=grammar_prefix)
        
        if False:
            # and can plot out here
            out_prefix = "{}/{}".format(args.out_dir, synergy_dir)
            plot_cmd = "plot-h5.synergy_results.2.R {} {} {} {} {} {} {} {}".format(
                synergy_file,
                DataKeys.SYNERGY_SCORES,
                DataKeys.SYNERGY_DIFF,
                DataKeys.SYNERGY_DIFF_SIG,
                DataKeys.SYNERGY_DIST,
                DataKeys.SYNERGY_MAX_DIST,
                out_prefix,
                "mpra.sample.all")
            print plot_cmd
            os.system(plot_cmd)

        # concatenate
        if grammar_idx == 0:
            mpra_all_df = mpra_sample_df
        else:
            mpra_all_df = pd.concat([mpra_all_df, mpra_sample_df])
            
    # adjust for MPRA
    # get barcodes and shuffle
    rand_state = RandomState(42)
    barcodes = pd.read_table(args.barcodes, header=None).iloc[:,0].values
    rand_state.shuffle(barcodes)

    # adjust sequences
    mpra_sequences = []
    barcode_idx = 0
    rand_seed = 0
    logging.info("adjusting {} sequences".format(mpra_all_df.shape[0]))
    incompatible = []
    for seq_idx in range(mpra_all_df.shape[0]):
        example_id = mpra_all_df.iloc[seq_idx]["example_id"]
        if seq_idx % 1000 == 0:
            print seq_idx
        sequence = mpra_all_df.iloc[seq_idx]["sequence.nn"]
        edge_indices = [
            int(float(val)) for val in mpra_all_df.iloc[seq_idx]["edge_indices"].split(",")]
        while True:
            barcode = barcodes[barcode_idx]
            if is_barcode_compatible(barcode):
                break
            barcode_idx += 1
        barcode_idx += 1
        try:
            sequence = trim_sequence_for_mpra(sequence, edge_indices)
        except AssertionError:
            incompatible.append(example_id)
            
        # after trimming, you reintroduce opportunities for the fragment to be wrong
        if not is_fragment_compatible(sequence):
            incompatible.append(example_id)
            mpra_sequences.append("NNNNN")
            continue
        
        mpra_sequence, rand_seed = build_mpra_sequence(sequence, barcode, rand_seed, seq_idx)
        mpra_sequences.append(mpra_sequence)

    mpra_all_df.insert(
        2, "sequence.mpra", mpra_sequences)
    print len(incompatible)

    # drop incompatible
    print mpra_all_df.shape
    mpra_all_df = mpra_all_df[~mpra_all_df["example_id"].isin(incompatible)]
    print mpra_all_df.shape

    if True:
        # look at duplicates
        print len(set(mpra_all_df["example_id"].values.tolist()))
        print len(mpra_all_df["example_metadata"].values.tolist())
        print len(set(mpra_all_df["example_metadata"].values.tolist()))
        print len(mpra_all_df["sequence.mpra"].values.tolist())
        print len(set(mpra_all_df["sequence.mpra"].values.tolist()))
    
    # finally save out
    mpra_all_df.to_csv("{}/mpra.seqs.txt".format(args.out_dir), sep="\t")
    
    return None


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])
    
    # make sure reproducible
    extract_sequences(args)
    
    return None


if __name__ == "__main__":
    main()
