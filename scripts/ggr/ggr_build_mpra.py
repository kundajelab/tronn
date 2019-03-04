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

from numpy.random import RandomState
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
GGR_RIGHT_CLIP = 580    

def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # inputs
    parser.add_argument(
        "--grammar_summaries", nargs="+",
        help="grammar summary files (one per run), format as key=file")
    parser.add_argument(
        "--synergy_dirs", nargs="+",
        help="folders where synergy dirs reside")
    parser.add_argument(
        "--barcodes",
        help="file of barcode sequences")
    parser.add_argument(
        "--barcodes_per_sequence", default=10, type=int,
        help="num barcodes to use for each sequence")
    parser.add_argument(
        "--sample_regions",
        help="BED file of regions from which to sample (constraint)")
    parser.add_argument(
        "--required_regions",
        help="BED file of required regions (if present)")
    
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
    if sequence.count(MPRA_PARAMS.RS_ECORI) != 0: return False
    if sequence.count(MPRA_PARAMS.RS_BAMHI) != 0: return False

    # cut sites (insert promoter and luc) should NOT exist
    if sequence.count(MPRA_PARAMS.RS_XHOI) != 0: return False
    if sequence.count(MPRA_PARAMS.RS_XBAI) != 0: return False
    
    return True


def is_fragment_compatible(sequence):
    """check fragment for compatibility
    """
    # check for NO cut sites
    if not is_rs_clean(sequence): return False

    # check again when attaching FWD primer
    if not is_rs_clean(MPRA_PARAMS.FWD_PCR_PRIMER + sequence): return False

    # check after attaching XHOI
    fragment_extended = sequence + MPRA_PARAMS.RS_XHOI
    if fragment_extended.count(MPRA_PARAMS.RS_ECORI) != 0: return False
    if fragment_extended.count(MPRA_PARAMS.RS_BAMHI) != 0: return False
    
    # check for N
    if sequence.count("N") != 0: return False
    
    return True
    

def is_barcode_compatible(barcode):
    """check barcode for compatibility
    """
    # check when attaching REV primer
    if not is_rs_clean(barcode + MPRA_PARAMS.REV_PCR_PRIMER): return False

    # check when attaching XBAI site
    barcode_extended = MPRA_PARAMS.RS_XBAI + barcode
    if barcode_extended.count(MPRA_PARAMS.RS_XBAI_dam2) != 0: return False
    
    # check overlaps
    if barcode_extended.count(MPRA_PARAMS.RS_ECORI) != 0: return False
    if barcode_extended.count(MPRA_PARAMS.RS_BAMHI) != 0: return False
    
    return True


def is_filler_compatible(filler):
    """check if filler is compatible with library design
    """
    # first check cut sites
    if not is_rs_clean(filler): return False
    
    # check overlaps
    filler_extended = MPRA_PARAMS.RS_XHOI + filler + MPRA_PARAMS.RS_XBAI
    if filler_extended.count(MPRA_PARAMS.RS_ECORI) != 0: return False
    if filler_extended.count(MPRA_PARAMS.RS_BAMHI) != 0: return False
    
    # check methylation
    if filler_extended.count(MPRA_PARAMS.RS_XBAI_dam1) != 0: return False
        
    return True


def generate_compatible_filler(rand_seed):
    """generate filler sequence and check methylation
    """
    while True:
        # generate random sequence
        rand_state = RandomState(rand_seed)
        random_seq = rand_state.choice(
            MPRA_PARAMS.LETTERS,
            size=MPRA_PARAMS.LEN_FILLER)
        random_seq = "".join(random_seq)
        
        # if passes checks then break
        if is_filler_compatible(random_seq):
            break

        # otherwise change the seed and keep going
        rand_seed += 1

    # and move up one more (for next filler)
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
    # get min possible len that contains motifs and check less than max possible len
    motif_padding_len = 10
    frag_minimal_len = edge_indices[1] - edge_indices[0] + (motif_padding_len * 2)
    assert frag_minimal_len < MPRA_PARAMS.MAX_FRAG_LEN

    # add padding and adjust start/stop
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

    # sanity check
    assert is_sequence_mpra_ready(sequence), log
    
    return sequence, rand_seed


def seq_list_compatible(seq_list, left_clip=420, right_clip=580):
    """helper function when checking sequences from h5 file
    """
    for seq in seq_list:
        if not is_fragment_compatible(seq[420:580]):
            return False
        if len(seq[420:580]) == 0:
            return False
        
    return True


def extract_sequence_info(h5_file, examples, left_clip=420, right_clip=580, prefix="id"):
    """save sequences to a df
    """
    # adjust indices for specific file
    with h5py.File(h5_file, "r") as hf:
        indices = np.where(np.isin(hf[DataKeys.SEQ_METADATA][:,0], examples))[0]
    assert indices.shape[0] == examples.shape[0]
        
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
                key_df = pd.DataFrame(
                    {key:key_df.astype(str).apply(",".join, axis=1)})
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
    
    # set up a combo id
    all_results.insert(
        0, "example_combo_id",
        prefix + "." + all_results["example_fileidx"].astype(str) + ".combo-" + all_results["combos"].str.replace(",", ""))

    return all_results


def get_synergy_file(synergy_dirs, grammar_prefix):
    """get synergy file
    """
    synergy_files = []
    for synergy_dir in synergy_dirs:
        synergy_file = "{}/{}/ggr.synergy.h5".format(synergy_dir, grammar_prefix)
        if os.path.isfile(synergy_file):
            synergy_files.append(synergy_file)
    assert len(synergy_files) == 1
    synergy_file = synergy_files[0]

    return synergy_file


def get_compatible_sequence_indices(sequences):
    """get indices of MPRA compatible sequences
    """
    compatible = []
    for seq_idx in range(sequences.shape[0]):
        if seq_list_compatible(sequences[seq_idx].squeeze().tolist()):
            compatible.append(seq_idx)

    return compatible


def get_consensus_sequences_across_runs(synergy_files):
    """compare synergy files
    """
    for file_idx in range(len(synergy_files)):
        with h5py.File(synergy_files[file_idx]) as hf:
            sequences = hf["{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ)][:] # {N, combos, 1}
            examples = hf[DataKeys.SEQ_METADATA][:,0]

        # only keep compatible
        compatible_indices = get_compatible_sequence_indices(sequences)
        examples = set(examples[compatible_indices].tolist())
        if file_idx == 0:
            consensus = examples
        else:
            consensus = consensus.intersection(examples)

    return sorted(list(consensus))


def string_arrays_equal(a, b):
    """not yet implemented in numpy??
    """
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False

    return True


def get_sampling_info_across_runs(synergy_files, examples):
    """collect information needed for guided sampling
    """
    for file_idx in range(len(synergy_files)):
        synergy_file = synergy_files[file_idx]
        with h5py.File(synergy_file, "r") as hf:
            # set up indices
            file_indices = np.where(np.isin(hf[DataKeys.SEQ_METADATA][:,0], examples))[0]
            assert file_indices.shape[0] == len(examples), "{}, {}".format(file_indices.shape[0], len(examples))

            # get examples
            file_metadata = hf[DataKeys.SEQ_METADATA][:,0][file_indices]
            sort_indices = np.argsort(file_metadata, axis=0)

            # sort on examples
            file_metadata = file_metadata[sort_indices]
            file_distances = hf["{}.0".format(DataKeys.SYNERGY_DIST)][:][file_indices][sort_indices] # {N}
            file_diffs_sig = hf["{}.0".format(DataKeys.SYNERGY_DIFF_SIG)][:][file_indices][sort_indices] # {N, logit}
            file_max_dist = hf["{}.0".format(DataKeys.SYNERGY_MAX_DIST)][()]
            file_motif_positions = hf[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT][:][file_indices][sort_indices][:,-1]
            max_pos = np.amax(file_motif_positions, axis=1)
            min_pos = np.amin(file_motif_positions, axis=1)
            file_grammar_centers = min_pos + np.divide(np.subtract(max_pos, min_pos), 2.) - GGR_LEFT_CLIP
            
        # merge
        if file_idx == 0:
            sampling_info = {
                "examples": file_metadata,
                "distances": file_distances,
                "centers": file_grammar_centers,
                "diff_sig": [file_diffs_sig],
                "max_dist": [file_max_dist]}
        else:
            assert string_arrays_equal(sampling_info["examples"], file_metadata)
            sampling_info["diff_sig"].append(file_diffs_sig)
            sampling_info["max_dist"].append(file_max_dist)

    # finally calculate means
    sampling_info["diff_sig"] = np.mean(sampling_info["diff_sig"], axis=0)
    sampling_info["max_dist"] = np.mean(sampling_info["max_dist"], axis=0)
    
    return sampling_info


def minimize_overlapping(sampling_info):
    """reduce ones that come from same region
    
    TODO note that same region can contain the instance MORE THAN ONCE - fine tune this

    """
    # make a df to easily manipulate
    reduce_df = pd.DataFrame(
        {"examples": sampling_info["examples"]})
    reduce_df = reduce_df["examples"].str.split(";", 3, expand=True)
    reduce_df["centers"] = np.abs(
        sampling_info["centers"] - ((GGR_RIGHT_CLIP - GGR_LEFT_CLIP) / 2))
    reduce_df["orig_indices"] = list(reduce_df.index)
    
    # get common regions
    regions = sorted(list(set(reduce_df.iloc[:,0].values.tolist())))

    # for each region, choose most centered
    best_indices = []
    for region in regions:
        region_set = reduce_df[reduce_df.iloc[:,0] == region]
        if region_set.shape[0] > 1:
            best_region_rowname = region_set["centers"].idxmin()
            best_indices.append(region_set.loc[best_region_rowname, "orig_indices"])
        else:
            best_indices.append(region_set.iloc[0]["orig_indices"])

    # and now reduce
    for key in sampling_info:
        if key == "max_dist":
            continue
        sampling_info[key] = sampling_info[key][best_indices]
    
    return sampling_info


def filter_for_region_set(sampling_info, sample_regions_file, filter_sampling_info=True):
    """given a sample regions file, only keep regions that fall into those regions
    """
    # get regions and make a BED file. KEEP ORDER
    filter_df = pd.DataFrame(
        {"examples": sampling_info["examples"]})
    filter_df = filter_df["examples"].str.split(";", 3, expand=True).iloc[:,1]
    filter_df = filter_df.str.split("=", 2, expand=True).iloc[:,1]
    filter_df = filter_df.str.split(":", 2, expand=True)
    chrom = filter_df.iloc[:,0]
    filter_df = filter_df.iloc[:,1].str.split("-", 2, expand=True)
    filter_df.insert(0, "chrom", chrom)
    
    # make bed file, overlap, return dataframe of results
    tmp_bed = "sampling.bed.gz"
    filter_df.to_csv(tmp_bed, sep="\t", header=False, index=False, compression="gzip")

    # overlap
    tmp_overlap_bed = "sampling.overlap.bed.gz"
    intersect = "bedtools intersect -c -a {} -b {} | gzip -c > {}".format(
        tmp_bed, sample_regions_file, tmp_overlap_bed)
    os.system(intersect)

    # read back in
    filter_df = pd.read_table(tmp_overlap_bed, header=None, index_col=False)

    # and then filter on this
    keep_indices = np.where(filter_df.iloc[:,3] == 1)[0]
    if filter_sampling_info:
        for key in sampling_info:
            if key == "max_dist":
                continue
            sampling_info[key] = sampling_info[key][keep_indices]

    # and clean up tmp files
    os.system("rm {} {}".format(tmp_bed, tmp_overlap_bed))
    
    return sampling_info, keep_indices


def get_diff_sample(sampling_info, sample_num):
    """get differential group of regions
    do this by starting with strictest threshold and gradually
    loosening - this ensures you get a sample of diff seqs 
    that are most diff. note that this is not effect size.
    """
    diffs_sig = sampling_info["diff_sig"] # {N, logit}
    min_diff = diffs_sig.shape[1]
    num_tasks_diff = np.sum(diffs_sig, axis=1)
    while True:
        diff_bool = np.greater_equal(num_tasks_diff, min_diff) # {N}
        diff_indices = np.where(diff_bool)[0]
        try:
            diff_sample_indices = diff_indices[
                np.random.choice(
                    diff_indices.shape[0],
                    sample_num,
                    replace=False)]
            break
        except ValueError:
            min_diff -= 1

        if min_diff == 0:
            diff_sample_indices = diff_indices
            break

    # convert from indices to example metadata
    diff_sample = sampling_info["examples"][diff_sample_indices]
    
    return diff_sample, diff_bool


def get_nondiff_proximal_sample(sampling_info, sample_num, diff_bool, min_dist=12):
    """get nondiff proximal examples
    """
    # get nondiff proximal indices
    distances = sampling_info["distances"]
    nondiff = np.logical_not(diff_bool) # {N}
    nondiff_bool = np.logical_and(nondiff, distances < sampling_info["max_dist"])
    nondiff_bool = np.logical_and(nondiff_bool, distances > min_dist)
    nondiff_indices = np.where(nondiff_bool)[0]

    # sample
    try:
        nondiff_sample_indices = nondiff_indices[
            np.random.choice(
                nondiff_indices.shape[0],
                sample_num,
                replace=False)]
    except ValueError:
        nondiff_sample_indices = nondiff_indices

    # convert from indices to example metadata
    nondiff_sample = sampling_info["examples"][nondiff_sample_indices]
        
    return nondiff_sample


def get_nondiff_distal_sample(sampling_info, sample_num, diff_bool):
    """get nondiff proximal examples
    """
    # get nondiff distal indices
    distances = sampling_info["distances"]
    nondiff = np.logical_not(diff_bool) # {N}
    nondiff_bool = np.logical_and(nondiff, distances > sampling_info["max_dist"])
    nondiff_indices = np.where(nondiff_bool)[0]

    # sample
    try:
        nondiff_sample_indices = nondiff_indices[
            np.random.choice(
                nondiff_indices.shape[0],
                sample_num,
                replace=False)]
    except ValueError:
        nondiff_sample_indices = nondiff_indices

    # convert from indices to example metadata
    nondiff_sample = sampling_info["examples"][nondiff_sample_indices]
        
    return nondiff_sample


def sample_sequences(
        summary_df,
        num_runs,
        sample_regions_file=None,
        required_regions_file=None):
    """sample sequences
    """
    diff_sample_num = 10
    nondiff_proximal_sample_num = 5
    nondiff_distal_sample_num = 5
    total_sample_num = diff_sample_num + nondiff_proximal_sample_num + nondiff_distal_sample_num
    min_dist = 12

    mpra_runs = [None]*num_runs
    for grammar_idx in range(summary_df.shape[0]):
        # set seed each time you run a new grammar
        np_seed = np.random.seed(1337)
        
        # pull out list of synergy files
        synergy_files = summary_df["combined"].iloc[grammar_idx]
    
        # get consensus examples across the files
        seq_metadata = get_consensus_sequences_across_runs(synergy_files)
        
        # collect relevant info for guided sampling
        sampling_info = get_sampling_info_across_runs(synergy_files, seq_metadata)
        
        # reduce out overlapping (NOTE: this may be over strict!)
        sampling_info = minimize_overlapping(sampling_info)

        # only use sequences that are within desired regions
        if sample_regions_file is not None:
            sampling_info, _ = filter_for_region_set(sampling_info, sample_regions_file)
        
        # get diff sample
        diff_sample, diff_bool = get_diff_sample(sampling_info, diff_sample_num)
        if diff_sample.shape[0] < (diff_sample_num / 2.):
            logging.info("{}: skipping because too few diff seqs: {}".format(grammar_idx, diff_sample.shape[0]))
            continue

        # get nondiff proximal
        nondiff_proximal_sample = get_nondiff_proximal_sample(
            sampling_info, nondiff_proximal_sample_num, diff_bool)
        if nondiff_proximal_sample.shape[0] < (nondiff_proximal_sample_num / 2.):
            logging.info("{}: skipping because too few nondiff proximal seqs: {}".format(grammar_idx, nondiff_proximal_sample.shape[0]))
            continue

        # get nondiff distal
        nondiff_distal_sample = get_nondiff_distal_sample(
            sampling_info, nondiff_distal_sample_num, diff_bool)
        if nondiff_distal_sample.shape[0] < (nondiff_distal_sample_num / 2.):
            logging.info("{}: skipping because too few nondiff distal seqs: {}".format(grammar_idx, nondiff_distal_sample.shape[0]))
            continue

        # now get required regions and keep ALL
        if required_regions_file is not None:
            _, required_indices = filter_for_region_set(
                sampling_info, required_regions_file, filter_sampling_info=False)
            required_sample = sampling_info["examples"][required_indices]
            diff_sample = np.concatenate([diff_sample, required_sample], axis=0)
            
        # collect all
        full_sample = np.unique(np.concatenate(
            [diff_sample,
             nondiff_proximal_sample,
             nondiff_distal_sample], axis=0))
        if full_sample.shape[0] < (total_sample_num / 2.):
            logging.info("{}: skipping because not enough sequences of interest: {}".format(grammar_idx, full_sample.shape[0]))
            continue
        logging.info("{}: sampled {}, got {} required".format(grammar_idx, full_sample.shape[0], required_sample.shape[0]))
        
        # and now extract
        grammar_info_per_run = []
        for synergy_file in synergy_files:
            grammar_prefix = synergy_file.split("/")[-2]
            mpra_sample_df = extract_sequence_info(
                synergy_file, full_sample, prefix=grammar_prefix)
            grammar_info_per_run.append(mpra_sample_df)

        # concatenate
        if mpra_runs[0] is None:
            mpra_runs = grammar_info_per_run
        else:
            for mpra_run_idx in range(len(mpra_runs)):
                mpra_runs[mpra_run_idx] = pd.concat([mpra_runs[mpra_run_idx], grammar_info_per_run[mpra_run_idx]])

    return mpra_runs

            
def build_mpra(args, mpra_all_df, run_idx):
    """build mpra
    """
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
    mpra_expanded_df = None
    for seq_idx in range(mpra_all_df.shape[0]):
        if seq_idx % 1000 == 0:
            print seq_idx

        # get the whole row out
        seq_info = pd.DataFrame(mpra_all_df.iloc[seq_idx]).transpose()
        example_id = seq_info.iloc[0]["example_id"]
        sequence = seq_info.iloc[0]["sequence.nn"]
        edge_indices = [
            int(float(val)) for val in seq_info.iloc[0]["edge_indices"].split(",")]

        # trim, if trimming doesn't work add to incompatible
        try:
            sequence = trim_sequence_for_mpra(sequence, edge_indices)
        except AssertionError:
            incompatible.append(example_id)
            continue
            
        # after trimming, you reintroduce opportunities for the fragment to be wrong
        if not is_fragment_compatible(sequence):
            incompatible.append(example_id)
            continue
        
        # generate with barcodes
        for barcode_idx in range(args.barcodes_per_sequence):
            # set up copy
            barcoded_info = seq_info.copy()

            # adjust the unique id
            barcoded_info.insert(0, "unique_id", "{}.barcode-{}".format(
                barcoded_info.iloc[0]["example_combo_id"], barcode_idx))
            
            # get a barcode
            while True:
                barcode = barcodes[barcode_idx]
                if is_barcode_compatible(barcode):
                    break
                barcode_idx += 1
            barcode_idx += 1

            # build sequence and add in
            mpra_sequence, rand_seed = build_mpra_sequence(sequence, barcode, rand_seed, seq_idx)
            barcoded_info.insert(1, "sequence.mpra", mpra_sequence)
            
            # add on to full set
            if mpra_expanded_df is None:
                mpra_expanded_df = barcoded_info
            else:
                mpra_expanded_df = pd.concat([mpra_expanded_df, barcoded_info])

    print len(incompatible)
    
    # drop incompatible
    print mpra_expanded_df.shape
    mpra_expanded_df = mpra_expanded_df[~mpra_expanded_df["example_id"].isin(incompatible)]
    print mpra_expanded_df.shape

    if True:
        # look at duplicates
        print len(set(mpra_expanded_df["example_id"].values.tolist()))
        print len(mpra_expanded_df["example_metadata"].values.tolist())
        print len(set(mpra_expanded_df["example_metadata"].values.tolist()))
        print len(mpra_expanded_df["sequence.nn"].values.tolist())
        print len(set(mpra_expanded_df["sequence.nn"].values.tolist()))
    
    # finally save out
    mpra_expanded_df.to_csv("{}/mpra.seqs.run-{}.txt".format(args.out_dir, run_idx), sep="\t")
    
    return mpra_expanded_df


def build_consensus_file_sets(grammar_summaries, synergy_dirs):
    """take the grammar sets and build consensus sets
    """
    # merge the summaries
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
            synergy_file = get_synergy_file(run_dirs, grammar_prefix)
            all_summaries.loc[df_index,run] = synergy_file

    # TODO check synergy files, make sure all are valid
            
    # make into list
    all_summaries["combined"] = all_summaries.values.tolist()
            
    return all_summaries


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    # set up grammar summaries
    args.grammar_summaries = [tuple(val.split("=")) for val in args.grammar_summaries]
    summary_df = build_consensus_file_sets(args.grammar_summaries, args.synergy_dirs)
    
    # now sample using consensus
    mpra_runs = sample_sequences(
        summary_df,
        len(args.grammar_summaries),
        sample_regions_file=args.sample_regions,
        required_regions_file=args.required_regions)

    # for each run, build mpra
    for mpra_run_idx in range(len(mpra_runs)):
        build_mpra(args, mpra_runs[mpra_run_idx], mpra_run_idx)
    
    return None


if __name__ == "__main__":
    main()
