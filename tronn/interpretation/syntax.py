"""description: code for elucidating grammatical syntax
"""

import os
import h5py

import numpy as np
import pandas as pd

from multiprocessing import Pool

from tronn.util.h5_utils import load_data_from_multiple_h5_files
from tronn.util.utils import DataKeys


def align_examples_by_max_pwm(
        score_array,
        max_indices,
        left_clip=432,
        final_extend_len=136):
    """takes pwm scores and max idx and align according to max val
    """
    # set up aligned array
    num_positions = score_array.shape[1]
    aligned_array = np.zeros((score_array.shape[0], 2*(num_positions+20)))
    
    # align each row separately
    for example_idx in range(score_array.shape[0]):
        offset = num_positions - (int(max_indices[example_idx]) - left_clip)
        aligned_array[example_idx, offset:(offset+num_positions)] = score_array[example_idx]
            
    # clean up edges
    clip_start = num_positions - final_extend_len
    clip_stop = num_positions + final_extend_len
    aligned_array = aligned_array[:,clip_start:clip_stop]
    
    return aligned_array


def get_syntax_matching_examples(
        max_vals, pwm_indices, orientations):
    """
    """
    # rc index (as needed)
    num_pwms_if_rc = max_vals.shape[1] / 2
    
    # go through each pwm and pull appropriate data
    for pwm_idx in range(len(pwm_indices)):
        global_pwm_idx = pwm_indices[pwm_idx]
        rc_idx = global_pwm_idx + num_pwms_if_rc # as needed

        # pull data
        if orientations[pwm_idx] == "FWD":
            pwm_present = max_vals[:,global_pwm_idx,0] > 0
        elif orientations[pwm_idx] == "REV":
            pwm_present = max_vals[:,rc_idx,0] > 0
        elif orientations[pwm_idx] == "BOTH":
            pwm_present = np.logical_or(
                max_vals[:,global_pwm_idx,0] > 0,
                max_vals[:,rc_idx,0] > 0)
        else:
            raise ValueError, "don't recognize orientation!"

        # merge as AND logic
        if pwm_idx == 0:
            keep_examples = pwm_present
        else:
            keep_examples = np.logical_and(
                keep_examples, pwm_present)

    # get file indices
    keep_examples = np.where(keep_examples)[0]

    return keep_examples


def _load_pwm_scores(args):
    """load pwm scores
    """
    h5_file, scores_key, example_indices, global_pwm_idx, orientation = args

    # rc index (as needed)
    with h5py.File(h5_file, "r") as hf:
        num_pwms_if_rc = hf[scores_key].shape[3] / 2
    rc_idx = global_pwm_idx + num_pwms_if_rc
        
    with h5py.File(h5_file, "r") as hf:
        # pull data
        if orientation == "FWD":
            pwm_scores = hf[scores_key][example_indices,:,:,global_pwm_idx]
        elif orientation == "REV":
            pwm_scores = hf[scores_key][example_indices,:,:,rc_idx]            
        elif orientation == "BOTH":
            pwm_scores = np.sum(
                [
                    hf[scores_key][example_indices,:,:,global_pwm_idx],
                    hf[scores_key][example_indices,:,:,rc_idx]
                ], axis=0)
        
    return pwm_scores
            

def analyze_syntax(
        h5_files,
        pwm_indices,
        max_val_key=DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL,
        max_idx_key=DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX,
        scores_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH,
        metadata_key=DataKeys.SEQ_METADATA,
        signal_keys={"ATAC_SIGNALS": [0,6,12], "H3K27ac_SIGNALS": [0,1,2]},
        orientations=["FWD", "FWD"],
        reduce_method="max",
        left_clip=432,
        final_extend_len=136,
        min_region_count=500,
        num_threads=24):
    """analyze syntax by aligning to anchor pwm and seeing what patterns
    look like around it

    Args:
      - h5_files: files with data
      - pwm_indices: which pwms are being tested. MUST provide at least TWO.
      - orientations: what orientation to test (FWD, REV, BOTH)
    """
    # assertions
    assert len(pwm_indices) > 1
    assert len(pwm_indices) == len(orientations)
    
    # load max vals (to determine which examples are of interest)
    max_vals_per_file = load_data_from_multiple_h5_files(
        h5_files, max_val_key, concat=False)
    
    # determine which examples fit the desired syntax
    example_indices = []
    for max_vals in max_vals_per_file:
        file_examples = get_syntax_matching_examples(
            max_vals, pwm_indices, orientations)
        example_indices.append(file_examples)

    # then collect data
    max_indices = load_data_from_multiple_h5_files(
        h5_files, max_idx_key, example_indices=example_indices)[:,pwm_indices[0],0]
    metadata = load_data_from_multiple_h5_files(
        h5_files, metadata_key, example_indices=example_indices)[:,0]
    signals = {}
    for key in signal_keys.keys():
        signals[key] = load_data_from_multiple_h5_files(
            h5_files, key, example_indices=example_indices)
        
    # do not continue if don't have enough regions
    if max_indices.shape[0] < min_region_count:
        return None

    # collect scores for pwms
    scores = []
    for pwm_idx in range(1, len(pwm_indices)):
        global_pwm_idx = pwm_indices[pwm_idx]
        print pwm_idx, global_pwm_idx
        pwm_scores = []

        # use pool to speed up loading
        pool = Pool(processes=min(num_threads, len(h5_files)))
        pool_inputs = []
        for h5_idx in range(len(h5_files)):
            pool_inputs.append(
                (h5_files[h5_idx],
                 scores_key,
                 example_indices[h5_idx],
                 global_pwm_idx,
                 orientations[pwm_idx]))
        pool_outputs = pool.map(_load_pwm_scores, pool_inputs)
        pwm_scores = np.concatenate(pool_outputs, axis=0)
        if reduce_method == "max":
            pwm_scores = np.max(pwm_scores, axis=1)
        else:
            raise ValueError, "Unimplemented reduce method!"
        print pwm_scores.shape
        scores.append(pwm_scores)
        
    # align each
    aligned_results = {}
    for pwm_idx in range(len(pwm_indices)-1):
        global_pwm_idx = pwm_indices[pwm_idx]
        aligned_results[global_pwm_idx] = {}

        # scores
        pwm_aligned_scores = align_examples_by_max_pwm(
            scores[pwm_idx], max_indices, left_clip=left_clip, final_extend_len=final_extend_len)
        positions = np.arange(pwm_aligned_scores.shape[1]) - int(pwm_aligned_scores.shape[1]/2.)
        scores_df = pd.DataFrame(data=pwm_aligned_scores, columns=positions.tolist())
        aligned_results[global_pwm_idx]["scores"] = scores_df
        
        # signals (use pwm hits to mark positions)
        pwm_aligned_hits = (pwm_aligned_scores > 0).astype(int)
        for key in signal_keys.keys():
            print key
            aligned_results[global_pwm_idx][key] = {}
            signal_task_indices = signal_keys[key]
            for task_idx in signal_task_indices:
                print task_idx
                # pull task signals
                task_signals = signals[key][:,task_idx]
                task_signals = np.expand_dims(task_signals, axis=-1)
                task_results = np.multiply(pwm_aligned_hits, task_signals) # {N, 321}
                
                # flatten, remove zeros
                task_df = pd.DataFrame(data=task_results, columns=positions.tolist())
                task_df = task_df[np.sum(task_df.values, axis=1) != 0]
                task_melt_df = task_df.melt()
                task_melt_df = task_melt_df[task_melt_df["value"] != 0]
                aligned_results[global_pwm_idx][key][task_idx] = task_melt_df
        
    return aligned_results
