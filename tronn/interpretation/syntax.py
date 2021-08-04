"""description: code for elucidating grammatical syntax
"""

import os
import h5py
import logging

import numpy as np
import pandas as pd

from multiprocessing import Pool

from tronn.util.formats import array_to_bed
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


def get_syntax_matching_examples(max_vals, pwm_indices):
    """requires that examples have positive scores for ALL
    pwm indices in list
    """
    # go through each pwm and pull appropriate data
    for i in range(len(pwm_indices)):
        global_pwm_idx = pwm_indices[i]
        pwm_present = max_vals[:,global_pwm_idx,0] > 0

        # merge as AND logic
        if i == 0:
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
    h5_file, scores_key, example_indices, global_pwm_idx = args
        
    with h5py.File(h5_file, "r") as hf:
        # pull data
        pwm_scores = hf[scores_key][example_indices,:,:,global_pwm_idx]
        
    return pwm_scores


def _load_importances(args):
    """load importances
    """
    h5_file, importances_key, example_indices, clip_start, clip_end = args
    with h5py.File(h5_file, "r") as hf:
        importances = np.sum(
            hf[importances_key][example_indices,:,:,:], axis=(1,3))
        importances = importances[:,clip_start:clip_end]

    return importances


def filter_for_pwm_importance_overlap(importances, pwm_scores, fract_thresh=0.9, window=20):
    """using max index to mark, get importance fract
    """
    assert importances.shape[0] == pwm_scores.shape[0]
    
    # set up results arrays
    pwm_impt_coverage = np.zeros((pwm_scores.shape[0]))

    # go through per example
    for example_idx in range(importances.shape[0]):
        example_importances = importances[example_idx] > 0
        
        # figure out how many positions were marked
        impt_sum_total = np.sum(example_importances).astype(float)
            
        # per index, get window around and collect importance scores
        pwm_pos_indices = np.where(pwm_scores[example_idx] > 0)[0]
            
        # collect importance scores within window
        impt_sum_pwm_overlap = 0.
        for pwm_pos_idx in pwm_pos_indices:
            start = max(pwm_pos_idx - window/2, 0)
            end = min(pwm_pos_idx + window/2, importances.shape[1])
            impt_sum_pwm_overlap += np.sum(example_importances[start:end]).astype(float)
        
        # and check
        fract_covered = impt_sum_pwm_overlap / impt_sum_total
        pwm_impt_coverage[example_idx] = fract_covered

    # filter
    keep_indices = np.where(pwm_impt_coverage >= fract_thresh)[0]
    
    return keep_indices


def analyze_multiplicity(
        h5_files,
        pwm_indices,
        max_val_key=DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL,
        scores_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH,
        metadata_key=DataKeys.SEQ_METADATA,
        signal_keys={"ATAC_SIGNALS": [0,6,12], "H3K27ac_SIGNALS": [0,1,2]},
        reduce_method="max",
        max_count=6,
        min_region_count=500,
        min_hit_region_count=50,
        solo_filter=False,
        solo_filter_fract=0.9,
        solo_filter_window=20,
        importances_key=DataKeys.WEIGHTED_SEQ_ACTIVE,
        impt_clip_start=12,
        impt_clip_end=148,
        num_threads=24):
    """analyze syntax by aligning to anchor pwm and seeing what patterns
    look like around it

    Args:
      - h5_files: files with data
      - pwm_indices: which pwms are being tested. MUST provide at least TWO.
      - orientations: what orientation to test (FWD, REV, BOTH)
    """
    # load max vals and determine which examples are of interest
    # don't care about which direction, so load jointly
    max_vals_per_file = load_data_from_multiple_h5_files(
        h5_files, max_val_key, concat=False)
    example_indices = []
    for max_vals in max_vals_per_file:
        file_examples = []
        for pwm_idx in pwm_indices:
            file_examples += get_syntax_matching_examples(
                max_vals, [pwm_idx]).tolist()
        file_examples = np.array(sorted(list(set(file_examples))))
        example_indices.append(file_examples)
        
    # do not continue if don't have enough regions
    total_regions = np.sum([examples.shape[0] for examples in example_indices])
    print total_regions
    if total_regions < min_region_count:
        return None
    
    # TODO - here could also choose to utilize mutate motifs to reduce the number
    # being considered.
    # this is complicated because need to go through ALL traj mutatemotifs results
    
    # collect scores for pwms
    scores = [] # list of arrays, one for each pwm
    for pwm_idx in range(len(pwm_indices)):
        global_pwm_idx = pwm_indices[pwm_idx]
        logging.info("collecting scores for {}".format(global_pwm_idx))
        
        # use pool to speed up loading
        pool = Pool(processes=min(num_threads, len(h5_files)))
        pool_inputs = []
        for h5_idx in range(len(h5_files)):
            pool_inputs.append(
                (h5_files[h5_idx],
                 scores_key,
                 example_indices[h5_idx],
                 global_pwm_idx))
        pool_outputs = pool.map(_load_pwm_scores, pool_inputs)
        pwm_scores = np.concatenate(pool_outputs, axis=0)
        # reduce across tasks
        if reduce_method == "max":
            pwm_scores = np.max(pwm_scores, axis=1)
        else:
            raise ValueError, "Unimplemented reduce method!"
        # DON'T SUM YET ACROSS POSITIONS - need for solo filter
        scores.append(pwm_scores)

    scores = np.sum(scores, axis=0) # {N, M} <- non-oriented now, since combined fwd/rev
    print scores.shape

    # metadata and signals
    metadata = load_data_from_multiple_h5_files(
        h5_files, metadata_key, example_indices=example_indices)[:,0]
    signals = {}
    for key in signal_keys.keys():
        signals[key] = load_data_from_multiple_h5_files(
            h5_files, key, example_indices=example_indices)
        
    # if solo filter, need to load importances and filter with those thresholds
    if solo_filter:
        logging.info("using solo filtering")
        # use pool to load faster
        pool = Pool(processes=min(num_threads, len(h5_files)))
        pool_inputs = []
        for h5_idx in range(len(h5_files)):
            pool_inputs.append(
                (h5_files[h5_idx],
                 importances_key,
                 example_indices[h5_idx],
                 impt_clip_start,
                 impt_clip_end))
        pool_outputs = pool.map(_load_importances, pool_inputs)
        importances = np.concatenate(pool_outputs, axis=0)

        # get solo indices. NOTE: the solo indices reference indices AFTER applying
        # example_indices!!
        solo_indices = filter_for_pwm_importance_overlap(
            importances,
            scores,
            fract_thresh=solo_filter_fract,
            window=solo_filter_window)

        print solo_indices.shape[0]
        if solo_indices.shape[0] < min_region_count:
            return None
        
        # filter all: scores, metadata, signals
        scores = scores[solo_indices]
        metadata = metadata[solo_indices]
        for key in signal_keys.keys():
            signals[key] = signals[key][solo_indices]

    print "examples found:", metadata.shape[0]
    
    # convert to hits
    hits = np.sum(scores > 0, axis=1)
    hits = pd.DataFrame(data=hits, index=metadata, columns=["hits"])
    hits["present"] = 1
    
    # attach signals and aggregate
    results = {"hits_per_region": hits}
    for key in sorted(signal_keys.keys()):
        signal_task_indices = signal_keys[key]
        results[key] = {}
        results[key]["count"] = np.zeros((
            len(signal_task_indices),
            max_count))
        for i in range(len(signal_task_indices)):
            task_idx = signal_task_indices[i]
            task_hits = hits.copy()
            task_hits[task_idx] = signals[key][:,task_idx]
            # aggregate
            task_results = task_hits.groupby("hits").median()
            task_results["present"] = task_hits.groupby("hits")["present"].sum()
            task_results = task_results[task_results["present"] > min_hit_region_count]

            # save in
            for count in range(1,max_count+1):
                if count in task_results.index:
                    results[key]["count"][i,count-1] = task_results.loc[count][task_idx]

    # also get num regions per count level
    results["num_regions_per_count"] = np.zeros(max_count)
    for count in range(1,max_count+1):
        regions_per_count = hits[hits["hits"] == count]
        if regions_per_count.shape[0] > 0:
            # convert to bed (active regions) and merge
            # this is to make sure don't count duplicate instances (overlapping strides)
            tmp_bed_file = "tronn.count_thresh.tmp.bed.gz"
            array_to_bed(regions_per_count.index.values, tmp_bed_file, merge=True)
            
            # count hits
            num_regions_per_count = pd.read_csv(tmp_bed_file, sep="\t", header=None).shape[0]
            results["num_regions_per_count"][count-1] = num_regions_per_count
            
            # cleanup
            os.system("rm {}".format(tmp_bed_file))

    import ipdb
    ipdb.set_trace()
                    
    return results


def analyze_syntax(
        h5_files,
        pwm_indices,
        max_val_key=DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL,
        max_idx_key=DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX,
        scores_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH,
        metadata_key=DataKeys.SEQ_METADATA,
        signal_keys={"ATAC_SIGNALS": [0,6,12], "H3K27ac_SIGNALS": [0,1,2]},
        reduce_method="max",
        left_clip=432,
        final_extend_len=136,
        min_region_count=500,
        min_dist=7,
        solo_filter=False,
        solo_filter_fract=0.9,
        solo_filter_window=20,
        importances_key=DataKeys.WEIGHTED_SEQ_ACTIVE,
        impt_clip_start=12,
        impt_clip_end=148,
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
    
    # load max vals and determine which examples are of interest
    max_vals_per_file = load_data_from_multiple_h5_files(
        h5_files, max_val_key, concat=False)
    example_indices = []
    for max_vals in max_vals_per_file:
        file_examples = get_syntax_matching_examples(
            max_vals, pwm_indices)
        example_indices.append(file_examples)
        
    # set up anchor position (max index of first pwm in list)
    max_indices = load_data_from_multiple_h5_files(
        h5_files, max_idx_key, example_indices=example_indices)
    max_indices = max_indices[:,pwm_indices[0],0]
        
    # do not continue if don't have enough regions
    if max_indices.shape[0] < min_region_count:
        return None

    # collect scores for pwms
    scores = [] # list of arrays, one for each pwm
    for pwm_idx in range(1, len(pwm_indices)):
        global_pwm_idx = pwm_indices[pwm_idx]

        # use pool to speed up loading
        pool = Pool(processes=min(num_threads, len(h5_files)))
        pool_inputs = []
        for h5_idx in range(len(h5_files)):
            pool_inputs.append(
                (h5_files[h5_idx],
                 scores_key,
                 example_indices[h5_idx],
                 global_pwm_idx))
        pool_outputs = pool.map(_load_pwm_scores, pool_inputs)
        pwm_scores = np.concatenate(pool_outputs, axis=0)
        if reduce_method == "max":
            pwm_scores = np.max(pwm_scores, axis=1)
        else:
            raise ValueError, "Unimplemented reduce method!"
        scores.append(pwm_scores)

    # metadata and signals
    metadata = load_data_from_multiple_h5_files(
        h5_files, metadata_key, example_indices=example_indices)[:,0]
    signals = {}
    for key in signal_keys.keys():
        signals[key] = load_data_from_multiple_h5_files(
            h5_files, key, example_indices=example_indices)
        
    # if solo filter, need to load importances and filter with those thresholds
    if solo_filter:
        # use pool to load faster
        pool = Pool(processes=min(num_threads, len(h5_files)))
        pool_inputs = []
        for h5_idx in range(len(h5_files)):
            pool_inputs.append(
                (h5_files[h5_idx],
                 importances_key,
                 example_indices[h5_idx],
                 impt_clip_start,
                 impt_clip_end))
        pool_outputs = pool.map(_load_importances, pool_inputs)
        importances = np.concatenate(pool_outputs, axis=0)

        # get solo indices. NOTE: the solo indices reference indices AFTER applying
        # example_indices!!
        solo_indices = filter_for_pwm_importance_overlap(
            importances,
            pwm_scores,
            fract_thresh=solo_filter_fract,
            window=solo_filter_window)

        # filter all: scores, metadata, signals
        for i in range(len(scores)):
            scores[i] = scores[i][solo_indices]
        metadata = metadata[solo_indices]
        for key in signal_keys.keys():
            signals[key] = signals[key][solo_indices]

    print "examples found:", metadata.shape[0]
            
    # align each
    aligned_results = {}
    for pwm_idx in range(1,len(pwm_indices)):
        global_pwm_idx = pwm_indices[pwm_idx]
        aligned_results[global_pwm_idx] = {}

        # scores
        pwm_aligned_scores = align_examples_by_max_pwm(
            scores[pwm_idx-1], max_indices, left_clip=left_clip, final_extend_len=final_extend_len)
        positions = np.arange(pwm_aligned_scores.shape[1]) - int(pwm_aligned_scores.shape[1]/2.)
        scores_df = pd.DataFrame(
            data=pwm_aligned_scores,
            columns=positions.tolist(),
            index=metadata.tolist())
        scores_df.loc[:, np.abs(scores_df.columns.values) < min_dist] = 0 # min dist
        aligned_results[global_pwm_idx]["scores"] = scores_df
        
        # signals (use pwm hits to mark positions)
        pwm_aligned_hits = (pwm_aligned_scores > 0).astype(int)
        for key in signal_keys.keys():
            aligned_results[global_pwm_idx][key] = {}
            signal_task_indices = signal_keys[key]
            for task_idx in signal_task_indices:
                # pull task signals
                task_signals = signals[key][:,task_idx]
                task_signals = np.expand_dims(task_signals, axis=-1)
                task_results = np.multiply(pwm_aligned_hits, task_signals) # {N, 321}
                
                # flatten, remove zeros
                task_df = pd.DataFrame(
                    data=task_results,
                    columns=positions.tolist(),
                    index=metadata.tolist())
                task_df = task_df[np.sum(task_df.values, axis=1) != 0]
                task_melt_df = task_df.melt()
                task_melt_df.columns = ["position", "score"] # todo fix
                task_melt_df = task_melt_df[task_melt_df["score"] != 0]
                aligned_results[global_pwm_idx][key][task_idx] = task_melt_df
                
        # and then reduce the scores df further, if zeros
        scores_df = scores_df[np.sum(scores_df.values, axis=1) > 0]
        aligned_results[global_pwm_idx]["scores"] = scores_df
        
    return aligned_results


def recombine_syntax_results(results, orientations, anchor_sides, signal_keys):
    """adjust results for proper syntax results
    """
    assert len(orientations) == len(anchor_sides)
    recombined = {}
    
    # first recombine scores
    merged_scores = []
    for i in range(len(orientations)):
        orientation = orientations[i]
        anchor_side = anchor_sides[i]
        try:
            scores = results[orientation]["scores"].copy()
        except KeyError:
            return {}
        
        if anchor_side == "+":
            scores.loc[:,scores.columns.values <= 0] = 0
        elif anchor_side == "-":
            scores.loc[:,scores.columns.values >= 0] = 0
        merged_scores.append(scores)
    merged_scores = pd.concat(merged_scores, axis=0)
    merged_scores = merged_scores[np.sum(merged_scores.values, axis=1) > 0]
    recombined["scores"] = merged_scores

    # then recombine signals
    for key in signal_keys:
        recombined[key] = {}
        try:
            tasks = results[orientation][key].keys()
        except KeyError:
            return {}
        
        for task in tasks:
            task_signals = []
            for i in range(len(orientations)):
                orientation = orientations[i]
                anchor_side = anchor_sides[i]
                signals = results[orientation][key][task].copy()
                if anchor_side == "+":
                    signals = signals[signals["position"] > 0]
                elif anchor_side == "-":
                    signals = signals[signals["position"] < 0]
                task_signals.append(signals)
            task_signals = pd.concat(task_signals, axis=0)
            recombined[key][task] = task_signals
            
    return recombined
