"""Description: code to analyze motifs
"""

import os
import h5py
import math
import logging

import numpy as np
import pandas as pd

from numpy.random import RandomState

from tronn.stats.nonparametric import select_features_by_permutation_test
from tronn.stats.nonparametric import threshold_by_qvalues
from tronn.stats.parametric import run_hypergeometric_test

from tronn.interpretation.clustering import get_clusters_from_h5

from tronn.util.pwms import MotifSetManager
from tronn.util.utils import DataKeys
from tronn.util.h5_utils import AttrKeys


def build_gc_matched_bins(
        gc_array,
        background_gc_array,
        increment=0.05):
    """build a GC matched background, using GC fract information
    
    Returns:
      indices to use for pulling a GC background out

    """
    # get intervals
    num_increments = 1. /  increment
    sorted_gc = np.sort(gc_array) # use this for intervals
    intervals = np.interp(
        np.linspace(0, len(sorted_gc), num=int(num_increments)+1),
        range(len(sorted_gc)),
        sorted_gc)

    # set up a struct to keep indexes for different bins
    binned_indices = []

    # go through bins and collect sequences
    for i in xrange(len(intervals)-1):
        range_min = intervals[i]
        range_max = intervals[i+1]

        background_indices = np.where(
            (background_gc_array >= range_min) &
            (background_gc_array < range_max))[0]

        # maintain the list
        binned_indices.append(background_indices)
        
    return binned_indices



def build_gc_matched_background(
        binned_indices,
        num_per_bin,
        rand_state=RandomState()):
    """given a set of bins, select num_per_bin from each and merge
    together to get a final set of indices to examples, where the 
    gc content distribution should be matched
    """
    # TODO consider wrapping in a while loop
    # which checks to make sure the GC proporion is correct
    # and will downsample if you get misproportional background
    for i in xrange(len(binned_indices)):
        background_indices = binned_indices[i]

        # sample
        if num_per_bin < len(binned_indices[i]):
            background_indices = rand_state.choice(
                background_indices,
                size=num_per_bin,
                replace=False)
        else:
            print "NOTE: not enough background sequences in GC bin!!",
            print len(background_indices)

        # add to set
        if i == 0:
            all_background_indices = background_indices
        else:
            all_background_indices = np.concatenate(
                [all_background_indices, background_indices])

    return all_background_indices



def run_hypergeometric_test_on_motif_hits(
        positives_h5_file,
        background_h5_file,
        target_key,
        target_indices=[],
        pwm_hits_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        pwm_names_key=AttrKeys.PWM_NAMES,
        gc_key=DataKeys.GC_CONTENT,
        gc_increment=0.05):
    """
    """
    with h5py.File(positives_h5_file, "r") as hf:
        targets = hf[target_key][:]
        positives_gc = hf[gc_key][:]
        positive_hits = (hf[pwm_hits_key][:] > 0).astype(int)
        pwm_names = hf[pwm_hits_key].attrs[pwm_names_key]
        
    with h5py.File(background_h5_file, "r") as hf:
        background_gc = hf[gc_key][:]
        background_hits = (hf[pwm_hits_key][:] > 0).astype(int)

    # adjust target indices
    if len(target_indices) == 0:
        target_indices = range(targets.shape[1])

    # results array {target_indices, task, M}
    pvals = np.ones([len(target_indices)] + list(positive_hits.shape[1:]))
        
    # iterate through
    for i in xrange(len(target_indices)):
        target_idx = target_indices[i]
        subset_indices = np.where(targets[:,target_idx] > 0)[0]

        subset =  positive_hits[subset_indices]
        subset_gc = positives_gc[subset_indices]
        
        background_gc_bins = build_gc_matched_bins(
            subset_gc,
            background_gc,
            increment=gc_increment)

        num_per_bin = int(subset.shape[0] * gc_increment) + 50

        background_indices = build_gc_matched_background(
            background_gc_bins,
            num_per_bin,
            rand_state=RandomState(0))
        
        matched_background = background_hits[background_indices]

        for task_idx in xrange(subset.shape[1]):
            # and run test
            pvals[i,task_idx,:] = run_hypergeometric_test(
                subset[:,task_idx,:],
                matched_background[:,task_idx,:])

    positive_hits_summed = np.sum(positive_hits, axis=0)

    import ipdb
    ipdb.set_trace()

    # TODO - like homer would also be nice to keep fract of sites the hits exist
    

    return pvals


def save_subset_patterns_to_txt(
        results_h5_file,
        subset_targets,
        pwm_scores_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        pwm_names_key=AttrKeys.PWM_NAMES):
    """save out trajectory patterns to txt file (for filtering and vis)
    """
    # use h5 file to prefix the outputs
    prefix = results_h5_file.split(".h5")[0]
    
    with h5py.File(results_h5_file, "r") as hf:
        targets = hf[subset_targets[0]][:]
        pwm_scores = hf[pwm_scores_key][:]
        pwm_hits = pwm_scores > 0
        pwm_names = hf[pwm_scores_key].attrs[pwm_names_key]

        logits = hf["logits"][:]
        labels = hf["ATAC_SIGNALS.NORM"][:]

    if len(subset_targets[1]) != 0:
        targets = targets[:,subset_targets[1]]
        
    for subset_idx in xrange(targets.shape[1]):
        # filter
        subset_indices = np.where(targets[:,subset_idx] > 0)[0]
        subset_scores = np.sum(
            pwm_scores[subset_indices], axis=0).transpose()
        scores_df = pd.DataFrame(
            data=subset_scores,
            index=pwm_names)

        # save out
        out_file = "{}.{}-{}.pwm_scores.txt".format(
            prefix,
            subset_targets[0],
            subset_targets[1][subset_idx])
        scores_df.to_csv(out_file, sep="\t")
    
    return

def run_bootstrap_differential_score_test(
        foreground_h5_file,
        background_h5_file,
        foreground_targets_key,
        foreground_targets_indices,
        background_targets_key,
        background_targets_indices,
        pwm_hits_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        pwm_names_key=AttrKeys.PWM_NAMES,
        gc_key=DataKeys.GC_CONTENT,
        gc_increment=0.05,
        num_bootstraps=1000):
    """run a differential test using bootstraps of background set
    """
    # background set gets pulled out once
    # the num tasks must match the ones used for inference (inference_targets)
    # need to get: targets, indices
    with h5py.File(background_h5_file, "r") as hf:
        background_targets = hf[background_targets_key][:]
        background_gc = hf[gc_key][:]
        background_hits = (hf[pwm_hits_key][:] != 0).astype(int)
    if len(background_targets_indices) != 0:
        background_targets = background_targets[:,background_targets_indices]
        
    # get foreground from h5
    with h5py.File(foreground_h5_file, "r") as hf:
        foreground_targets = hf[foreground_targets_key][:]
        foreground_gc = hf[gc_key][:]
        foreground_hits = (hf[pwm_hits_key][:] != 0).astype(int)
        pwm_names = hf[pwm_hits_key].attrs[pwm_names_key]
    if len(foreground_targets_indices) != 0:
        foreground_targets = foreground_targets[:,foreground_targets_indices]

    assert foreground_hits.shape[1] == background_targets.shape[1]
    
    # and iterate through for each set
    raw_pvals = np.ones([foreground_targets.shape[1]] + list(foreground_hits.shape[1:]))
    positive_hits_summary = np.zeros(raw_pvals.shape)
    for foreground_idx in xrange(foreground_targets.shape[1]):
        print "Running foreground {}".format(foreground_idx)
        
        # get foreground and foreground GC
        current_foreground_indices = np.where(
            foreground_targets[:,foreground_idx] > 0)[0]
        current_foreground_hits = foreground_hits[current_foreground_indices]
        current_foreground_gc = foreground_gc[current_foreground_indices]
        
        # now here need to drop down another level to get different backgrounds
        # per cell state
        for task_idx in xrange(current_foreground_hits.shape[1]):
            print "Running task {}".format(task_idx)
            
            # get subset
            task_foreground_hits = current_foreground_hits[:,task_idx] # {N, M}
            
            # set up background set
            task_background_indices = np.where(
                background_targets[:,task_idx] > 0)[0]
            task_background_hits = background_hits[task_background_indices][:,task_idx] # {N, M}
            task_background_gc = background_gc[task_background_indices]
            
            # set up gc matched bins
            background_gc_bins = build_gc_matched_bins(
                current_foreground_gc,
                task_background_gc,
                increment=gc_increment)
            num_per_bin = int(task_foreground_hits.shape[0] * gc_increment)
            
            # set up boostrapped background sets
            matched_backgrounds = []
            for i in xrange(num_bootstraps):
                background_indices = build_gc_matched_background(
                    background_gc_bins,
                    num_per_bin,
                    rand_state=RandomState(i))
                matched_background = task_background_hits[background_indices] # {N, M}
                matched_background = np.mean(matched_background, axis=0) # {M}
                matched_backgrounds.append(matched_background) # list of {M}
            matched_backgrounds = np.stack(matched_backgrounds, axis=0) # {bootstraps, M}
            
            # subtract from each other
            score_minus_background = np.subtract(
                np.mean(np.expand_dims(task_foreground_hits, axis=0), axis=1),
                matched_backgrounds)

            # and then determine how often the diff is 0 or less
            raw_pvals[foreground_idx, task_idx] = np.mean(score_minus_background <= 0, axis=0)
            
    # and save the pvals to the h5 file
    with h5py.File(foreground_h5_file, "a") as hf:
        out_key = "{}/{}".format(DataKeys.PWM_PVALS, foreground_targets_key)
        if hf.get(out_key) is not None:
            del hf[out_key]
        hf.create_dataset(out_key, data=raw_pvals)
        hf[out_key].attrs[AttrKeys.TASK_INDICES] = foreground_targets_indices
        hf[DataKeys.PWM_PVALS].attrs[AttrKeys.PWM_NAMES] = pwm_names
    
    return raw_pvals



def threshold_and_save_pwms(pvals, pwm_list, out_file, pval_thresh=0.05, qval_thresh=0.05):
    """assumes you're taking in a reduced set, produces ONE pwm file
    """
    pass_qval_thresh = threshold_by_qvalues(
        pvals, qval_thresh=qval_thresh, num_bins=50)
    pass_qval_thresh = np.any(pass_qval_thresh, axis=0) # {M}

    import ipdb
    ipdb.set_trace()
    
    for i in xrange(len(pwm_list)):
        if pass_qval_thresh[i]:
            pwm_list[i].to_motif_file(out_file)
    
    return None



def select_pwms_by_permutation_test_and_reduce(
        array,
        pwm_list,
        hclust,
        num_shuffles=1000,
        pval_thresh=0.01):
    """use permutation test to get sig pwms
    and then reduce by similarity
    """
    # select features
    sig_pwms = select_features_by_permutation_test(
        array,
        num_shuffles=num_shuffles,
        pval_thresh=pval_thresh)

    # and then reduce
    sig_pwms = np.multiply(
        sig_pwms,
        MotifSetManager.reduce_pwms(
            pwm_list,
            hclust,
            np.median(array, axis=0)))

    # debug
    print "final pwm count:", np.sum(sig_pwms)
    indices = np.where(sig_pwms > 0)[0].tolist()
    print [pwm_list[k].name for k in indices]
    
    return sig_pwms


# TODO move to... stats?
def aggregate_array(
        array,
        agg_fn=np.sum, #np.mean,
        agg_axis=0,
        mask=None):
    """aggregate dataset
    """
    agg_array = agg_fn(array, axis=agg_axis) # {task, M}
    agg_array = agg_array[:,mask>0]
    
    return agg_array


def select_task_pwms(array, pwm_list, hclust):
    """wrapper to make it easy to work with task dimension
    """
    assert len(array.shape) == 3

    sig_pwms = np.zeros((array.shape[2]))
    for task_idx in xrange(array.shape[1]):
        task_data = array[:,task_idx,:]
        sig_pwms += select_pwms_by_permutation_test_and_reduce(
            task_data, pwm_list, hclust)
        
    return sig_pwms


def refine_sig_pwm_clusters(clusters):
    """adjust clusters
    """
    # TODO if needed

    

    return


def extract_significant_pwms(
        h5_file,
        pwm_list,
        data_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        clusters_key=DataKeys.CLUSTERS,
        pwm_names_attr_key=AttrKeys.PWM_NAMES,
        pwm_sig_global_key=DataKeys.PWM_SIG_GLOBAL,
        pwm_scores_agg_global_key=DataKeys.PWM_SCORES_AGG_GLOBAL,
        pwm_sig_clusters_key=DataKeys.PWM_SIG_CLUST,
        pwm_sig_clusters_all_key=DataKeys.PWM_SIG_CLUST_ALL,
        pwm_scores_agg_clusters_key=DataKeys.PWM_SCORES_AGG_CLUST,
        refine_clusters=False,
        num_threads=24):
    """
    """
    # get a hierarchical clustering on pwms by sequence distance
    cor_filt_mat, distances = MotifSetManager.correlate_pwms_by_seq_similarity(
        pwm_list, cor_thresh=0.3, ncor_thresh=0.2, num_threads=num_threads)
    hclust = MotifSetManager.hclust(distances)

    # outputs
    outputs = {}
    
    # pull out data
    with h5py.File(h5_file, "r") as hf:
        data = hf[data_key][:] # {N, task, M}
        pwm_names = hf[data_key].attrs[pwm_names_attr_key]

    # (1) get global sig (and aggregate results)
    # TODO when stable use multiprocessing Pool
    sig_pwms = select_task_pwms(data, pwm_list, hclust)
    agg_data = aggregate_array(data, mask=sig_pwms)

    # save out
    pwm_sig_global_names = pwm_names[sig_pwms > 0]
    outputs[pwm_sig_global_key] = (sig_pwms, pwm_sig_global_names)
    outputs[pwm_scores_agg_global_key] = (agg_data, pwm_sig_global_names)
    print "globally significant:", pwm_sig_global_names
    
    # (2) get cluster sig (and aggregate results)
    clusters = get_clusters_from_h5(h5_file, clusters_key)
    num_clusters = len(clusters.get_active_cluster_ids())
    sig_pwms = np.zeros((num_clusters, data.shape[2]))
    generator = clusters.cluster_mask_generator()
    cluster_idx = 0
    for cluster_id, cluster_mask in generator:
        print "cluster_id: {}".format(cluster_id)
        print "num examples: {}".format(np.sum(cluster_mask))
            
        # select
        cluster_data = data[np.where(cluster_mask)[0],:,:] # {N, task, M}
        cluster_pwms = np.zeros((data.shape[2]))

        # get sig and aggregate results
        sig_pwms[cluster_idx,:] = select_task_pwms(cluster_data, pwm_list, hclust)
        pwm_sig_cluster_names = pwm_names[sig_pwms[cluster_idx,:] > 0]
        print pwm_sig_cluster_names
        cluster_idx += 1
        
    # adjust clustering as needed
    # remove clusters with no sig motifs?
    if refine_clusters:
        pass

    # save out
    pwm_sig_clusters_all = np.any(sig_pwms > 0, axis=0).astype(int) # {M}
    pwm_sig_cluster_global_names = pwm_names[pwm_sig_clusters_all > 0]
    outputs[pwm_sig_clusters_all_key] = (
        pwm_sig_clusters_all,
        pwm_sig_cluster_global_names)
    print pwm_sig_cluster_global_names
    
    # and aggregate {cluster, task, M}
    agg_data = np.zeros((
        num_clusters,
        data.shape[1],
        np.sum(outputs[pwm_sig_clusters_all_key][0])))
    generator = clusters.cluster_mask_generator()
    cluster_idx = 0
    
    for cluster_id, cluster_mask in generator:
        cluster_data = data[np.where(cluster_mask)[0],:,:]
        agg_data[cluster_idx,:,:] = aggregate_array(
            cluster_data, mask=outputs[pwm_sig_clusters_all_key][0])
        cluster_idx += 1
        
    # save out
    outputs[pwm_sig_clusters_key] = (sig_pwms, pwm_sig_cluster_global_names)
    outputs[pwm_scores_agg_clusters_key] = (agg_data, pwm_sig_cluster_global_names)
    
    # and then save all of this out
    with h5py.File(h5_file, "a") as out:
        for key in outputs.keys():
            if out.get(key) is not None:
                del out[key]
            out.create_dataset(key, data=outputs[key][0])
            out[key].attrs[AttrKeys.PWM_NAMES] = outputs[key][1]
    
    return None


def visualize_significant_pwms_R(
        h5_file,
        pwm_scores_agg_clusters_key=DataKeys.PWM_SCORES_AGG_CLUST,
        pwm_names_attr_key=AttrKeys.PWM_NAMES):
    """plot out the pwm maps
    """
    r_cmd = (
        "plot-h5.sig_pwms.R {} {} {}").format(
            h5_file,
            pwm_scores_agg_clusters_key,
            pwm_names_attr_key)
    logging.info(r_cmd)
    os.system(r_cmd)
    
    return None
