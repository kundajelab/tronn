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
    num_increments = 1. / increment
    sorted_gc = np.sort(gc_array) # use this for intervals
    intervals = np.interp(
        np.linspace(0, len(sorted_gc), num=int(num_increments)+1),
        range(len(sorted_gc)),
        sorted_gc)
    
    # go through bins and collect sequences
    binned_indices = []
    for i in xrange(len(intervals)-1):
        range_min = intervals[i]
        range_max = intervals[i+1]
        background_indices = np.where(
            (background_gc_array >= range_min) &
            (background_gc_array < range_max))[0]
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


def _boostrap_gc_matched_backgrounds(
        background_hits,
        foreground_gc,
        background_gc,
        gc_increment=0.05,
        num_bootstraps=1000):
    """collect background sets that are GC matched, of the 
    desired sample size
    """
    # set up GC matched bins
    background_gc_bins = build_gc_matched_bins(
        foreground_gc, background_gc, increment=gc_increment)
    num_per_bin = int(foreground_gc.shape[0] * gc_increment)

    # bootstrap
    matched_backgrounds = []
    for i in xrange(num_bootstraps):
        background_indices = build_gc_matched_background(
            background_gc_bins,
            num_per_bin,
            rand_state=RandomState(i))
        matched_background = background_hits[background_indices] # {N, ...}
        matched_background = np.mean(matched_background, axis=0) # {...}
        matched_backgrounds.append(matched_background) # list of {...}
    matched_backgrounds = np.stack(matched_backgrounds, axis=0) # {bootstraps, ...}

    return matched_backgrounds


def test_differential_motifs(
        foreground_scores,
        background_scores,
        foreground_gc,
        background_gc,
        gc_increment=0.05,
        num_bootstraps=1000,
        qval_thresh=0.05,
        reduce_sig_type="any",
        out_key=DataKeys.PWM_DIFF_GROUP):
    """run a differential test using bootstraps of background set
    assumes axis 1 is different tasks, and will adjust to make
    sure background is positives for each task
    """
    # get GC-matched background sets
    matched_backgrounds = _boostrap_gc_matched_backgrounds(
        background_scores,
        foreground_gc,
        background_gc,
        gc_increment=gc_increment,
        num_bootstraps=1000)
    
    # subtract from each other
    score_minus_background = np.subtract(
        np.mean(np.expand_dims(foreground_scores, axis=0), axis=1),
        matched_backgrounds)
    
    # and then determine how often the diff is 0 or less
    pvals = np.mean(score_minus_background <= 0, axis=0)
    
    return pvals


def get_sig_pwm_vector(
        h5_file,
        sig_key,
        foregrounds,
        reduce_type="any"):
    """convenience function to quickly extract a sig pwm vector
    """
    with h5py.File(h5_file, "r") as hf:

        # get indices of foregrounds to keep
        existing_foregrounds = hf[sig_key].attrs["foregrounds"]
        foregrounds_keys = hf[sig_key].attrs["foregrounds.keys"]
        keep_indices = []
        for i in range(len(existing_foregrounds)):
            if existing_foregrounds[i] in foregrounds:
                keep_indices.append(i)
        
        # add them up
        foreground_exists = False
        for i in range(len(keep_indices)):
            foreground_idx = keep_indices[i]
            foreground_key = foregrounds_keys[foreground_idx]
            foreground_key = "{}/{}/sig".format(sig_key, foreground_key)
            logging.info("loading from {}".format(foreground_key))
            index_sig_pwms = hf[foreground_key][:]

            if i == 0:
                sig_pwms = np.zeros((
                    len(foregrounds),
                    index_sig_pwms.shape[0]))
                sig_pwms[i] = index_sig_pwms
                foreground_exists = True
            else:
                sig_pwms[i] = index_sig_pwms

        assert foreground_exists

    # reduce
    if reduce_type == "any":
        sig_pwms = np.any(sig_pwms != 0,axis=0) # {M}
    else:
        raise ValueError, "reduce type requested is not implemented"

    return sig_pwms


def copy_sig_pwm_vectors_to_h5(
        old_h5_file,
        new_h5_file,
        sig_key,
        target_key,
        target_indices):
    """convenience function to copy pwm vectors
    """
    group_key = "{}/{}".format(sig_key, target_key)
    with h5py.File(old_h5_file, "r") as hf:
        for i in xrange(len(target_indices)):
            index_key = "{}/{}/{}".format(
                group_key, target_indices[i], DataKeys.PWM_SIG_ROOT)
            with h5py.File(new_h5_file, "a") as out:
                out.create_dataset(index_key, data=hf[index_key][:])

    return None










# OLD BELOW


# TODO deprecate
def select_pwms_by_permutation_test_and_reduce_OLD(
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
# TODO - deprecate?
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


# TODO consider deprecating this
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

