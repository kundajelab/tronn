"""Description: code to analyze motifs
"""

import os
import h5py

import numpy as np

from tronn.stats.nonparametric import select_features_by_permutation_test

from tronn.interpretation.clustering import get_clusters_from_h5

from tronn.util.pwms import MotifSetManager
from tronn.util.utils import DataKeys
from tronn.util.h5_utils import AttrKeys


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
        agg_fn=np.median,
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
            #del out[key]
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
    print r_cmd
    os.system(r_cmd)
    
    return None
