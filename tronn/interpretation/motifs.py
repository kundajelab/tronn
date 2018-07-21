"""Description: code to analyze motifs
"""

import h5py

import numpy as np

from tronn.stats.nonparametric import select_features_by_permutation_test

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
        data_key=DataKeys.FEATURES,
        cluster_key=DataKeys.CLUST_FILT,
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
    with h5py.File(h5_file, "r") as hf:
        clusters = hf[cluster_key][:]

    if len(clusters.shape) == 1:
        # hard clusters
        cluster_ids = sorted(list(set(clusters.tolist())))
        if -1 in cluster_ids: cluster_ids.remove(-1)
        hard_clusters = True
    else:
        # soft clusters
        cluster_ids = range(clusters.shape[1])
        hard_clusters = False

    # go through clusters to get sig pwms
    sig_pwms = np.zeros((len(cluster_ids), data.shape[2]))
    for cluster_idx in xrange(len(cluster_ids)):
        print cluster_idx,
        cluster_id = cluster_ids[cluster_idx]

        # get which examples are in cluster
        if hard_clusters:
            in_cluster = clusters == cluster_id
        else:
            in_cluster = clusters[:,cluster_id] == 1
        print "num examples: {}".format(np.sum(in_cluster))
            
        # select
        cluster_data = data[np.where(in_cluster)[0],:,:] # {N, task, M}
        cluster_pwms = np.zeros((data.shape[2]))

        # get sig and aggregate results
        sig_pwms[cluster_idx,:] = select_task_pwms(cluster_data, pwm_list, hclust)
        pwm_sig_cluster_names = pwm_names[sig_pwms[cluster_idx,:] > 0]
        print pwm_sig_cluster_names
    
    # adjust clustering as needed
    if refine_clusters:
        pass

    # save out
    outputs[pwm_sig_clusters_all_key] = np.any(sig_pwms > 0, axis=0) # {M}
    pwm_sig_cluster_global_names = pwm_names[outputs[pwm_sig_clusters_all_key] > 0]
    print pwm_sig_cluster_global_names
    
    # and aggregate {cluster, task, M}
    agg_data = np.zeros((
        len(cluster_ids),
        data.shape[1],
        np.sum(outputs[pwm_sig_clusters_all_key])))
    for cluster_idx in xrange(len(cluster_ids)):
        print cluster_idx
        cluster_id = cluster_ids[cluster_idx]

        # get which examples are in cluster
        if hard_clusters:
            in_cluster = clusters == cluster_id
        else:
            in_cluster = clusters[:,cluster_id] == 1

        cluster_data = data[np.where(in_cluster)[0],:,:]
        agg_data[cluster_idx,:,:] = aggregate_array(
            cluster_data, mask=outputs[pwm_sig_clusters_all_key])
    
    # save out
    outputs[pwm_sig_clusters_key] = (sig_pwms, pwm_sig_cluster_global_names)
    outputs[pwm_scores_agg_clusters_key] = (agg_data, pwm_sig_cluster_global_names)
    
    # and then save all of this out
    with h5py.File(h5_file, "a") as out:
        for key in outputs.keys():
            del out[key]
            out.create_dataset(key, data=outputs[key][0])
            out[key].attrs[AttrKeys.PWM_NAMES] = outputs[key][1]
    
    return None
