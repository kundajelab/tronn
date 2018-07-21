# description: functions around clustering (ie unsupervised learning)

import os
import h5py
import logging
import phenograph

import numpy as np

from collections import Counter

from tronn.stats.distances import score_distances
from tronn.stats.thresholding import threshold_at_recall
from tronn.stats.thresholding import threshold_at_fdr

from tronn.util.utils import DataKeys
from tronn.util.h5_utils import AttrKeys


def run_clustering(
        h5_file,
        dataset_key,
        cluster_key=DataKeys.CLUST_ROOT,
        cluster_filt_key=DataKeys.CLUST_FILT,
        num_threads=24):
    """wrapper around favorite clustering method
    """
    # get data
    with h5py.File(h5_file, "r") as hf:
        data = hf[dataset_key][:]

    # ----------------------------------------------
    # YOUR FAVORITE CLUSTERING METHOD HERE
    # Requires: return a {N} size array with numerical
    # clusters, -1 if example is not clustered

    data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
    communities, graph, Q = phenograph.cluster(
        data, n_jobs=num_threads)
    null_cluster_present = True
    
    # END CLUSTERING
    # ----------------------------------------------
    
    # save clusters
    with h5py.File(h5_file, "a") as out:
        out.create_dataset(cluster_key, data=communities)

    # refine as desired
    if cluster_filt_key is not None:
        remove_small_clusters(
            h5_file,
            cluster_key=cluster_key,
            cluster_filt_key=cluster_filt_key,
            null_cluster_present=null_cluster_present)
    
    return None


def remove_small_clusters(
        h5_file,
        cluster_key=DataKeys.CLUST_ROOT,
        cluster_filt_key=DataKeys.CLUST_FILT,
        null_cluster_present=False,
        fractional_threshold=0.01):
    """remove small clusters
    """
    # get clusters out
    with h5py.File(h5_file, "r") as hf:
        clusters = hf[cluster_key][:]

    # get num examples, adjust if null cluster is present
    if null_cluster_present:
        num_clustered_examples = clusters[clusters != -1].shape[0]
    else:
        num_clustered_examples = clusters.shape[0]
        
    # determine counts
    counts = Counter(clusters.tolist())

    # adjust based on fractional thresh and renumber based on size
    cluster_ids_and_counts = counts.most_common()
    for cluster_idx in xrange(len(cluster_ids_and_counts)):
        cluster_id, count = cluster_ids_and_counts[cluster_idx]

        if float(count) / num_clustered_examples < fractional_threshold:
            clusters[clusters==cluster_id] = -1
        else:
            clusters[clusters==cluster_id] = cluster_idx

    # backcheck TODO

    # save
    with h5py.File(h5_file, "a") as out:
        out.create_dataset(cluster_filt_key, data=clusters)
    
    return None


def get_cluster_bed_files(
        h5_file,
        clusters_key=DataKeys.CLUST_FILT):
    """generate BED files from metadata
    """
    prefix = h5_file.split(".h5")[0]

    # get clusters
    with h5py.File(h5_file, "r") as hf:
        clusters = hf[cluster_key][:,0]

    # get cluster ids
    if len(clusters.shape) == 1:
        cluster_ids = sorted(list(set(clusters.tolist())))
        hard_clusters = True
    else:
        cluster_ids = range(clusters.shape[1])
        hard_clusters = False
        
    # for each cluster make a bed
    for cluster_idx in xrange(len(cluster_ids)):
        cluster_id = cluster_ids[cluster_idx]
        
        if hard_clusters:
            in_cluster = clusters == cluster_id
        else:
            in_cluster = clusters[:,cluster_idx] > 0
    
        with h5py.File(h5_file, "r") as hf:
            metadata = hf["example_metadata"][:][in_cluster]
            
        cluster_prefix = "{0}.cluster-{1}".format(prefix, cluster_id)
        metadata_file = "{}.metadata.txt".format(cluster_prefix)
        metadata_bed = "{}.bed".format(cluster_prefix)
        # TODO rewrite this function
        #make_bed(metadata, metadata_file, metadata_bed)
        print "DOES NOT WORK YET"
        
    return


def refine_manifold(
        clusters,
        centers,
        thresholds,
        cluster_ids,
        min_count=0):
    """given manifold metrics, remove empty clusters
    """
    # look at (soft) clusters
    keep_clusters = np.ones((clusters.shape[1]))
    for cluster_idx in xrange(clusters.shape[1]):

        if np.sum(clusters[:,cluster_idx]) <= min_count:
            # empty cluster (or too small) - mark for removal
            keep_clusters[cluster_idx] = 0

    # remove
    clusters = clusters[:,keep_clusters==1]
    centers = centers[keep_clusters==1,:,:]
    thresholds = thresholds[keep_clusters==1,:]
    cluster_ids = np.array(cluster_ids)[keep_clusters==1].tolist()
    
    return clusters, centers, thresholds, cluster_ids


# TODO - make one big manifold (flatten) and go from there
def summarize_clusters_on_manifold(
        h5_file,
        features_key=DataKeys.FEATURES,
        cluster_key=DataKeys.CLUST_FILT,
        centers_key=DataKeys.MANIFOLD_CENTERS,
        thresholds_key=DataKeys.MANIFOLD_THRESHOLDS,
        manifold_clusters_key=DataKeys.MANIFOLD_CLUST,
        fdr=0.50,
        recall_thresh=0.25,
        refine_clusters=True):
    """extract metrics on the pwm manifold
    """
    # get clusters and data
    with h5py.File(h5_file, "r") as hf:
        data = hf[features_key][:] # {N, task, M}
        clusters = hf[cluster_key][:] # {N}
    cluster_ids = sorted(list(set(clusters.tolist())))
    if -1 in cluster_ids:
        cluster_ids.remove(-1)
    print cluster_ids
    
    # set up save arrays as dict
    out_shape = (len(cluster_ids), data.shape[1], data.shape[2])
    out_arrays = {}
    out_arrays[centers_key] = np.zeros(out_shape)
    out_arrays[thresholds_key] = np.zeros((len(cluster_ids), data.shape[1]))
    out_arrays[manifold_clusters_key] = np.ones((data.shape[0], len(cluster_ids)))
    
    # then calculate metrics on each cluster separately
    for cluster_idx in xrange(len(cluster_ids)):
        cluster_id = cluster_ids[cluster_idx]
        print "cluster_id: {}".format(cluster_id),
        in_cluster = clusters == cluster_id # chekc this
            
        # get the data in this cluster and remove zeros (if any)
        scores_in_cluster = data[np.where(in_cluster)[0],:,:] # {N, task, M}
        print "total examples: {}".format(scores_in_cluster.shape[0]),
        scores_in_cluster = scores_in_cluster[np.max(scores_in_cluster, axis=(1,2))>0]
        print "after remove zeros: {}".format(scores_in_cluster.shape[0])
        
        # calculate cluster center using median
        out_arrays[centers_key][cluster_idx,:,:] = np.median(
            scores_in_cluster, axis=0) # {task, M}
        
        # calculate thresholds for each task separately
        # TODO - do this as pool when stable
        # TODO - consider making tasks into 1 manifold and traversing that
        for task_idx in xrange(data.shape[1]):
            print "task: {}; ".format(task_idx),

            # score distances
            distances = score_distances(
                data[:,task_idx,:],
                out_arrays[centers_key][cluster_idx,task_idx,:],
                method="jaccard")
            
            # get threshold
            # TODO adjust this to grab back threshold for top 100?
            threshold = threshold_at_recall(
                in_cluster,
                distances,
                recall_thresh=recall_thresh)

            threshold = threshold_at_fdr(
                in_cluster,
                distances,
                fdr=fdr)
            
            print "Threshold for when recall is {} (ie TPR): {};".format(recall_thresh, threshold),
            passed_threshold = distances > threshold # CHANGE IF METRIC CHANGES (less than)
            
            print "passing filter: {};".format(np.sum(passed_threshold)),
            print "true positives: {}".format(np.sum(passed_threshold * in_cluster))
            # NOTE: right now must pass at least ALL task thresholds
            # is there a better way to do this?
            out_arrays[thresholds_key][cluster_idx,task_idx] = threshold
            out_arrays[manifold_clusters_key][:,cluster_idx] *= passed_threshold

        passed_threshold_final = out_arrays[manifold_clusters_key][:,cluster_idx]
        print "final passing filter: {};".format(np.sum(passed_threshold_final)),
        print "final true positives: {}".format(np.sum(passed_threshold_final * in_cluster))

    if refine_clusters:
        clusters, centers, thresholds, cluster_ids = refine_manifold(
            out_arrays[manifold_clusters_key],
            out_arrays[centers_key],
            out_arrays[thresholds_key],
            cluster_ids,
            min_count=50)
    
        out_arrays[manifold_clusters_key] = clusters
        out_arrays[centers_key] = centers
        out_arrays[thresholds_key] = thresholds
    
    # save out
    with h5py.File(h5_file, "a") as out:
        for key in out_arrays.keys():
            del out[key]
            out.create_dataset(key, data=out_arrays[key])
            out[key].attrs[AttrKeys.CLUSTER_IDS] = cluster_ids
            
    return None



def visualize_clustering_results_R(
        h5_file, cluster_key):
    """use R to visualize results
    """
    

    
    return
