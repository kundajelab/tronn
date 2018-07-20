# description: functions around clustering (ie unsupervised learning)

import os
import h5py
import logging
import phenograph

import numpy as np
import pandas as pd

from numpy.linalg import norm
from collections import Counter

# TODO move this to motifs
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from tronn.stats.distances import score_distances
from tronn.stats.thresholding import threshold_at_recall
from tronn.stats.thresholding import threshold_at_fdr

from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys


def cluster_dataset(
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
    # TODO set up for soft clustering, for now only look at hard clustering

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



# TODO deprecate
def refine_clusters(
        h5_file,
        clusters_key=DataKeys.CLUST_ROOT,
        out_key=DataKeys.CLUST_FILT,
        null_cluster_present=False,
        fractional_threshold=0.01): # TODO - adjust this fractional threshold based on number actually clustered
    """Given a clusters dataset, remove small clusters 
    and save out to a new dataset
    """
    with h5py.File(h5_file, "a") as hf:
        num_examples = hf["example_metadata"].shape[0]

        # generate a new dataset that is {N, 1}
        #del hf[out_key]
        refined_clusters_hf = hf.create_dataset(
            out_key, hf[clusters_key].shape, dtype=int)
        
        # then for each column, refine
        for i in xrange(hf[clusters_key].shape[1]):
            clusters = hf[clusters_key][:,i]
            new_idx = len(list(set(clusters.tolist()))) # this is the null cluster
            
            new_clusters = np.zeros(clusters.shape)
            counts = Counter(clusters.tolist())
            print counts.most_common(10)
            
            # adjust denominator for fractional threshold based on if
            # null cluster exists
            if null_cluster_present:
                clusters_uniq = list(set(clusters.tolist()))
                max_id = max(clusters_uniq)
                denominator = clusters[clusters != max_id].shape[0]
                print "here"
            else:
                denominator = num_examples
                
            # for each cluster, check size and change as necessary
            cluster_ids = counts.most_common()
            for j in xrange(len(cluster_ids)):
                cluster_id, count = cluster_ids[j]
                if float(count) / denominator < fractional_threshold:
                    # replace cluster id with new index
                    new_clusters[clusters==cluster_id] = new_idx
                else:
                    # keep as current cluster
                    new_clusters[clusters==cluster_id] = j
                    
            # back check
            print "reduced num clusters:", len(list(set(new_clusters.tolist())))
            print set(new_clusters.tolist())
                    
            # save into dataset
            refined_clusters_hf[:,i] = new_clusters
                    
    return None


# TODO deprecate
def generate_simple_metaclusters(
        h5_file, dataset_keys, out_key, num_threads=24):
    """Given datasets, bind them together to 
    make a giant matrix, cluster, and save out cluster info
    """
    with h5py.File(h5_file, "a") as hf:
        num_examples = hf["example_metadata"].shape[0]
        
        # set up out cluster
        clusters_hf = hf.create_dataset(
            out_key, (num_examples, 1), dtype=int)
        
        # extract all the data and merge together
        for i in xrange(len(dataset_keys)):
            dataset_tmp = pd.DataFrame(hf[dataset_keys[i]][:])
            
            if i == 0:
                dataset = dataset_tmp
            else:
                dataset = pd.concat([
                    dataset.reset_index(drop=True),
                    dataset_tmp], axis=1)
                
        # cluster with louvain
        communities, graph, Q = phenograph.cluster(
            dataset, n_jobs=num_threads)
        clusters_hf[:,0] = communities
    
    return None



# check redundancy
def get_distance_matrix(
        motif_mat,
        corr_method="pearson",
        pval_thresh=0.05,
        corr_min=0.4):
    """Given a matrix, calculate a distance metric for each pair of 
    columns and look at p-val. If p-val is above threshold, keep 
    otherwise leave as 0.
    """
    num_columns = motif_mat.shape[1]
    correlation_vals = np.zeros((num_columns, num_columns))
    correlation_pvals = np.zeros((num_columns, num_columns))
    for i in xrange(num_columns):
        for j in xrange(num_columns):

            if corr_method == "pearson":
                cor_val, pval = pearsonr(motif_mat[:,i], motif_mat[:,j])
                if pval < pval_thresh and cor_val > corr_min:
                    correlation_vals[i,j] = cor_val
                    correlation_pvals[i,j] = pval
            elif corr_method == "continuous_jaccard":
                min_vals = np.minimum(motif_mat[:,i], motif_mat[:,j])
                max_vals = np.maximum(motif_mat[:,i], motif_mat[:,j])
                if np.sum(max_vals) != 0:
                    similarity = np.sum(min_vals) / np.sum(max_vals)
                elif i == j:
                    similarity = 1.0
                else:
                    similarity = 0.0
                correlation_vals[i,j] = similarity
                correlation_pvals[i,j] = similarity
            elif corr_method == "intersection_size":
                intersect = np.sum(
                    np.logical_and(motif_mat[:,i] > 0, motif_mat[:,j] > 0))
                #intersect_fract = float(intersect) / motif_mat.shape[0]
                intersect_fract = intersect
                correlation_vals[i,j] = intersect_fract
                correlation_pvals[i,j] = intersect_fract
                
    return correlation_vals, correlation_pvals



# check redundancy
def get_correlation_file(
        mat_file,
        corr_file,
        corr_method="intersection_size", # continuous_jaccard, pearson
        corr_min=0.4,
        pval_thresh=0.05):
    """Given a matrix file, calculate correlations across the columns
    """
    mat_df = pd.read_table(mat_file, index_col=0)
    mat_df = mat_df.drop(["community"], axis=1)
            
    corr_mat, pval_mat = get_significant_correlations(
        mat_df.as_matrix(),
        corr_method=corr_method,
        corr_min=corr_min,
        pval_thresh=pval_thresh)
    
    corr_mat_df = pd.DataFrame(corr_mat, index=mat_df.columns, columns=mat_df.columns)
    corr_mat_df.to_csv(corr_file, sep="\t")

    return None



def rownorm2d(array):
    """generic row normalization
    """
    max_vals = np.max(array, axis=1, keepdims=True)
    array_norm = np.divide(
        array,
        max_vals,
        out=np.zeros_like(array),
        where=max_vals!=0)
    
    return array_norm

# rename - characterize_clusters_in_manifold
def get_manifold_metrics(
        h5_file,
        manifold_h5_file,
        features_key=DataKeys.FEATURES,
        cluster_key=DataKeys.CLUST_FILT,
        recall_thresh=0.25):
    """extract metrics on the pwm manifold
    """
    # TODO move this somewhere else
    #from tronn.interpretation.motifs import get_individual_pwm_thresholds

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
    out_arrays[DataKeys.MANIFOLD_CENTERS] = np.zeros(out_shape)
    out_arrays[DataKeys.MANIFOLD_THRESHOLDS] = np.zeros((len(cluster_ids), data.shape[1]))
    out_arrays[DataKeys.MANIFOLD_CLUST] = np.ones((data.shape[0], len(cluster_ids)))
    
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
        out_arrays[DataKeys.MANIFOLD_CENTERS][cluster_idx,:,:] = np.median(
            scores_in_cluster, axis=0) # {task, M}
        
        # calculate thresholds for each task separately
        # TODO - do this as pool?
        for task_idx in xrange(data.shape[1]):
            print "task: {}; ".format(task_idx),

            # score distances
            distances = score_distances(
                data[:,task_idx,:],
                out_arrays[DataKeys.MANIFOLD_CENTERS][cluster_idx,task_idx,:],
                method="jaccard")
            
            # get threshold
            # TODO adjust this to grab back threshold for top 100?
            threshold = threshold_at_recall(
                in_cluster,
                distances,
                recall_thresh=recall_thresh)

            fdr = 0.50
            threshold = threshold_at_fdr(
                in_cluster,
                distances,
                fdr=fdr)



            
            print "Threshold for when recall is {} (ie TPR): {};".format(recall_thresh, threshold),
            passed_threshold = distances > threshold # CHANGE IF METRIC CHANGES (less than)
            
            print "passing filter: {};".format(np.sum(passed_threshold)),
            print "true positives: {}".format(np.sum(passed_threshold * in_cluster))
            # NOTE: right now must pass ALL task thresholds
            out_arrays[DataKeys.MANIFOLD_THRESHOLDS][cluster_idx,task_idx] = threshold
            out_arrays[DataKeys.MANIFOLD_CLUST][:,cluster_idx] *= passed_threshold

        passed_threshold_final = out_arrays[DataKeys.MANIFOLD_CLUST][:,cluster_idx]
        print "final passing filter: {};".format(np.sum(passed_threshold_final)),
        print "final true positives: {}".format(np.sum(passed_threshold_final * in_cluster))

    import ipdb
    ipdb.set_trace()

    # and save all this out to h5
    with h5py.File(manifold_h5_file, "w") as out:
        for key in out_arrays.keys():
            out.create_dataset(key, data=out_arrays[key])
            out[key].attrs[AttrKeys.CLUSTER_IDS] = cluster_ids

    quit()
    # and back to original file?
    with h5py.File(h5_file, "a") as out:
        for key in out_arrays.keys():
            out.create_dataset(key, data=out_arrays[key])
            out[key].attrs[AttrKeys.CLUSTER_IDS] = cluster_ids


            
    return None


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



def aggregate_pwm_results(
        results_h5_file,
        dataset_keys,
        agg_key,
        manifold_h5_file):
    """creates a task x pwm file
    """
    # get the master pwm vector from manifold file
    with h5py.File(manifold_h5_file, "r") as hf:
        master_pwm_vector = hf["master_pwm_vector"][:]

    # extract those motif scores
    with h5py.File(results_h5_file, "a") as hf:
        num_motifs = hf[dataset_keys[0]].shape[1]
        tasks_x_pwm = np.zeros((len(dataset_keys), num_motifs))
        pwm_names = hf[dataset_keys[0]].attrs["pwm_names"]
        
        for task_i in xrange(len(dataset_keys)):

            # get data
            data = hf[dataset_keys[task_i]][:]
            
            # row normalize and remove zeroes
            max_vals = np.max(data, axis=1, keepdims=True)
            data_norm = np.divide(
                data,
                max_vals,
                out=np.zeros_like(data),
                where=max_vals!=0)
            data_norm = data_norm[np.max(data_norm, axis=1)>0]

            # sum
            tasks_x_pwm[task_i,:] = np.sum(data_norm, axis=0)
            
        # if needed, filter for master pwm vector
        tasks_x_pwm = tasks_x_pwm[:,master_pwm_vector > 0]
        pwm_names = pwm_names[master_pwm_vector > 0]
            
        # and save out
        #del hf[agg_key]
        hf.create_dataset(agg_key, data=tasks_x_pwm)
        hf[agg_key].attrs["pwm_names"] = pwm_names
        
    return None


# TODO can use this for dmim too - rename?
def aggregate_pwm_results_per_cluster(
        results_h5_file,
        cluster_key,
        dataset_keys,
        agg_key,
        manifold_h5_file,
        cluster_col=0,
        remove_final_cluster=True,
        soft_clustering=False):
    """creates a task x pwm dataset
    """
    # get the master pwm vector from manifold file
    with h5py.File(manifold_h5_file, "r") as hf:
        master_pwm_vector = hf["master_pwm_vector"][:]
        
    # extract those motif scores at a cluster level
    with h5py.File(results_h5_file, "a") as hf:
        num_motifs = hf[dataset_keys[0]].shape[1]
        pwm_names = hf[dataset_keys[0]].attrs["pwm_names"]

        # cluster ids
        if not soft_clustering:
            clusters = hf[cluster_key][:,cluster_col]
            cluster_ids = sorted(list(set(clusters.tolist())))
        else:
            clusters = hf[cluster_key][:]
            cluster_ids = range(hf[cluster_key].shape[1])

        # remove final id
        if remove_final_cluster:
            cluster_ids = cluster_ids[0:-1]

        # agg results dataset
        agg_data = np.zeros((
            len(cluster_ids),
            len(dataset_keys),
            num_motifs))
            
        for cluster_idx in xrange(len(cluster_ids)):
            cluster_id = cluster_ids[cluster_idx]
            
            for dataset_idx in xrange(len(dataset_keys)):
                
                # get data
                data = hf[dataset_keys[dataset_idx]][:]

                # TODO row normalize for now, because not calibrated
                # row normalize and remove zeroes
                #max_vals = np.max(data, axis=1, keepdims=True)
                #data_norm = np.divide(
                #    data,
                #    max_vals,
                #    out=np.zeros_like(data),
                #    where=max_vals!=0)
                data_norm = data
                
                # sum
                if not soft_clustering:
                    # actually should do median
                    agg_data[cluster_idx, dataset_idx,:] = np.sum(
                        data_norm[clusters==cluster_id], axis=0)
                    #agg_data[cluster_idx, dataset_idx,:] = np.median(
                    #    data_norm[clusters==cluster_id], axis=0)
                else:
                    agg_data[cluster_idx, dataset_idx,:] = np.sum(
                        data_norm[clusters[:,cluster_id] >= 1], axis=0)
            
        # if needed, filter for master pwm vector
        agg_data = agg_data[:,:,master_pwm_vector > 0]
        pwm_names = pwm_names[master_pwm_vector > 0]
            
        # and save out
        #del hf[agg_key]
        hf.create_dataset(agg_key, data=agg_data)
        hf[agg_key].attrs["pwm_names"] = pwm_names
        
    return None



