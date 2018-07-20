# description: functions around clustering (ie unsupervised learning)

import os
import h5py
import phenograph

import numpy as np
import pandas as pd

from numpy.linalg import norm
from collections import Counter

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from tronn.interpretation.learning import threshold_at_recall


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

# put into different module
def generalized_jaccard_similarity(a, b):
    """Calculates a jaccard on real valued vectors
    """
    assert np.all(a >= 0)
    assert np.all(b >= 0)

    # get min and max
    min_vals = np.minimum(a, b)
    max_vals = np.maximum(a, b)

    # divide sum(min) by sum(max)
    if np.sum(max_vals) != 0:
        similarity = np.sum(min_vals) / np.sum(max_vals)
    else:
        similarity = 0
        
    return similarity


def get_threshold_on_jaccard_similarity(
        mean_vector, data, labels, recall_thresh=0.40):
    """Given a mean vector and a dataset (with labels),
    get back the threshold on jaccard similarity
    """
    assert mean_vector.shape[0] == data.shape[1]
    assert data.shape[0] == labels.shape[0]

    # set up similarity scores vector
    scores = np.zeros((data.shape[0]))

    # for each example, get score
    for example_idx in xrange(data.shape[0]):
        example_vector = data[example_idx,:]
        scores[example_idx] = generalized_jaccard_similarity(
            mean_vector, example_vector)
        
    # choose threshold
    threshold = threshold_at_recall(
        labels, scores, recall_thresh=recall_thresh)
    print "Threshold for similarity: {}".format(threshold)
    threshold_filter = scores >= threshold

    return threshold, threshold_filter


def get_threshold_on_dot_product(
        mean_vector, data, labels, recall_thresh=0.40):
    """Given a mean vector and a dataset (with labels),
    get back the threshold on jaccard similarity
    """
    assert mean_vector.shape[0] == data.shape[1]
    assert data.shape[0] == labels.shape[0]

    # set up similarity scores vector
    scores = np.zeros((data.shape[0]))

    # for each example, get score
    for example_idx in xrange(data.shape[0]):
        example_vector = data[example_idx,:]
        
        scores[example_idx] = np.dot(
            mean_vector, example_vector)
        #scores[example_idx] = np.sum(np.multiply(
        #    mean_vector, example_vector))
        
    # choose threshold
    threshold = threshold_at_recall(
        labels, scores, recall_thresh=recall_thresh)
    print "Threshold for similarity: {}".format(threshold)
    threshold_filter = scores >= threshold

    return threshold, threshold_filter



def get_threshold_on_euclidean_distance(
        mean_vector, data, labels, recall_thresh=0.40):
    """Given a mean vector and a dataset (with labels),
    get back the threshold on jaccard similarity
    """
    assert mean_vector.shape[0] == data.shape[1]
    assert data.shape[0] == labels.shape[0]

    # set up similarity scores vector
    scores = np.zeros((data.shape[0]))

    # for each example, get score
    for example_idx in xrange(data.shape[0]):
        example_vector = data[example_idx,:]
        scores[example_idx] = norm(
            (example_vector - mean_vector), ord=2)
        
    # choose threshold
    threshold = -threshold_at_recall(
        labels, -scores, recall_thresh=recall_thresh)
    print "Threshold for similarity: {}".format(threshold)
    threshold_filter = scores <= threshold

    return threshold, threshold_filter



# TODO - eventually try a multiple gaussians threshold
def sd_cutoff_old(array, col_mask, std_thresh=3, axis=1):
    """Current heuristic along the motif axis
    """
    # row normalize and mask
    array_norm = np.divide(array, np.max(array, axis=1, keepdims=True))
    array_means = np.mean(array_norm, axis=0)
    array_means_masked = np.multiply(col_mask, array_means)
    nonzero_means = array_means_masked[np.nonzero(array_means_masked)]
    
    # across all vals, get a mean and standard dev
    mean_val = np.mean(nonzero_means)
    std_val = np.std(nonzero_means)

    # threshold, and multiply with original mask for final mask
    sd_mask = (array_means > (mean_val + (std_val * std_thresh))).astype(int)
    final_mask = np.multiply(sd_mask, col_mask)
    
    return final_mask


def sd_cutoff(array, col_mask=None, std_thresh=2, axis=1):
    """Current heuristic along the motif axis
    """
    # row normalize and mask
    max_vals = np.max(array, axis=1, keepdims=True)
    array_norm = np.divide(
        array,
        max_vals,
        out=np.zeros_like(array),
        where=max_vals!=0)
    array_means = np.mean(array_norm, axis=0)

    if col_mask is not None:
        array_means_masked = np.multiply(col_mask, array_means)
        nonzero_means = array_means_masked[np.nonzero(array_means_masked)]
    else:
        array_means_masked = array_means
        nonzero_means = array_means

    # mirror it to get a full normal centered at zero
    nonzero_means = np.hstack((nonzero_means, -nonzero_means))
    
    # across all vals, get a mean and standard dev
    mean_val = np.mean(nonzero_means)
    std_val = np.std(nonzero_means)

    # threshold, and multiply with original mask for final mask
    #sd_mask = (array_means > (mean_val + (std_val * std_thresh))).astype(int)
    sd_mask = (array_means > (mean_val + (std_val * std_thresh))).astype(int)
    if col_mask is not None:
        final_mask = np.multiply(sd_mask, col_mask)
    else:
        final_mask = sd_mask
    
    return final_mask


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


def get_manifold_metrics(
        h5_file,
        pwm_list,
        manifold_h5_file,
        features_key=DataKeys.FEATURES,
        cluster_key=DataKeys.CLUST_FILT,
        null_cluster_present=True,
        recall_thresh=0.10):
    """extract metrics on the pwm manifold
    """
    from tronn.interpretation.motifs import correlate_pwms
    from tronn.interpretation.motifs import reduce_pwms
    from tronn.interpretation.motifs import get_individual_pwm_thresholds

    # TODO move this out. doesn't belong in pure manifold calling
    # set up hclust on pwms (to know which pwms look like each other)
    cor_filt_mat, distances = correlate_pwms(
        pwm_list,
        cor_thresh=0.3,
        ncor_thresh=0.2,
        num_threads=24)
    hclust = linkage(squareform(1 - distances), method="ward")

    # get clusters and data
    with h5py.File(h5_file, "r") as hf:
        data = hf[features_key][:] # {N, task, M}
        raw_data = hf[DataKeys.ORIG_SEQ_PWM_SCORES_SUM][:] # {N, 1, M}
        clusters = hf[cluster_key][:] # {N}
    cluster_ids = sorted(list(set(clusters.tolist())))
    print cluster_ids
    if null_cluster_present:
        cluster_ids.remove(-1)
    
    # set up save arrays as dict
    out_shape = (len(cluster_ids), data.shape[1], data.shape[2])
    out_arrays = {}
    out_arrays[DataKeys.MANIFOLD_CENTERS] = np.zeros(out_shape)
    out_arrays[DataKeys.MANIFOLD_WEIGHTINGS] = np.zeros(out_shape)
    out_arrays[DataKeys.MANIFOLD_THRESHOLDS] = np.zeros((len(cluster_ids), data.shape[1]))
    out_arrays[DataKeys.MASTER_PWMS] = np.zeros((data.shape[2]))

    # TODO add in soft clustering array (for each cluster, after recall)
    
    # then calculate metrics on each cluster separately
    for cluster_idx in xrange(len(cluster_ids)):
        cluster_id = cluster_ids[cluster_idx]
        print "cluster_id: {}".format(cluster_id),
        in_cluster = clusters == cluster_id # chekc this
            
        # get the data in this cluster
        scores_in_cluster = data[np.where(in_cluster)[0],:,:] # {N, task, M}
        print "total examples: {}".format(scores_in_cluster.shape[0]),
        raw_scores_in_cluster = raw_data[np.where(in_cluster)[0],:,:] # {N, 1, M}
        raw_center = np.median(raw_scores_in_cluster, axis=0) # {1, M}

        # row norm
        orig_shape = scores_in_cluster.shape
        scores_in_cluster = np.reshape(scores_in_cluster, [-1, orig_shape[2]]) # {N*task, M}
        scores_in_cluster = rownorm2d(scores_in_cluster) # TODO consider whether this is useful
        scores_in_cluster = np.reshape(scores_in_cluster, orig_shape)

        # ignore zeros
        scores_in_cluster = scores_in_cluster[np.max(scores_in_cluster, axis=[1,2])>0]
        print "after remove zeros: {}".format(scores_in_cluster.shape[0])
        
        # calculate cluster medoids
        centers = np.median(scores_in_cluster, axis=0) # {task, M}
        center_ratios = np.divide(centers, raw_center) # {task, M}

        # weighted raw data
        weighted_raw_data = np.multiply(raw_data, np.expand_dims(center_ratios, 0)) # {N, task, M}
        weighted_raw_data[np.logical_not(np.isfinite(weighted_raw_data))] = 0
        
        # calculate thresholds for each task separately (since euclidean and jaccard dont do matrix multiplies)
        for task_idx in xrange(data.shape[1]):
            print "task: {}, ".format(task_idx),

            # get threshold by first weighting the raw sequence
            # TODO check this again
            # NOTE: compared dot product, euclidean, and jaccard. dot product was best
            similarity_threshold, threshold_filter = get_threshold_on_dot_product(
                centers[task_idx,:],
                weighted_raw_data,
                in_cluster,
                recall_thresh=recall_thresh) # ie grab the top 10% of regions
            print "passing filter: {}".format(np.sum(threshold_filter)),
            print "true positives: {}".format(np.sum(threshold_filter * cluster_labels)),

            # TODO still do this?
            # but likely pull out of this function?
            # and better to do this with the soft clustering results
            # IF keeping this, MUST row norm...
            # needed for PWM vector though
            # so if it depends on row norm, what does it really mean? its a filter on rank?
            # get significant motifs with their thresholds
            #weighted_raw_scores_in_cluster = weighted_raw_scores[np.where(cluster_labels)[0],:]
            #pwm_vector = reduce_pwms(weighted_raw_scores, hclust, pwm_list) # TODO consider switching to weighted
            #pwm_thresholds = get_individual_pwm_thresholds(
            #    weighted_raw_scores,
            #    cluster_labels,
            #    pwm_vector)
            
            # if there are no significant pwms, do not save out
            # TODO but somehow need to filter this out?
            if np.sum(pwm_vector) == 0:
                continue
                
            # and save into master pwm vector
            master_pwm_vector += pwm_vector
            print "total in master pwms: {}".format(np.sum(master_pwm_vector > 0))
            
            # save this info out
            out_arrays[DataKeys.MANIFOLD_CENTERS][cluster_idx,task_idx,:] = center
            out_arrays[DataKeys.MANIFOLD_WEIGHTINGS][cluster_idx,task_idx,:] = center_ratio
            out_arrays[DataKeys.MANIFOLD_THRESHOLDS][cluster_idx,task_idx] = similarity_threshold
            #manifold_pwm_thresholds[i,j,:] = pwm_thresholds

        
    # remember to save out cluster ids also

    return

def get_manifold_centers(
        scores_h5_file,
        dataset_keys,
        cluster_key,
        manifold_h5_file,
        pwm_list,
        pwm_dict,
        null_cluster=True,
        recall_thresh=0.10):
    """get manifold centers
    """
    from tronn.interpretation.motifs import correlate_pwms
    from tronn.interpretation.motifs import reduce_pwms
    from tronn.interpretation.motifs import get_individual_pwm_thresholds
    
    raw_scores_key = "raw-pwm-scores"
    raw_scores_key = DataKeys.ORIG_SEQ_PWM_SCORES # TODO consider using the hits instead?
    
    # prep: set up hclust for pwms
    cor_filt_mat, distances = correlate_pwms(
        pwm_list,
        cor_thresh=0.3,
        ncor_thresh=0.2,
        num_threads=24)
    hclust = linkage(squareform(1 - distances), method="ward")

    # go through clusters
    with h5py.File(scores_h5_file, "r") as hf:

        # determine the number of clusters
        cluster_ids_by_example = hf[cluster_key][:,0]
        cluster_ids = list(set(cluster_ids_by_example.tolist()))
        if null_cluster:
            max_id = max(cluster_ids)
            cluster_ids.remove(max_id)

        # get raw scores
        raw_scores = hf[raw_scores_key][:]

        # TEMP
        raw_scores = np.sum(np.squeeze(raw_scores), axis=1)
        print raw_scores.shape
        print "REMOVE THIS LATER"

        # set up master pwm vector (union of all seen significant motifs)
        master_pwm_vector = np.zeros((raw_scores.shape[1]))

        # set up master arrays
        out_shape = (len(cluster_ids), len(dataset_keys), raw_scores.shape[1])
        manifold_centers = np.zeros(out_shape)
        manifold_weights = np.zeros(out_shape)
        manifold_thresholds = np.zeros((len(cluster_ids), len(dataset_keys)))
        manifold_pwm_thresholds = np.zeros(out_shape)

        # per cluster, extract the manifold description,
        # which is the motifspace vector and threshold for distance,
        # and the significant motifs to check.
        for i in xrange(len(cluster_ids)):

            # get cluster id
            cluster_id = cluster_ids[i]
            print "cluster_id: {}".format(cluster_id)
            cluster_labels = cluster_ids_by_example == cluster_id
            
            # get the raw scores in this cluster and the mean
            raw_scores_in_cluster = raw_scores[np.where(cluster_labels)[0],:]
            mean_raw_scores_in_cluster = np.mean(raw_scores_in_cluster, axis=0)

            for j in xrange(len(dataset_keys)):
                print "task: {}, ".format(j),
                dataset_key = dataset_keys[j]
                
                # get subset
                weighted_scores = hf[dataset_key][:]
                weighted_scores_in_cluster = weighted_scores[np.where(cluster_labels)[0],:]
                print "total examples: {}".format(weighted_scores_in_cluster.shape),

                # row normalize and remove zeroes
                max_vals = np.max(weighted_scores_in_cluster, axis=1, keepdims=True)
                weighted_scores_in_cluster = np.divide(
                    weighted_scores_in_cluster,
                    max_vals,
                    out=np.zeros_like(weighted_scores_in_cluster),
                    where=max_vals!=0)
                weighted_scores_in_cluster = weighted_scores_in_cluster[
                    np.max(weighted_scores_in_cluster, axis=1) >0]
                print "after remove zeroes: {}".format(weighted_scores_in_cluster.shape),

                # get the mean vector
                #mean_weighted_scores_in_cluster = np.mean(weighted_scores_in_cluster, axis=0)
                mean_weighted_scores_in_cluster = np.median(weighted_scores_in_cluster, axis=0)
                
                # get mean ratio
                mean_ratio_scores_in_cluster = np.divide(
                    mean_weighted_scores_in_cluster,
                    mean_raw_scores_in_cluster)

                # get threshold by first weighting the raw sequence
                # NOTE: compared dot product, euclidean, and jaccard. dot product was best
                weighted_raw_scores = np.multiply(
                    raw_scores,
                    np.expand_dims(mean_ratio_scores_in_cluster, axis=0))
                weighted_raw_scores[np.logical_not(np.isfinite(weighted_raw_scores))] = 0
                
                similarity_threshold, threshold_filter = get_threshold_on_dot_product(
                    mean_weighted_scores_in_cluster,
                    weighted_raw_scores,
                    cluster_labels,
                    recall_thresh=recall_thresh) # ie grab the top 10% of regions
                print "passing filter: {}".format(np.sum(threshold_filter)),
                print "true positives: {}".format(np.sum(threshold_filter * cluster_labels)),

                # get significant motifs with their thresholds
                weighted_raw_scores_in_cluster = weighted_raw_scores[np.where(cluster_labels)[0],:]
                pwm_vector = reduce_pwms(weighted_raw_scores, hclust, pwm_list) # TODO consider switching to weighted
                pwm_thresholds = get_individual_pwm_thresholds(
                    weighted_raw_scores,
                    cluster_labels,
                    pwm_vector)

                # if there are no significant pwms, do not save out
                # TODO but somehow need to filter this out?
                if np.sum(pwm_vector) == 0:
                    continue
                
                # and save into master pwm vector
                master_pwm_vector += pwm_vector
                print "total in master pwms: {}".format(np.sum(master_pwm_vector > 0))
                
                # save this info out
                manifold_centers[i,j,:] = mean_weighted_scores_in_cluster
                manifold_weights[i,j,:] = mean_ratio_scores_in_cluster
                manifold_thresholds[i,j] = similarity_threshold
                manifold_pwm_thresholds[i,j,:] = pwm_thresholds
                cluster_key_prefix = "motifspace.cluster-{}".format(cluster_id)

    # finally, save out all results
    with h5py.File(manifold_h5_file, "a") as out:
        out.create_dataset("manifold_centers", data=manifold_centers)
        out.create_dataset("manifold_weights", data=manifold_weights)
        out.create_dataset("manifold_thresholds", data=manifold_thresholds)
        out.create_dataset("pwm_thresholds", data=manifold_pwm_thresholds)
        out.create_dataset("pwm_names", data=[pwm.name for pwm in pwm_list])
        out.create_dataset("master_pwm_vector", data=master_pwm_vector)
    
    return None


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
                max_vals = np.max(data, axis=1, keepdims=True)
                data_norm = np.divide(
                    data,
                    max_vals,
                    out=np.zeros_like(data),
                    where=max_vals!=0)
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



