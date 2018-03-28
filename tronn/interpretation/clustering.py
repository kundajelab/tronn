# description: functions around clustering (ie unsupervised learning)

import os
import h5py

import numpy as np
import pandas as pd

from numpy.linalg import norm
from collections import Counter

from tronn.interpretation.learning import threshold_at_recall

import phenograph


def cluster_by_task(
        h5_file,
        h5_dataset_keys,
        out_key,
        num_threads=24):
    """Get a clustering per task (ie, per cell state)
    Currently set to a phenograph clustering
    """
    with h5py.File(h5_file, "a") as hf:
        num_examples = hf["example_metadata"].shape[0]

        # generate a new dataset that is {N, len(h5_dataset_keys)}
        clusters_hf = hf.create_dataset(
            out_key, (num_examples, len(h5_dataset_keys)), dtype=int)

        # add attribute
        clusters_hf.attrs["tasks"] = h5_dataset_keys
        
        # for each dataset, cluster and save to correct spot
        for i in xrange(len(h5_dataset_keys)):
            h5_dataset_key = h5_dataset_keys[i]
            dataset = hf[h5_dataset_key][:]
            communities, graph, Q = phenograph.cluster(
                dataset, n_jobs=num_threads)
            clusters_hf[:,i] = communities

    return None


def enumerate_metaclusters(
        h5_file,
        h5_clusters_key,
        out_key):
    """Given a series of task-specific clusters,
    enumerate the metacommunities - groups of regions
    that share all the same task-specific communities
    """
    with h5py.File(h5_file, "a") as hf:
        num_examples = hf["example_metadata"].shape[0]

        #del hf[out_key]
        
        # generate a new dataset that is {N, 1}
        metaclusters_hf = hf.create_dataset(
            out_key, (num_examples, 1), dtype="S100")
    
        # pull in the clusters dataset
        task_clusters = pd.DataFrame(hf[h5_clusters_key][:]).astype(int)
    
        # enumerate
        task_clusters["enumerated"] = ["" for i in xrange(task_clusters.shape[0])]
        for i in xrange(task_clusters.shape[1]):
            task_clusters["enumerated"] = task_clusters["enumerated"] + task_clusters.iloc[
                :, task_clusters.shape[1]-i-2].astype(str).str.zfill(2)

        # save back to file
        metaclusters_hf[:,0] = task_clusters["enumerated"].tolist()

    return None


def refine_clusters(
        h5_file,
        clusters_key,
        out_key,
        fractional_threshold=0.005):
    """Given a clusters dataset, remove small clusters 
    and save out to a new dataset
    """
    with h5py.File(h5_file, "a") as hf:
        num_examples = hf["example_metadata"].shape[0]

        # generate a new dataset that is {N, 1}
        refined_clusters_hf = hf.create_dataset(
            out_key, hf[clusters_key].shape, dtype=int)
        
        # then for each column, refine
        for i in xrange(hf[clusters_key].shape[1]):
            clusters = hf[clusters_key][:,i]
            new_idx = len(list(set(clusters.tolist())))
            
            new_clusters = np.zeros(clusters.shape)
            counts = Counter(clusters.tolist())

            # for each cluster, check size and change as necessary
            cluster_ids = counts.most_common()
            for j in xrange(len(cluster_ids)):
                cluster_id, count = cluster_ids[j]
                if float(count) / num_examples < fractional_threshold:
                    # replace cluster id with new index
                    new_clusters[clusters==cluster_id] = new_idx
                else:
                    new_clusters[clusters==cluster_id] = j

            # back check
            print "reduced num clusters:", len(list(set(new_clusters.tolist())))
            print set(new_clusters.tolist())
                    
            # save into dataset
            refined_clusters_hf[:,i] = new_clusters
                    
    return None


def visualize_clusters(
        h5_file,
        dataset_key,
        cluster_key,
        cluster_col=0,
        remove_final_cluster=0):
    """Visualize clustering. Note that the R script is downsampling
    to make things visible.
    """
    # do this in R
    plot_example_x_pwm = (
        "plot.example_x_pwm.from_h5.R {0} {1} {2} {3} {4}").format(
            h5_file, dataset_key, cluster_key, cluster_col+1, remove_final_cluster)
    print plot_example_x_pwm
    os.system(plot_example_x_pwm)
    
    return None


def visualize_cluster_means(
        h5_file,
        dataset_keys,
        cluster_key,
        cluster_col,
        remove_final_cluster=0):
    """Visualize cluster means
    """
    # do this in R
    plot_cluster_mean = (
        "plot.pwm_x_task.from_h5.R {0} {1} {2} {3} {4}").format(
            h5_file, dataset_key, cluster_key, cluster_col+1, remove_final_cluster)
    print plot_cluster_mean
    os.system(plot_cluster_mean)
    
    return None


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


def sd_cutoff(array, col_mask, std_thresh=2, axis=1):
    """Current heuristic along the motif axis
    """
    # row normalize and mask
    array_norm = np.divide(array, np.max(array, axis=1, keepdims=True))
    array_means = np.mean(array_norm, axis=0)
    array_means_masked = np.multiply(col_mask, array_means)
    nonzero_means = array_means_masked[np.nonzero(array_means_masked)]

    # mirror it to get a full normal centered at zero
    nonzero_means = np.hstack((nonzero_means, -nonzero_means))
    
    # across all vals, get a mean and standard dev
    mean_val = np.mean(nonzero_means)
    std_val = np.std(nonzero_means)

    # threshold, and multiply with original mask for final mask
    #sd_mask = (array_means > (mean_val + (std_val * std_thresh))).astype(int)
    sd_mask = (array_means > (mean_val + (std_val * std_thresh))).astype(int)
    final_mask = np.multiply(sd_mask, col_mask)
    
    return final_mask
