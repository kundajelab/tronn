# description: functions around clustering (ie unsupervised learning)

import h5py

import numpy as np
import pandas as pd

from collections import Counter

import phenograph


def cluster_by_task(
        h5_file,
        h5_dataset_keys,
        out_key,
        visualize=True,
        num_threads=24):
    """Get a clustering per task (ie, per cell state)
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

        #del hf[out_key]
        
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
                #count = counts[cluster_id]
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

