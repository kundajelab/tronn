# description: functions around clustering (ie unsupervised learning)

import os
import h5py
import logging
import phenograph

import numpy as np

from tronn.stats.distances import score_distances
from tronn.stats.thresholding import threshold_at_recall
from tronn.stats.thresholding import threshold_at_fdr

from tronn.util.utils import DataKeys
from tronn.util.h5_utils import AttrKeys
from tronn.util.bioinformatics import make_bed

from tronn.visualization import visualize_clustered_h5_dataset_full


class ClustersManager(object):
    """manage hard or soft clusters in a consistent way"""
    
    def __init__(self, array, active_cluster_ids=None):
        """from the array, extract whether 
        hard vs soft and also cluster ids
        """
        # save array
        self.array = array
        
        # determine hard or soft clustering
        if len(array.shape) == 1:
            self.all_cluster_ids = sorted(
                list(set(self.array.tolist())))
            self.hard_clustering = True
        else:
            self.all_cluster_ids = range(
                self.array.shape[1])
            self.hard_clustering = False

        # set up active cluster ids if exist
        if active_cluster_ids is not None:
            self.active_cluster_ids = active_cluster_ids.tolist()
        else:
            self.active_cluster_ids = self.all_cluster_ids
        
            
    def cluster_mask_generator(self):
        """go through the active cluster ids
        return the cluster id and the cluster mask
        """
        for cluster_idx in xrange(len(self.active_cluster_ids)):
            cluster_id = self.active_cluster_ids[cluster_idx]

            if self.hard_clustering:
                cluster_mask = self.array == cluster_id
            else:
                cluster_mask = self.array[:,cluster_id] > 0

            yield cluster_id, cluster_mask


    def get_clusters(self):
        """return the cluster array
        """
        return self.array

    
    def get_active_cluster_ids(self):
        """get the cluster ids
        """
        return self.active_cluster_ids

    
    def get_all_cluster_ids(self):
        """get the cluster ids
        """
        return self.all_cluster_ids

    
    def which_cluster_ids_active(self):
        """provide back a vector of which cluster
        ids to use (assuming ordered)
        """
        return np.isin(
            np.array(self.all_cluster_ids),
            np.array(self.active_cluster_ids))

    
    def get_num_examples(self):
        """get the number of examples that 
        belong to an existing cluster
        """
        num_examples = 0
        generator = self.cluster_mask_generator()
        for cluster_id, cluster_mask in generator:
            num_examples += np.sum(cluster_mask)
        return num_examples
            

    def remove_cluster_id(self, cluster_id):
        """remove cluster id (but maintain the old clusters - change this?)
        """
        self.active_cluster_ids.remove(cluster_id)
    

def get_clusters_from_h5(
        h5_file,
        clusters_key,
        cluster_ids_attr_key=AttrKeys.CLUSTER_IDS):
    """helper function to standardize grabbing clusters from h5
    """
    with h5py.File(h5_file, "r") as hf:
        clusters = hf[clusters_key][:]
        cluster_ids = hf[clusters_key].attrs[AttrKeys.CLUSTER_IDS]
    clusters = ClustersManager(clusters, active_cluster_ids=cluster_ids)
    
    return clusters
        

def run_clustering(
        h5_file,
        dataset_key,
        cluster_key=DataKeys.CLUSTERS,
        num_threads=24,
        refine=True):
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
    cluster_ids = sorted(list(set(communities)))
    if -1 in cluster_ids:
        cluster_ids.delete(-1)

    # END CLUSTERING
    # ----------------------------------------------

    # save clusters
    with h5py.File(h5_file, "a") as out:
        #del out[cluster_key]
        out.create_dataset(cluster_key, data=communities)
        out[cluster_key].attrs[AttrKeys.CLUSTER_IDS] = cluster_ids

    # refine as desired
    if refine:
        filter_clusters_by_size(
            h5_file,
            cluster_key=cluster_key,
            filter_by="fraction",
            fractional_threshold=0.01)
    
    return None


def filter_clusters_by_size(
        h5_file,
        cluster_key,
        cluster_ids_attr_key=AttrKeys.CLUSTER_IDS,
        filter_by="fraction",
        fractional_threshold=0.01,
        min_count=50):
    """filter clusters by size, relative (fraction)
    or absolute (count)
    """
    # get clusters
    clusters = get_clusters_from_h5(h5_file, cluster_key)
    num_clustered_examples = clusters.get_num_examples()
    print "adjusting clusters..."
    print clusters.get_active_cluster_ids()
    
    # adjust based on fractional thresh and renumber based on size
    generator = clusters.cluster_mask_generator()
    remove_ids = []
    for cluster_id, cluster_mask in generator:

        if filter_by == "fraction":
            cluster_fraction = np.sum(cluster_mask) / float(num_clustered_examples)
            if cluster_fraction < fractional_threshold:
                remove_ids.append(cluster_id)
        elif filter_by == "count":
            cluster_count = np.sum(cluster_mask)
            if cluster_count < min_count:
                remove_ids.append(cluster_id)

    for cluster_id in remove_ids:
        clusters.remove_cluster_id(cluster_id)
        
    # backcheck
    print clusters.get_active_cluster_ids()
    
    # just save the attribute level
    with h5py.File(h5_file, "a") as out:
        out[cluster_key].attrs[cluster_ids_attr_key] = clusters.get_active_cluster_ids()
        
    return clusters.get_active_cluster_ids()


def get_cluster_bed_files(
        h5_file,
        file_prefix,
        clusters_key=DataKeys.CLUSTERS,
        seq_metadata_key=DataKeys.SEQ_METADATA):
    """generate BED files from metadata
    """
    # get clusters
    clusters = get_clusters_from_h5(h5_file, clusters_key)

    # get metadata
    with h5py.File(h5_file, "r") as hf:
        metadata = hf[seq_metadata_key][:]
    
    # for each cluster make a bed file
    generator = clusters.cluster_mask_generator()
    for cluster_id, cluster_mask in generator:
        cluster_metadata = metadata[cluster_mask]
        cluster_prefix = "{0}.cluster-{1}".format(file_prefix, cluster_id)
        metadata_file = "{}.metadata.txt".format(cluster_prefix)
        metadata_bed = "{}.bed".format(cluster_prefix)
        np.savetxt(metadata_file, cluster_metadata, fmt="%s")
        make_bed(metadata_file, metadata_bed)
        
    return None


# TODO - make one big manifold (flatten) and go from there
# since that's how clustering was calculated
def summarize_clusters_on_manifold(
        h5_file,
        features_key=DataKeys.FEATURES,
        cluster_key=DataKeys.CLUSTERS,
        centers_key=DataKeys.MANIFOLD_CENTERS,
        thresholds_key=DataKeys.MANIFOLD_THRESHOLDS,
        manifold_clusters_key=DataKeys.MANIFOLD_CLUST,
        fdr=0.50,
        recall_thresh=0.25,
        min_count=50,
        refine_clusters=True):
    """extract metrics on the pwm manifold
    """
    # get clusters and data
    clusters = get_clusters_from_h5(h5_file, cluster_key)
    with h5py.File(h5_file, "r") as hf:
        data = hf[features_key][:] # {N, task, M}
    num_clusters = len(clusters.get_active_cluster_ids())
        
    # set up save arrays as dict
    out_shape = (num_clusters, data.shape[1], data.shape[2])
    out_arrays = {}
    out_arrays[centers_key] = np.zeros(out_shape)
    out_arrays[thresholds_key] = np.zeros((num_clusters, data.shape[1]))
    out_arrays[manifold_clusters_key] = np.ones((data.shape[0], num_clusters))
    
    # then calculate metrics on each cluster separately
    generator = clusters.cluster_mask_generator()
    cluster_idx = 0
    for cluster_id, cluster_mask in generator:
        print "cluster_id: {}".format(cluster_id),
        
        # get the data in this cluster and remove zeros (if any)
        cluster_data = data[np.where(cluster_mask)[0],:,:] # {N, task, M}
        print "total examples: {}".format(cluster_data.shape[0]),
        cluster_data = cluster_data[np.max(cluster_data, axis=(1,2))>0]
        print "after remove zeros: {}".format(cluster_data.shape[0])
        
        # calculate cluster center using median
        out_arrays[centers_key][cluster_idx,:,:] = np.median(
            cluster_data, axis=0) # {task, M}
        
        # calculate thresholds for each task separately
        # TODO - do this as pool when stable
        # TODO - consider making tasks into 1 manifold and traversing that
        passed_threshold = np.zeros((data.shape[0], data.shape[1]))
        for task_idx in xrange(data.shape[1]):
            print "task: {}; ".format(task_idx),

            # score distances
            distances = score_distances(
                data[:,task_idx,:],
                out_arrays[centers_key][cluster_idx,task_idx,:],
                method="jaccard")
            
            # get threshold
            threshold = threshold_at_fdr(
                cluster_mask,
                distances,
                fdr=fdr)

            print "Threshold for when recall is {} (ie TPR): {};".format(
                recall_thresh, threshold),
            passed_threshold[:, task_idx] = distances > threshold
            
            print "passing filter: {};".format(np.sum(passed_threshold[:, task_idx])),
            print "true positives: {}".format(
                np.sum(passed_threshold[:, task_idx] * cluster_mask))
            out_arrays[thresholds_key][cluster_idx,task_idx] = threshold

        # calculate passing
        # NOTE this currently says need to pass threshold twice
        final_passed_threshold = np.all(passed_threshold, axis=1)
        out_arrays[manifold_clusters_key][:,cluster_idx] = final_passed_threshold
            
        passed_threshold_final = out_arrays[manifold_clusters_key][:,cluster_idx]
        print "final passing filter: {};".format(
            np.sum(passed_threshold_final)),
        print "final true positives: {}".format(
            np.sum(passed_threshold_final * cluster_mask))
        cluster_idx += 1
    
    # save out
    with h5py.File(h5_file, "a") as out:
        for key in out_arrays.keys():
            #del out[key]
            out.create_dataset(key, data=out_arrays[key])
            out[key].attrs[AttrKeys.CLUSTER_IDS] = clusters.get_active_cluster_ids()

    if refine_clusters:
        active_cluster_ids = filter_clusters_by_size(
            h5_file,
            cluster_key=manifold_clusters_key,
            filter_by="count",
            min_count=50)

        # and need to attach these to the
        # centers and thresholds
        with h5py.File(h5_file, "a") as out:
            out[manifold_clusters_key].attrs[AttrKeys.CLUSTER_IDS] = active_cluster_ids
            out[centers_key].attrs[AttrKeys.CLUSTER_IDS] = active_cluster_ids
            out[thresholds_key].attrs[AttrKeys.CLUSTER_IDS] = active_cluster_ids
            
    return None



def visualize_clustered_features_R(
        h5_file,
        data_key=DataKeys.FEATURES,
        clusters_key=DataKeys.CLUSTERS,
        cluster_ids_attr_key=AttrKeys.CLUSTER_IDS,
        colnames_attr_key=AttrKeys.PWM_NAMES):
    """this specifically handles the features
    """
    # example vs key
    visualize_clustered_h5_dataset_full(
        h5_file,
        data_key,
        clusters_key,
        cluster_ids_attr_key,
        colnames_attr_key,
        three_dims=True,
        cluster_columns=True,
        row_normalize=True,
        large_view=True,
        use_raster=True)
    
    # cluster map? {cluster, features}
    
    return None


def visualize_clustered_outputs_R(
        h5_file,
        visualize_keys,
        clusters_key=DataKeys.CLUSTERS,
        colnames_attr_key=AttrKeys.FILE_NAMES,
        cluster_ids_attr_key=AttrKeys.CLUSTER_IDS):
    """plot out the full matrix and then reduced to clusters
    """
    for key in visualize_keys:
        indices = visualize_keys[key][0]

        visualize_clustered_h5_dataset_full(
            h5_file,
            key,
            clusters_key,
            cluster_ids_attr_key,
            colnames_attr_key,
            indices=indices)

        visualize_clustered_h5_dataset_full(
            h5_file,
            key,
            clusters_key,
            cluster_ids_attr_key,
            colnames_attr_key,
            indices=indices,
            viz_type="cluster_map")

    return None


def visualize_multikey_outputs_R(
        h5_file,
        visualize_keys,
        clusters_key=DataKeys.CLUSTERS,
        colnames_attr_key=AttrKeys.FILE_NAMES,
        cluster_ids_attr_key=AttrKeys.CLUSTER_IDS):
    """plot out multi key outputs
    """
    # NOTE that key is a comma separated list
    for key in visualize_keys:
        indices = visualize_keys[key][0]
        
        visualize_clustered_h5_dataset_full(
            h5_file,
            key,
            clusters_key,
            cluster_ids_attr_key,
            colnames_attr_key,
            indices=indices,
            viz_type="multi_key")

    return None
