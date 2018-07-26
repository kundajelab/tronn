"""contains nets to locate examples in motifspace manifolds
"""

import h5py

import numpy as np
import tensorflow as tf

from tronn.nets.filter_nets import filter_and_rebatch

from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys


def batch_jaccard_distance_from_centers(features, centers, axis=-1):
    """calculates the jaccard for multiple
    centers on a batch of examples
    calculates on the last axis
    """
    min_vals = tf.minimum(features, centers) # {N, cluster, task, M}
    max_vals = tf.maximum(features, centers) # {N, cluster, task, M}

    similarities = tf.divide(
        tf.reduce_sum(min_vals, axis=axis),
        tf.reduce_sum(max_vals, axis=axis)) # {N, cluster, task}
    
    return similarities


def batch_euclidean_distance_from_centers(features, centers, axis=-1):
    """calculate euclidean distance from the centers
    """
    diffs = tf.subtract(features, centers) # {N, cluster, task, M}
    norms = tf.norm(diffs, ord="euclidean", axis=axis) # {N, cluster, task}

    return norms


def batch_dot_distance_from_centers(features, centers, axis=-1):
    """calculate dot product with centers
    """
    distances = tf.reduce_sum(tf.multiply(features, centers), axis=axis)
    
    return distances


def score_distances_on_manifold(inputs, params):
    """score distances from center
    """
    assert inputs.get(DataKeys.FEATURES) is not None
    assert params.get("manifold") is not None
    
    # features
    features = inputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM]
    outputs = dict(inputs)

    # centers and thresholds
    with h5py.File(params["manifold"], "r") as hf:
        centers = hf[DataKeys.MANIFOLD_CENTERS][:].astype(np.float32)
        thresholds = hf[DataKeys.MANIFOLD_THRESHOLDS][:].astype(np.float32)
        cluster_ids = hf[DataKeys.MANIFOLD_CENTERS].attrs[AttrKeys.CLUSTER_IDS]

    # adjust by cluster ids
    centers = centers[cluster_ids]
    thresholds = thresholds[cluster_ids]
    
    # calculate distances
    features = tf.expand_dims(features, axis=1)
    centers = tf.expand_dims(centers, axis=0)
    distances = batch_jaccard_distance_from_centers(features, centers)
    outputs[DataKeys.MANIFOLD_SCORES] = distances
    
    # compare to thresholds
    thresholds = tf.expand_dims(thresholds, axis=0) # {N, cluster, task}
    passed_thresholds = tf.greater(distances, thresholds)

    # give back a single score - passed ALL thresholds
    passed_all_thresholds = tf.cast(tf.reduce_all(passed_thresholds, axis=-1), tf.float32) # {N, cluster}
    outputs[DataKeys.MANIFOLD_CLUST] = passed_all_thresholds
    
    return outputs, params


def filter_by_manifold_distances(inputs, params):
    """filter by passed thresholds
    """
    assert inputs.get(DataKeys.MANIFOLD_CLUST) is not None

    # features
    features = inputs[DataKeys.MANIFOLD_CLUST]
    
    # check condition
    inputs["condition_mask"] = tf.reduce_any(
        tf.greater(features, 0), axis=1)

    # filter
    params.update({"name": "manifold_filter"})
    outputs, params = filter_and_rebatch(inputs, params)
    
    return outputs, params
