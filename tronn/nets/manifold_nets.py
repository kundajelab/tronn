"""contains nets to locate examples in motifspace manifolds
"""

import h5py

import numpy as np
import tensorflow as tf

from tronn.nets.filter_nets import filter_and_rebatch


def distance_metric(features, centers, metric="dot"):
    """determine distance to center
    """
    if metric == "dot":
        distances = tf.reduce_sum(tf.multiply(features, centers), axis=2)
    else:
        print "distance metric does not exist yet"
        
    return distances


def score_manifold_distances(inputs, params):
    """given motifspace centers, score distance from the centers
    """
    # assertions
    assert params.get("raw-pwm-scores-key") is not None
    assert inputs.get(params["raw-pwm-scores-key"]) is not None
    assert params.get("manifold") is not None
    assert params.get("pwms") is not None
    
    # get features and pass on rest
    features = inputs[params["raw-pwm-scores-key"]] # {N, motif}
    features = tf.expand_dims(features, axis=1) # {N, 1, motif}
    features = tf.expand_dims(features, axis=3) # {N, 1, motif, 1}
    outputs = dict(inputs)

    # params
    manifold_h5_file = params["manifold"]

    # get the weights
    with h5py.File(manifold_h5_file, "r") as hf:
        feature_weights = hf["manifold_weights"][:] # {cluster, task, motif}
    feature_weights = np.transpose(feature_weights, axes=[1, 2, 0]) # {task, motif, cluster}
    feature_weights = np.expand_dims(feature_weights, axis=0) # {1, task, motif, cluster}
    feature_weights = tf.convert_to_tensor(feature_weights, dtype=tf.float32)

    # multiply by the features
    features = tf.multiply(features, feature_weights) # {N, task, motif, cluster}

    # get the centers
    with h5py.File(manifold_h5_file, "r") as hf:
        cluster_centers = hf["manifold_centers"][:] # {cluster, task, motif}
    cluster_centers = np.transpose(cluster_centers, axes=[1, 2, 0]) # {task, motif, cluster}
    cluster_centers = np.expand_dims(cluster_centers, axis=0) # {1, task, motif, cluster}
    cluster_centers = tf.convert_to_tensor(cluster_centers, dtype=tf.float32)

    # get distances
    outputs["manifold_distances"] = distance_metric(
        features, cluster_centers, metric="dot") # {N, task, cluster}

    # get thresholds and thresholded results
    with h5py.File(manifold_h5_file, "r") as hf:
        manifold_thresholds = hf["manifold_thresholds"][:] # {cluster, task}
    manifold_thresholds = np.transpose(manifold_thresholds, axes=[1, 0])
    manifold_thresholds = np.expand_dims(manifold_thresholds, axis=0) # {1, task, cluster}
    manifold_thresholds = tf.convert_to_tensor(manifold_thresholds, dtype=tf.float32)

    # compare to distances
    # currently - must match on ALL tasks, but ANY cluster
    passes_thresholds = tf.cast(
        tf.greater_equal(outputs["manifold_distances"], manifold_thresholds),
        tf.float32)
    outputs["manifold_clusters"] = tf.reduce_mean(passes_thresholds, axis=1) # {N, cluster}
    
    # now get motif thresholds
    with h5py.File(manifold_h5_file, "r") as hf:
        pwm_thresholds = hf["pwm_thresholds"][:] # {cluster, task, motif}
    pwm_thresholds = np.transpose(pwm_thresholds, axes=[1, 2, 0]) # {task, motif, cluster}
    pwm_thresholds = np.expand_dims(pwm_thresholds, axis=0) # {1, task, motif, cluster}
    pwm_thresholds = tf.convert_to_tensor(pwm_thresholds, dtype=tf.float32)
    cutoffs = tf.reduce_sum(pwm_thresholds, axis=2) # {1, task, cluster}

    # get whether passed cutoffs or not
    passes_thresholds = tf.reduce_sum(
        tf.cast(
            tf.greater_equal(features, pwm_thresholds),
            tf.float32),
        axis=2) # {1, task, cluster}
    passed_cutoffs = tf.cast(tf.greater_equal(passes_thresholds, cutoffs), tf.float32) # {N, task, cluster}
    outputs["sig_pwms_present"] = tf.reduce_mean(passed_cutoffs, axis=1) # {N, cluster}
    
    return outputs, params


def filter_by_manifold_distance(inputs, params):
    """given the manifold distances, filter
    """
    # assertions
    assert inputs.get("manifold_clusters") is not None
    features = inputs["manifold_clusters"] # {N, cluster}

    # get condition mask
    inputs["condition_mask"] = tf.greater_equal(
        tf.reduce_max(features, axis=1),
        [1]) # {N}

    # filter and rebatch
    params.update({"name": "manifold_filter"})
    outputs, params = filter_and_rebatch(inputs, params)
    
    return outputs, params


def filter_by_sig_pwm_presence(inputs, params):
    """given the manifold distances, filter
    """
    # assertions
    assert inputs.get("sig_pwms_present") is not None
    features = inputs["sig_pwms_present"] # {N, cluster}
    
    # currently - must match on ALL tasks, but ANY cluster
    inputs["condition_mask"] = tf.greater_equal(
        tf.reduce_max(features, axis=1),
        [1]) # {N}

    # filter and rebatch
    params.update({"name": "sig_motif_filter"})
    outputs, params = filter_and_rebatch(inputs, params)
    
    return outputs, params



