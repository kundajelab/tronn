"""Description: graphs that transform importance scores to other representations
"""

import tensorflow as tf


def input_x_grad(features, labels, config, is_training=False):
    """Layer-wise Relevance Propagation (Batch et al), implemented
    as input * gradient (equivalence is demonstrated in deepLIFT paper,
    Shrikumar et al). Returns the raw scores, adjust/normalize as needed.
    
    Args:
      features: the input tensor on which you want the importance scores
      labels: not used
    
    Returns:
      Input tensor weighted by gradient backpropagation.
    """
    assert is_training == False
    assert config.get("anchor") is not None
    
    anchor = config.get("anchor")
    [feature_grad] = tf.gradients(anchor, [features])
    features = tf.multiply(features, feature_grad, 'input_x_grad')
    
    return features, labels, config


# TODO basic deeplift
# TODO integrated gradients?


def multitask_importances(features, labels, config, is_training=False):
    """Set up importances coming from multiple tasks
    """
    assert is_training == False
    
    anchors = config.get("anchors")
    assert anchors is not None

    # TODO convert anchors
    
    importances_fn = config.get("importances_fn", input_x_grad)
    assert importances_fn is not None

    # get task specific importances
    task_importances = []
    for anchor_idx in xrange(len(anchors)):
        config["anchor"] = anchors[anchor_idx]
        task_importance = importances_fn(
            features, labels, config)
        task_importances.append(task_importance)
    features = tf.stack(task_importances, axis=1)

    return features, labels, config


def multitask_global_importance(features, labels, config, is_training=False):
    """Also get global importance
    """
    assert is_training == False
    
    append = config.get("append", True)

    features_max = tf.reduce_max(features, axis=1, keep_dims=True)

    if append:
        features = tf.stack(tf.unstack(features) + [features_max])
    else:
        features = features_max

    return features, labels, config


def stdev_cutoff(signal, num_stdev=3): # change this?
    """Given importance scores, calculates poisson pval
    and thresholds at that pval

    Args:
      signal: input tensor
      pval: the pval threshold

    Returns:
      out_tensor: output thresholded tensor
    """
    #percentile_val = 1. - pval
    signal_shape = signal.get_shape()
    
    # get mean and stdev
    signal_mean, signal_var = tf.nn.moments(signal, axes=[1, 2, 3])
    signal_stdev = tf.sqrt(signal_var)
    thresholds = tf.add(signal_mean, tf.scalar_mul(num_stdev, signal_stdev))

    for dim_idx in range(1, len(signal_shape)):
        to_stack = [thresholds for i in range(signal_shape[dim_idx])] # here?
        thresholds = tf.stack(to_stack, dim_idx)

    # and threshold
    greaterthan_tensor = tf.cast(tf.greater(signal, thresholds), tf.float32)
    lessthan_tensor = tf.cast(tf.less(signal, -thresholds), tf.float32)
    mask_tensor = tf.add(greaterthan_tensor, lessthan_tensor)
    thresholded_tensor = signal * mask_tensor

    #out_tensor = tf.transpose(tf.squeeze(thresholded_tensor), [0, 2, 1])

    return thresholded_tensor
