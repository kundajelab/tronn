"""Contains nets for quick thresholding
"""

import tensorflow as tf


# TODO(dk)


def shuffle_null_threshold(features, labels, config, is_training=False):
    """Shuffle values to get a null distribution at each position
    """
    assert is_training == False
    num_shuffles = config.get("shuffled_null.num_shuffles", 100)
    top_k = int(100*config.get("shuffled_null.pval", 0.05))
    
    examples_list = tf.unstack(features) # list of {1, seq_len, 4}
    
    for example in examples_list:
        # shuffle

        # get top k

        # if that position is in top k, pass through otherwise zero out
        
        pass
    
    return


def threshold_poisson(signal, pval):
    """Given importance scores, calculates poisson pval
    and thresholds at that pval

    Args:
      signal: input tensor
      pval: the pval threshold

    Returns:
      out_tensor: output thresholded tensor
    """
    percentile_val = 1. - pval
    signal_shape = signal.get_shape()
    print "feature shape", signal_shape
    
    # get mean
    signal_mean = tf.reduce_mean(signal, axis=[1, 2, 3])
    print "signal mean", signal_mean.get_shape()

    # calculate poisson
    pois_distributions = tf.random_poisson(signal_mean, [1000])
    thresholds = percentile(pois_distributions, percentile_val, axis=0)

    for dim_idx in range(1, len(signal_shape)):
        to_stack = [thresholds for i in range(signal_shape[dim_idx])] # here?
        thresholds = tf.stack(to_stack, dim_idx)

    print "thresh full", thresholds.get_shape()
        
    # and threshold
    greaterthan_tensor = tf.cast(tf.greater(signal, thresholds), tf.float32)
    thresholded_tensor = signal * greaterthan_tensor

    out_tensor = tf.transpose(tf.squeeze(thresholded_tensor), [0, 2, 1])

    return out_tensor, signal_mean


def threshold_gaussian(features, labels, config, is_training=False):
    """Given importance scores, calculates stdev cutoff
    and thresholds at that pval

    Args:
      signal: input tensor
      pval: the pval threshold

    Returns:
      out_tensor: output thresholded tensor
    """
    assert is_training == False
    
    num_stdev = config.get("stdev", 3)
    two_tailed = config.get("two_tailed", True)
    
    # get mean and stdev to get threshold
    signal_mean, signal_var = tf.nn.moments(features, axes=[1, 2, 3])
    signal_stdev = tf.sqrt(signal_var)
    thresholds = tf.add(signal_mean, tf.scalar_mul(num_stdev, signal_stdev))

    # expand thresholds for broadcasting
    signal_shape = features.get_shape()
    for dim_idx in xrange(1, len(signal_shape)):
        thresholds = tf.expand_dims(thresholds, axis=dim_idx)
    
    #for dim_idx in range(1, len(signal_shape)):
    #    to_stack = [thresholds for i in range(signal_shape[dim_idx])] # here?
    #    thresholds = tf.stack(to_stack, dim_idx)

    # set up mask
    if two_tailed:
        greaterthan_tensor = tf.cast(tf.greater(features, thresholds), tf.float32)
        lessthan_tensor = tf.cast(tf.less(features, -thresholds), tf.float32)
        mask_tensor = tf.add(greaterthan_tensor, lessthan_tensor)
    else:
        mask_tensor = tf.cast(tf.greater(features, thresholds), tf.float32)
        
    # mask out insignificant features
    features = features * mask_tensor

    return features, labels, config



def threshold_topk(features, labels, config, is_training=False):
    """Get top k hits across all axes
    """
    assert is_training == False

    k_val = config.get("k_val", 4)
    
    # TODO(dk) allow adjustment of which axes to get topk over
    
    # grab the top k and determine threshold val
    input_shape = features.get_shape().as_list()
    top_k_shape = [input_shape[0], input_shape[1]*input_shape[2]*input_shape[3]]
    top_k_vals, top_k_indices = tf.nn.top_k(tf.reshape(features, top_k_shape), k=k_val)
    
    thresholds = tf.reshape(tf.reduce_min(top_k_vals, axis=1, keep_dims=True), [input_shape[0], 1, 1, 1])
    thresholds_mask = tf.cast(tf.greater_equal(features, thresholds), tf.float32) # out: (example, 1, bp_pos, motif)

    # threshold values
    top_scores = tf.multiply(features, thresholds_mask) # this gives you back actual scores

    # and aggregate
    #top_hits_per_example = tf.squeeze(tf.reduce_mean(top_hits_w_location, axis=2)) # out: (example, motif)
    features = tf.squeeze(tf.reduce_max(top_hits_w_location, axis=2)) # out: (example, motif)

    return features, labels, config

