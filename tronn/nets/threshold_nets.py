"""Contains nets for quick thresholding
"""

import tensorflow as tf


def pois_cutoff(signal, pval):
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


def stdev_cutoff(signal, num_stdev=3):
    """Given importance scores, calculates stdev cutoff
    and thresholds at that pval

    Args:
      signal: input tensor
      pval: the pval threshold

    Returns:
      out_tensor: output thresholded tensor
    """
    #percentile_val = 1. - pval
    signal_shape = signal.get_shape()
    print "feature shape", signal_shape
    
    # get mean and stdev
    signal_mean, signal_var = tf.nn.moments(signal, axes=[1, 2, 3])
    signal_stdev = tf.sqrt(signal_var)
    thresholds = tf.add(signal_mean, tf.scalar_mul(num_stdev, signal_stdev))

    for dim_idx in range(1, len(signal_shape)):
        to_stack = [thresholds for i in range(signal_shape[dim_idx])] # here?
        thresholds = tf.stack(to_stack, dim_idx)

    print "thresh full", thresholds.get_shape()
        
    # and threshold
    greaterthan_tensor = tf.cast(tf.greater(signal, thresholds), tf.float32)
    thresholded_tensor = signal * greaterthan_tensor

    #out_tensor = tf.transpose(tf.squeeze(thresholded_tensor), [0, 2, 1])

    return out_tensor
