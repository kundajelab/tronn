"""Description: graphs that transform importance scores to other representations
"""

import tensorflow as tf


def stdev_cutoff(signal, num_stdev=3):
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

    return thresholded_tensor

def importances_stdev_cutoff(signal, num_stdev=3):
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
    signal_mean, signal_var = tf.nn.moments(signal, axes=[1, 2])
    signal_stdev = tf.sqrt(signal_var)
    thresholds = tf.add(signal_mean, tf.scalar_mul(num_stdev, signal_stdev))

    for dim_idx in range(1, len(signal_shape)):
        to_stack = [thresholds for i in range(signal_shape[dim_idx])] # here?
        thresholds = tf.stack(to_stack, dim_idx)

    # and threshold
    greaterthan_tensor = tf.cast(tf.greater(signal, thresholds), tf.float32)
    thresholded_tensor = signal * greaterthan_tensor

    #out_tensor = tf.transpose(tf.squeeze(thresholded_tensor), [0, 2, 1])

    return thresholded_tensor


def zscore(signal, num_stdev=3):
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
    signal_mean, signal_var = tf.nn.moments(signal, axes=[1, 2])
    signal_stdev = tf.sqrt(signal_var)
    thresholds = tf.add(signal_mean, tf.scalar_mul(num_stdev, signal_stdev))

    for dim_idx in range(1, len(signal_shape)):
        to_stack = [thresholds for i in range(signal_shape[dim_idx])] # here?
        thresholds = tf.stack(to_stack, dim_idx)

    # and threshold
    greaterthan_tensor = tf.cast(tf.greater(signal, thresholds), tf.float32)
    thresholded_tensor = signal * greaterthan_tensor

    # and zscore
    zscores = (thresholded_tensor - signal_mean) / signal_stdev # use median?
    
    #out_tensor = tf.transpose(tf.squeeze(thresholded_tensor), [0, 2, 1])

    return zscores



def normalize_to_probs(input_tensors, final_probs):
    """Given importance scores, normalize such that total weight
    is the final probability value (ie, think of if you had a total weight
    of 1, how should it be spread, and then weight that by the final
    probability value)
    """
    #weight_sums = tf.reduce_sum(input_tensors, axis=[1, 2], keep_dims=True)
    weight_sums = tf.reduce_sum(input_tensors, axis=[1, 2, 3], keep_dims=True)
    out_tensor = tf.multiply(
        tf.divide(input_tensors, weight_sums),
        tf.reshape(final_probs, weight_sums.get_shape()))
    
    return out_tensor
