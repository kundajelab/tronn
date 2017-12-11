"""Contains nets for quick thresholding
"""

import tensorflow as tf


# TODO(dk)


def threshold_shufflenull(features, labels, config, is_training=False):
    """Shuffle values to get a null distribution at each position
    """
    assert is_training == False
    print config
    num_shuffles = config.get("num_shuffles", 100)
    k_val = int(num_shuffles*config.get("pval", 0.05)) # too strict?
    two_tailed = config.get("two_tailed", False)
    
    # separate out tasks first and reduce to get importance on one axis
    task_list = tf.unstack(tf.reduce_sum(features, axis=3), axis=1)
    
    task_thresholds = []
    for task_features in task_list:

        # then separate out examples
        examples_list = tf.unstack(task_features) # list of {seq_len}
        threshold_masks = []
        for example in examples_list:
            #example_reduced = tf.reduce_sum(example, axis=1)
            # shuffle
            shuffles = []
            for i in xrange(num_shuffles):
                shuffles.append(tf.random_shuffle(example))
            all_shuffles = tf.stack(shuffles, axis=1) # {seq_len, 100}
            
            # get top k
            top_k_vals, top_k_indices = tf.nn.top_k(all_shuffles, k=k_val)
            thresholds = tf.reshape(tf.reduce_min(top_k_vals, axis=1, keep_dims=True), [example.get_shape().as_list()[0]])
            greaterthan_mask = tf.cast(tf.greater_equal(example, thresholds), tf.float32) # {seq_len, 4}
            if two_tailed:
                top_k_vals, top_k_indices = tf.nn.top_k(-all_shuffles, k=k_val)
                thresholds = tf.reshape(tf.reduce_min(top_k_vals, axis=1, keep_dims=True), [example.get_shape().as_list()[0]])
                lessthan_mask = tf.cast(tf.less_equal(example, -thresholds), tf.float32) # {seq_len, 4}
                greaterthan_mask = tf.add(greaterthan_mask, lessthan_mask)

            # just keep masks and broadcast to channels on the original features
            threshold_masks.append(greaterthan_mask)

        # stack
        threshold_mask = tf.stack(threshold_masks, axis=0)
        task_thresholds.append(threshold_mask)

    # stack
    threshold_mask = tf.expand_dims(tf.stack(task_thresholds, axis=1), axis=3)
    features = tf.multiply(features, threshold_mask)
        
    return features, labels, config


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


def add_per_example_kval(features, labels, config, is_training=False):
    """Extracts a per example max motifs val to pass onto threshold_topk_by_example
    """
    motif_len = tf.constant(config.get("motif_len", 5), tf.float32)
    max_hits = config.get("k_val", 4)
    
    num_motifs = tf.divide(
        tf.reduce_sum(
            tf.cast(tf.not_equal(features, 0), tf.float32),
            axis=[1,2,3]),
        motif_len) # {N, 1}
    num_motifs = tf.minimum(num_motifs, max_hits) # heuristic for now
    num_motifs_list = tf.unstack(num_motifs)

    config["k_val"] = num_motifs_list
    
    return features, labels, config


def threshold_topk_by_example(features, labels, config, is_training=False):
    """Get top k hits across all axes
    """
    assert is_training == False
    splitting_axis = config.get("splitting_axis", 0)
    position_axis = config.get("position_axis", 2)
    #two_tailed = config.get("two_tailed", True)

    # separate by the axis desired
    features = [tf.expand_dims(tensor, axis=splitting_axis)
                for tensor in tf.unstack(features, axis=splitting_axis)] # {1, 1, pos, M}
    k_val = config.get("k_val", [4 for i in xrange(len(features))])
    
    # grab the top k and determine threshold val
    features_topk = []
    for i in xrange(len(features)):
        top_k_vals, top_k_indices = tf.nn.top_k(
            tf.reshape(tf.abs(features[i]), [-1]), # -1 flattens into 1D
            k=tf.cast(k_val[i], tf.int32))
        threshold = tf.reduce_min(top_k_vals, keep_dims=True)
        
        # threshold both pos and neg
        greaterthan_w_location = tf.cast(tf.greater_equal(features[i], threshold), tf.float32)
        lessthan_w_location = tf.cast(tf.less_equal(features[i], -threshold), tf.float32)
        threshold_mask = tf.add(greaterthan_w_location, lessthan_w_location)

        # and mask
        top_scores_w_location = tf.multiply(features[i], threshold_mask) # {1, 1, pos, motif}
        features_topk.append(top_scores_w_location)

    # and restack
    features = tf.concat(features_topk, axis=splitting_axis) # {N, 1, pos, motif}

    # and reduce
    features = tf.squeeze(tf.reduce_sum(features, axis=position_axis)) # {N, motif}

    return features, labels, config


def multitask_threshold_topk_by_example(features, labels, config, is_training=False):
    """Split into tasks and then call threshold_by_topk_example
    """
    assert is_training == False

    task_axis = 1

    # unstack features by task
    features = [tf.expand_dims(tensor, axis=task_axis) for tensor in tf.unstack(features, axis=task_axis)] # {N, 1, pos, M}
    
    # run through thresholding
    thresholded = []
    for task_features in features:
        task_features_thresholded, labels, config = threshold_topk_by_example(
            task_features, labels, config, is_training=is_training)
        thresholded.append(tf.expand_dims(task_features_thresholded, axis=task_axis))
    features = tf.concat(thresholded, axis=task_axis)

    return features, labels, config


def clip_edges(features, labels, config, is_training=False):
    """Grab just the middle base pairs
    """
    assert is_training == False

    left_clip = config.get("left_clip", 0)
    right_clip = config.get("right_clip", features.get_shape().as_list()[2])
    features = features[:,:,left_clip:right_clip,:]

    return features, labels, config
