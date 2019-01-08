"""Contains nets for quick thresholding
"""

import logging

import tensorflow as tf

from tronn.nets.util_nets import rebatch

from tronn.util.utils import DataKeys


def _get_threshold_on_null_distribution(tensor, pval_thresh):
    """get null distribution
    """
    # melt the tensor
    tensor_melted = tf.reshape(tensor, [-1])

    # get absolute value (two sided threshold)
    tensor_melted = tf.abs(tensor_melted)
    
    # get the val which will be threshold
    k_val = tf.multiply(
        pval_thresh,
        tensor_melted.get_shape().as_list()[0])
    k_val = tf.cast(k_val, tf.int32)
    
    # get threshold
    top_k_vals, top_k_indices = tf.nn.top_k(tensor_melted, k=k_val)
    threshold_val = top_k_vals[-1]

    return threshold_val


def build_null_distribution_threshold_fn(pval_thresh):
    """build fn to give to map_fn
    """
    def threshold_fn(tensor):
        return _get_threshold_on_null_distribution(tensor, pval_thresh)

    return threshold_fn


def threshold_shufflenull_OLD(inputs, params):
    """build distribution from the shuffles to get threshold
    # note that this is for sequence (or things that have a sequential order)
    """
    assert inputs.get(DataKeys.FEATURES) is not None
    assert params.get("shuffle_key") is not None
    
    # features
    features = inputs[DataKeys.FEATURES]
    shuffles = inputs[params["shuffle_key"]]
    outputs = dict(inputs)
    
    # adjust the shuffle key so that when running map_fn
    # you get a threshold for each example for each task
    shuffles = tf.transpose(shuffles, perm=[0,2,1,3,4]) # this is sequence specific
    shuffles = tf.concat(tf.unstack(shuffles, axis=0), axis=0) # {N*task, shuf, pos, 4}
    shuffles = tf.reduce_sum(shuffles, axis=3) # this is sequence specific
    
    # params
    pval_thresh = params.get("pval_thresh", 0.01) # for a seq of 1000bp, 10bp pass by chance
    threshold_fn = build_null_distribution_threshold_fn(pval_thresh)

    # get thresholds for each example, for each task
    thresholds = tf.map_fn(
        threshold_fn,
        shuffles)
    
    # and apply
    # note that first two dimensions are batch and task
    feature_shape = features.get_shape().as_list()
    thresholds = tf.reshape(
        thresholds,
        feature_shape[0:2] + [1 for i in xrange(len(feature_shape[2:]))])
    
    pass_positive_thresh = tf.cast(tf.greater(features, thresholds), tf.float32)
    pass_negative_thresh = tf.cast(tf.less(features, -thresholds), tf.float32)
    pass_thresh = tf.add(pass_positive_thresh, pass_negative_thresh)
    outputs[DataKeys.FEATURES] = tf.multiply(features, pass_thresh)

    # also save out the thresholds
    
    # also apply to the shuffles (used for pwm scoring)
    if params.get("process_shuffles", False):
        thresholds = tf.expand_dims(thresholds, axis=1)
        shuffles = inputs[params["shuffle_key"]]
        pass_positive_thresh = tf.cast(tf.greater(shuffles, thresholds), tf.float32)
        pass_negative_thresh = tf.cast(tf.less(shuffles, -thresholds), tf.float32)
        pass_thresh = tf.add(pass_positive_thresh, pass_negative_thresh)
        outputs[params["shuffle_key"]] = tf.multiply(shuffles, pass_thresh)
    
    return outputs, params



def get_threshold_mask_twotailed(inputs, params):
    """apply a threshold and return a boolean mask for things
    that pass the threshold in both the positive and negative directions
    """
    features = inputs[DataKeys.FEATURES]
    thresholds = tf.abs(inputs[params["thresholds_key"]]) # abs is just in case
    outputs = dict(inputs)

    # adjust thresholds internally as needed for different feature dims
    features_shape = features.get_shape().as_list()
    thresholds_shape = thresholds.get_shape().as_list()

    diff_dims = len(features_shape) - len(thresholds_shape)
    if diff_dims > 0:
        thresholds = tf.reshape(
            thresholds,
            thresholds_shape + [1 for i in xrange(diff_dims)])

    # apply two tailed
    pass_positive_thresh = tf.cast(tf.greater(features, thresholds), tf.float32)
    pass_negative_thresh = tf.cast(tf.less(features, -thresholds), tf.float32)
    pass_thresh = tf.add(pass_positive_thresh, pass_negative_thresh)
    outputs[DataKeys.FEATURES] = pass_thresh

    #logging.debug("RESULTS: {}".format(outputs[DataKeys.FEATURES].get_shape()))
    
    return outputs, params




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



def threshold_gaussian(inputs, params):
    """Given importance scores, calculates stdev cutoff
    and thresholds at that pval

    Args:
      signal: input tensor
      pval: the pval threshold

    Returns:
      out_tensor: output thresholded tensor
    """
    # get features and pass rest through
    features = inputs["features"]
    outputs = dict(inputs)

    # params
    num_stdev = params.get("stdev", 3)
    two_tailed = params.get("two_tailed", True)

    # separate out tasks first
    task_features_list = [
        tf.expand_dims(tensor, axis=1)
        for tensor in tf.unstack(features, axis=1)]
    
    thresholded = []
    for task_features in task_features_list:
    
        # get mean and stdev to get threshold
        signal_mean, signal_var = tf.nn.moments(task_features, axes=[1, 2, 3])
        signal_stdev = tf.sqrt(signal_var)
        thresholds = tf.add(signal_mean, tf.scalar_mul(num_stdev, signal_stdev))

        # expand thresholds for broadcasting
        signal_shape = task_features.get_shape()
        for dim_idx in xrange(1, len(signal_shape)):
            thresholds = tf.expand_dims(thresholds, axis=dim_idx)

        # set up mask
        if two_tailed:
            greaterthan_tensor = tf.cast(tf.greater_equal(task_features, thresholds), tf.float32)
            lessthan_tensor = tf.cast(tf.less_equal(task_features, -thresholds), tf.float32)
            mask_tensor = tf.add(greaterthan_tensor, lessthan_tensor)
        else:
            mask_tensor = tf.cast(tf.greater_equal(task_features, thresholds), tf.float32)
        
        # mask out insignificant features and append
        task_features = tf.multiply(task_features, mask_tensor)
        thresholded.append(task_features)
        
    # remerge
    features = tf.concat(thresholded, axis=1) #{N, task, pos, C}

    outputs["features"] = features
    
    return outputs, params



def add_per_example_kval(features, labels, config, is_training=False):
    """Extracts a per example max motifs val to pass onto threshold_topk_by_example
    """
    motif_len = tf.constant(config.get("motif_len", 10), tf.float32)
    max_k = config.get("max_k", 4)

    # split features by task
    features_by_task = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)] # list of {N, 1, pos, C}
    
    all_k_vals = []
    for task_features in features_by_task:
        num_motifs = tf.divide(
            tf.reduce_sum(
                tf.cast(tf.not_equal(task_features, 0), tf.float32),
                axis=[1,2,3]),
            motif_len) # {N, 1}
        num_motifs = tf.minimum(num_motifs, max_k) # heuristic for now
        num_motifs_list = tf.unstack(num_motifs) # list of [1], len is N
        all_k_vals.append(num_motifs_list)
        
    config["per_example_kvals"] = all_k_vals

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
    k_val = config.get("k_val", [10 for i in xrange(len(features))])
    
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
    
    # sometimes there are noisy scores, leading to many matches, zero them out
    #features_present = tf.cast(tf.not_equal(features, 0), tf.float32) # {N, 1, pos, motif}
    #features_counts = tf.reduce_sum(features_present, axis=[1, 2, 3], keep_dims=True) # {N, 1, 1, 1}
    #features_kval_threshold = tf.cast(
     #   tf.less_equal(features_counts, tf.reshape(tf.stack(k_val), features_counts.get_shape())),
     #   tf.float32)
    #features = tf.multiply(features_kval_threshold, features)
    
    # TODO: set up to be counts?
    #features = tf.cast(tf.not_equal(features, 0), tf.float32) # {N, 1, pos, motif}

    # and reduce
    #features = tf.squeeze(tf.reduce_sum(features, axis=position_axis)) # {N, motif}
    features = tf.squeeze(features)
    
    return features, labels, config


def multitask_threshold_topk_by_example(features, labels, config, is_training=False):
    """Split into tasks and then call threshold_by_topk_example
    """
    assert is_training == False

    task_axis = 1
    per_example_kvals = config.get("per_example_kvals")
    
    # unstack features by task
    features = [tf.expand_dims(tensor, axis=task_axis) for tensor in tf.unstack(features, axis=task_axis)] # {N, 1, pos, M}
    
    # run through thresholding
    thresholded = []
    for task_idx in xrange(len(features)):
        if per_example_kvals is not None:
            config["k_val"] = per_example_kvals[task_idx]
        task_features_thresholded, labels, config = threshold_topk_by_example(
            features[task_idx], labels, config, is_training=is_training)
        thresholded.append(tf.expand_dims(task_features_thresholded, axis=task_axis))
    features = tf.concat(thresholded, axis=task_axis)

    return features, labels, config



def clip_edges(inputs, params):
    """Grab just the middle base pairs
    """
    assert inputs.get(DataKeys.FEATURES) is not None

    # get features and pass rest through
    features = inputs[DataKeys.FEATURES]
    outputs = dict(inputs)

    left_clip = params.get("left_clip", 0)
    right_clip = params.get("right_clip", features.get_shape().as_list()[2])
    outputs[DataKeys.FEATURES] = features[:,:,left_clip:right_clip,:]
    
    # adjust other seq as necessary
    if inputs.get(DataKeys.ORIG_SEQ) is not None:
        outputs[DataKeys.ORIG_SEQ_ACTIVE] = inputs[DataKeys.ORIG_SEQ][:,:,left_clip:right_clip,:]

    if inputs.get(DataKeys.ORIG_SEQ_SHUF) is not None:
        outputs[DataKeys.ORIG_SEQ_ACTIVE_SHUF] = inputs[DataKeys.ORIG_SEQ_SHUF][:,:,:,left_clip:right_clip,:]
        
    if inputs.get(DataKeys.WEIGHTED_SEQ) is not None:
        outputs[DataKeys.WEIGHTED_SEQ_ACTIVE] = inputs[DataKeys.WEIGHTED_SEQ][:,:,left_clip:right_clip,:]

    if inputs.get(DataKeys.WEIGHTED_SEQ_SHUF) is not None:
        outputs[DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF] = inputs[DataKeys.WEIGHTED_SEQ_SHUF][:,:,:,left_clip:right_clip,:]
        
    # fix this?
    if params.get("clip_string") is not None:
        outputs["features.string"] = tf.substr(
            outputs["features.string"],
            left_clip,
            right_clip - left_clip)
        
    return outputs, params
