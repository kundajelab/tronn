"""Contains nets that perform PWM convolutions
"""

import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.initializers import pwm_simple_initializer
from tronn.util.tf_utils import get_fan_in


def pwm_convolve_new(features, labels, config, is_training=False):
    """Inspired by Av's jaccard-like distance
    """
    # jaccard: essentially intersection over union
    # use the jaccard distance to weight the matches.
    # ie, score = sum(importances matching pwm) * jaccard distance
    pwm_list = config.get("pwms")
    assert pwm_list is not None
        
    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)
    logging.info("Total PWMs: {}".format(num_filters))
    
    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]
    logging.info("Filter size: {}".format(max_size))

    # first, jaccard

    # get |a intersect b| using convolutional net
    # define |a intersect b| as approx min of convolving bool(a) and b, but with ones
    # this way you capture the intersect by presence of specific base pairs
    
    # convolve pwms with binarized features, [-1, 1]
    binarized_features = tf.add(
        tf.cast(tf.greater(features, 0), tf.float32),
        -tf.cast(tf.less(features, 0), tf.float32))
    conv1_filter_size = [1, max_size]
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        # pwm cross correlation
        jaccard_intersection = slim.conv2d(
            binarized_features, num_filters, conv1_filter_size,
            scope='pwm/conv')
        jaccard_intersection = tf.abs(jaccard_intersection)

    # and get summed importance in that window
    summed_feature_impt = tf.reduce_sum(
        tf.multiply(
            slim.avg_pool2d(features, [1, max_size], stride=[1,1], padding="VALID"),
            max_size),
        axis=3, keep_dims=True)
    print summed_feature_impt.get_shape()

    # numerator: summed importance * jaccard intersection score.
    numerator = tf.multiply(summed_feature_impt, jaccard_intersection)
    print numerator.get_shape()
    
    # union: |a| + |b| - |a intersect b|
    # getting |input| - max pool on binarized features
    # getting |pwm| - convolve on full sequence
    summed_feature_presence = tf.reduce_sum(
        tf.multiply(
            slim.avg_pool2d(tf.abs(binarized_features), [1, max_size], stride=[1,1], padding="VALID"),
            max_size),
        axis=3, keep_dims=True)

    # make another convolution net for the normalization factor, with squared filters
    ones_input = tf.ones(features.get_shape())
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features), squared=True), # <- this is why you can't merge
            biases_initializer=None,
            trainable=False):
        # normalization factor
        summed_pwm_squared = slim.conv2d(
            ones_input, num_filters, conv1_filter_size,
            scope='nonzero_pwm/conv')
        summed_pwm_abs = tf.sqrt(summed_pwm_squared)

    # denominator: |input| + |pwm| - intersection
    denominator = tf.subtract(tf.add(summed_feature_presence, summed_pwm_abs), jaccard_intersection)
        
    # divide the two answers.
    final_score = tf.divide(numerator, denominator)
    
    return final_score, labels, config


def pwm_convolve_old(features, labels, config, is_training=False):
    """Convolve with PWMs and normalize with vector projection:

      projection = a dot b / | b |
    """
    pwm_list = config.get("pwms")
    assert pwm_list is not None
        
    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)
    logging.info("Total PWMs: {}".format(num_filters))
    
    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]
    logging.info("Filter size: {}".format(max_size))
            
    # make the convolution net for dot product, normal filters
    conv1_filter_size = [1, max_size]
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        # pwm cross correlation
        pwm_scores = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='pwm/conv')

    # make another convolution net for the normalization factor, with squared filters
    nonzero_features = tf.cast(tf.not_equal(features, 0), tf.float32)
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features), squared=True), # <- this is why you can't merge
            biases_initializer=None,
            trainable=False):
        # normalization factor
        nonzero_squared_vals = slim.conv2d(
            nonzero_features, num_filters, conv1_filter_size,
            scope='nonzero_pwm/conv')
        nonzero_vals = tf.sqrt(nonzero_squared_vals)
    
    # and then normalize using the vector projection formulation
    # todo this is probably screwing with the vectors
    pseudocount = 0.00000001
    features = tf.divide(pwm_scores, tf.add(nonzero_vals, pseudocount)) # {N, task, seq_len, M}

    return features, labels, config


def pwm_convolve(features, labels, config, is_training=False):
    """Convolve with PWMs and normalize with vector projection:

      projection = a dot b / | b |
    """
    pwm_list = config.get("pwms")
    reuse = config.get("reuse_pwm_layer", False)
    # TODO options for running (1) FWD, (2) REV, (3) COMB
    
    assert pwm_list is not None
        
    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)
    #logging.info("Total PWMs: {}".format(num_filters))
    
    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]
    #logging.info("Filter size: {}".format(max_size))
    config["filter_width"] = max_size
            
    # make the convolution net for dot product, normal filters
    # here, the pwms are already normalized to unit vectors for vector projection
    conv1_filter_size = [1, max_size]
    with tf.variable_scope("pwm_layer", reuse=reuse):
        with slim.arg_scope(
                [slim.conv2d],
                padding='VALID',
                activation_fn=None,
                weights_initializer=pwm_simple_initializer(
                    conv1_filter_size,
                    pwm_list,
                    get_fan_in(features),
                    unit_vector=True,
                    length_norm=False),
                biases_initializer=None,
                trainable=False):
            # pwm cross correlation
            features = slim.conv2d(
                features, num_filters, conv1_filter_size)
        
    return features, labels, config


def pwm_convolve_inputxgrad(features, labels, config, is_training=False):
    """Convolve both pos and negative scores with PWMs. Prevents getting scores
    when the negative correlation is stronger.
    """
    # do positive sequence. ignore negative scores. only keep positive results
    #with tf.variable_scope("pos_seq_pwm"):
    pos_seq_scores, _, _ = pwm_convolve(tf.nn.relu(features), labels, config, is_training=is_training)
    pos_seq_scores = tf.nn.relu(pos_seq_scores)
        
    # do negative sequence. ignore positive scores. only keep positive (ie negative) results
    #with tf.variable_scope("neg_seq_pwm"):
    config.update({"reuse_pwm_layer": True})
    neg_seq_scores, _, _ = pwm_convolve(tf.nn.relu(-features), labels, config, is_training=is_training)
    neg_seq_scores = tf.nn.relu(neg_seq_scores)
        
    # and then take max (best score, whether pos or neg) do not use abs; and keep sign
    max_seq_scores = tf.reduce_max(tf.stack([pos_seq_scores, neg_seq_scores], axis=0), axis=0) # {N, task, seq_len/pool_width, M}
    
    # and now get the sign and mask
    pos_scores_masked = tf.multiply(
        pos_seq_scores,
        tf.cast(tf.equal(pos_seq_scores, max_seq_scores), tf.float32))
    neg_scores_masked = tf.multiply(
        -neg_seq_scores,
        tf.cast(tf.equal(neg_seq_scores, max_seq_scores), tf.float32))

    features = tf.add(pos_scores_masked, neg_scores_masked)
    
    return features, labels, config


def pwm_motif_max(features, labels, config, is_training=False):
    """Get max for each motif? this likely reduces noise, but is it necessary?
    # is this sorta subsumed by position squeeze later?
    """
    features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)] # list of {N, 1, pos, M}

    # TODO build a function to filter for two sided max?
    features_pos_max = []
    for i in xrange(len(features)):
        task_features = features[i]
        # fix this? is wrong?
        features_max_vals = tf.reduce_max(tf.abs(task_features), axis=2, keep_dims=True) # {N, 1, 1, M}
        features_max_mask = tf.multiply(
            tf.cast(tf.equal(tf.abs(task_features), features_max_vals), tf.float32),
            tf.cast(tf.not_equal(task_features, 0), tf.float32))
        task_features = tf.multiply(task_features, features_max_mask)
        features_pos_max.append(task_features)
        
    # restack
    features = tf.concat(features_pos_max, axis=1) # {N, task, pos, M}

    return features, labels, config


def get_bp_overlap(features, labels, config, is_training=False):
    """Re-weight by num of importance-weighted base pairs that are nonzero
    """
    features_present = tf.cast(tf.not_equal(features, [0]), tf.float32)
    max_size = config.get("filter_width")
    assert max_size is not None
    nonzero_bp_fraction_per_window = tf.reduce_sum(
        slim.avg_pool2d(
            features_present, [1, max_size], stride=[1,1], padding="VALID"),
        axis=3, keep_dims=True)
    #features = tf.multiply(
    #    features,
    #    nonzero_bp_fraction_per_window)
    
    return nonzero_bp_fraction_per_window, labels, config


def pwm_match_filtered_convolve(features, labels, config, is_training=False):
    """Run pwm convolve twice, with importance scores and without.
    Choose max for motif across positions using raw sequence
    """
    raw_sequence = config["outputs"].get("onehot_sequence_clipped")
    #reuse = config.get("reuse_pwm_layer", False)
    assert raw_sequence is not None

    # run on raw sequence
    if raw_sequence is not None:
        binarized_features = raw_sequence
        if config["keep_ism_results"] is None:
            del config["outputs"]["onehot_sequence"] # remove this from outputs now
        else:
            print "WARNING DID NOT DELETE RAW SEQUENCE"
    
    #with tf.variable_scope("binarize_filt"):
    pwm_binarized_feature_scores, _, _ = pwm_convolve_inputxgrad(
        binarized_features, labels, config, is_training=is_training) # {N, 1, pos, M}

    # adjust the raw scores and save out
    if config.get("keep_pwm_raw_scores") is not None:
        raw_bp_overlap, _, _ = get_bp_overlap(binarized_features, labels, config)
        raw_scores = tf.multiply(
            pwm_binarized_feature_scores,
            raw_bp_overlap)
        raw_scores = tf.squeeze(tf.reduce_max(raw_scores, axis=2)) # {N, M}
        config["outputs"][config["keep_pwm_raw_scores"]] = raw_scores

    # multiply by raw sequence matches
    pwm_binarized_feature_maxfilt_mask = tf.cast(
        tf.greater(pwm_binarized_feature_scores, [0]), tf.float32)
    
    # run on impt weighted features
    #with tf.variable_scope("impt_weighted"):
    pwm_impt_weighted_scores, _, _ = pwm_convolve_inputxgrad(
        features, labels, config, is_training=is_training)
    
    # and filter through mask
    filt_features = tf.multiply(
        pwm_binarized_feature_maxfilt_mask,
        pwm_impt_weighted_scores)

    # at this stage also need to perform the weighting by bp presence
    impt_bp_overlap, _, _ = get_bp_overlap(features, labels, config)
    features = tf.multiply(
        filt_features,
        impt_bp_overlap)

    # keep for grammars
    if config.get("keep_pwm_scores_full") is not None:
        # attach to config
        config["outputs"][config["keep_pwm_scores_full"]] = features # {N, task, pos, M}
        
    return features, labels, config


def pwm_maxpool(features, labels, config, is_training=False):
    """Two tailed pooling operation when have both pos/neg scores
    """
    pool_width = config.get("pool_width", None)
    assert pool_width is not None

    # testing
    # big maxpool across 15 pwm width, but per position

    # end testing

    
    # get the max vals, both pos and neg, VALID padding
    maxpool_pos = slim.max_pool2d(features, [1, pool_width], stride=[1, pool_width]) # {N, task, seq_len/pool_width, M}
    maxpool_neg = slim.max_pool2d(-features, [1, pool_width], stride=[1, pool_width]) # {N, task, seq_len/pool_width, M}
    maxpool_abs = tf.reduce_max(tf.stack([maxpool_pos, maxpool_neg], axis=0), axis=0) # {N, task, seq_len/pool_width, M}
    
    # get the right values
    maxpool_pos_masked = tf.multiply(
        maxpool_pos,
        tf.cast(tf.equal(maxpool_pos, maxpool_abs), tf.float32))
    maxpool_neg_masked = tf.multiply(
        -maxpool_neg,
        tf.cast(tf.equal(maxpool_neg, maxpool_abs), tf.float32))
    features = tf.add(maxpool_pos_masked, maxpool_neg_masked)

    return features, labels, config


# TODO more correctly, need to set up a check so that a motif across time is only kept if seen at least
# twice across time too


def pwm_consistency_check(features, labels, config, is_training=False):
    """Try to keep most consistent motifs across tasks. max scores are accounted for later
    """

    # TODO is this useful? check multitask global importance to see thresholding
    
    # split by example
    features_by_example = [tf.expand_dims(tensor, axis=0) for tensor in tf.unstack(features)] # {1, task, pos, M}

    # for each example, get sum across tasks
    masked_features_list = []
    for example_features in features_by_example:
        motif_present = tf.cast(tf.not_equal(example_features, 0), tf.float32) # {1, task, pos, M}

        # TODO: probably just want max sum across the real scores, not counts.
        motif_counts = tf.reduce_sum(
            motif_present, axis=1, keep_dims=True) # sum across tasks {1, 1, pos, M}
        #motif_counts = tf.reduce_sum(
        #    tf.abs(example_features), axis=1, keep_dims=True) # sum across tasks {1, 1, pos, M}
        motif_max = tf.reduce_max(motif_counts, axis=3, keep_dims=True) # max across motifs {1, 1, pos, 1}
        # then mask based on max position
        motif_mask = tf.cast(tf.greater_equal(motif_counts, motif_max), tf.float32) # {1, 1, pos, M}
        # and mask
        masked_features = tf.multiply(motif_mask, example_features)
        masked_features_list.append(masked_features)

    # stack
    features = tf.concat(masked_features_list, axis=0) # {N, task, pos, M}

    # sometimes keep (for grammars)
    if config.get("keep_pwm_scores_full") is not None:
        # attach to config
        config["outputs"][config["keep_pwm_scores_full"]] = features # {N, task, pos, M}

    return features, labels, config


def pwm_positional_max(features, labels, config, is_training=False):
    """Get max at a position
    """
    features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)] # list of {N, 1, pos, M}

    # TODO build a function to filter for two sided max?
    features_pos_max = []
    for i in xrange(len(features)):
        task_features = features[i]
        # fix this? is wrong?
        features_max_vals = tf.reduce_max(tf.abs(task_features), axis=3, keep_dims=True) # {N, 1, pos, 1}
        features_max_mask = tf.multiply(
            tf.cast(tf.equal(tf.abs(task_features), features_max_vals), tf.float32),
            tf.cast(tf.not_equal(task_features, 0), tf.float32))
        task_features = tf.multiply(task_features, features_max_mask)
        features_pos_max.append(task_features)
        
    # restack
    features = tf.concat(features_pos_max, axis=1) # {N, task, pos, M}

    return features, labels, config


def pwm_position_squeeze(features, labels, config, is_training=False):
    """Squeeze position
    """
    squeeze_type = config.get("squeeze_type", "max")
    if squeeze_type == "max":
        features = tf.reduce_max(features, axis=2) # features {N, task, pos, M}
    elif squeeze_type == "mean":
        features = tf.reduce_mean(features, axis=2)
    elif squeeze_type == "sum":
        features = tf.reduce_sum(features, axis=2)

    return features, labels, config


def pwm_relu(features, labels, config, is_training=False):
    """Only keep positive
    """
    features = tf.nn.relu(features)
    
    return features, labels, config


def pwm_convolve_v3(features, labels, config, is_training=False):
    """Convolve with PWMs and normalize with vector projection:

      projection = a dot b / | b |
    """
    pwm_list = config.get("pwms")
    pool = config.get("pool", False)
    pool_width = config.get("pool_width", None)
    assert pwm_list is not None
    if pool is not None:
        assert pool_width is not None
    positional_max = config.get("positional_max", False)
        
    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)
    logging.info("Total PWMs: {}".format(num_filters))

    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]
    logging.info("Filter size: {}".format(max_size))
            
    # make the convolution net for dot product, normal filters
    conv1_filter_size = [1, max_size]
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        # pwm cross correlation
        pwm_scores = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='pwm/conv')

    # make another convolution net for the normalization factor, with squared filters
    nonzero_features = tf.cast(tf.not_equal(features, 0), tf.float32)
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features), squared=True), # <- this is why you can't merge
            biases_initializer=None,
            trainable=False):
        # normalization factor
        nonzero_squared_vals = slim.conv2d(
            nonzero_features, num_filters, conv1_filter_size,
            scope='nonzero_pwm/conv')
        nonzero_vals = tf.sqrt(nonzero_squared_vals)
    
    # and then normalize using the vector projection formulation
    # todo this is probably screwing with the vectors
    pseudocount = 0.00000001
    features = tf.divide(pwm_scores, tf.add(nonzero_vals, pseudocount)) # {N, task, seq_len, M}

    # max pool if requested - loses the negatives? yup
    # max pool and min pool and then take the larger val?
    
    # first, need to convolve with abs scores. that way, you can only keep the top scores.
    # once you know those, make a mask
    # then need to reconvolve with the raw scores. then treat with the mask
    # this gives positive and negative motifs correctly.
    if pool:
        # get the max vals, both pos and neg
        maxpool_pos = slim.max_pool2d(features, [1, pool_width], stride=[1, pool_width]) # {N, task, seq_len/pool_width, M}
        maxpool_neg = slim.max_pool2d(-features, [1, pool_width], stride=[1, pool_width]) # {N, task, seq_len/pool_width, M}
        maxpool_abs = tf.reduce_max(tf.abs(tf.stack([maxpool_pos, maxpool_neg], axis=0)), axis=0) # {N, task, seq_len/pool_width, M}

        # get the right values
        maxpool_pos_masked = tf.multiply(
            maxpool_pos,
            tf.cast(tf.greater_equal(maxpool_pos, maxpool_abs), tf.float32))
        maxpool_neg_masked = tf.multiply(
            -maxpool_neg,
            tf.cast(tf.less_equal(-maxpool_neg, -maxpool_abs), tf.float32))
        features = tf.add(maxpool_pos_masked, maxpool_neg_masked)
        
    # only keep max at each position if requested, separately for each task
    if positional_max:
        features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)] # list of {N, 1, pos, M}
        
        features_pos_max = []
        for i in xrange(len(features)):
            task_features = features[i]
            features_max_vals = tf.reduce_max(tf.abs(task_features), axis=3, keep_dims=True) # {N, 1, pos, 1}
            features_max_mask = tf.multiply(
                tf.add(
                    tf.cast(tf.greater_equal(task_features, features_max_vals), tf.float32),
                    tf.cast(tf.less_equal(task_features, -features_max_vals), tf.float32)), # add two sided threshold masks
                tf.cast(tf.not_equal(task_features, 0), tf.float32)) # and then make sure none are zero. {N, 1, pos, motif}
            
            task_features = tf.multiply(task_features, features_max_mask)
            features_pos_max.append(task_features)
        # restack
        features = tf.concat(features_pos_max, axis=1) # {N, task, pos, M}
        
    return features, labels, config


def motif_assignment(features, labels, model_params, is_training=False):
    """This specifically takes in features and then tries to match to only one motif
    It does an approximation check to choose how many motifs it believes should be assigned
    """
    # get params
    pwm_list = model_params["pwms"]
    assert pwm_list is not None
    pool = model_params.get("pool", False)
    max_hits = model_params.get("k_val", 4)
    motif_len = tf.constant(model_params.get("motif_len", 5), tf.float32)

    # approximation: check num important base pairs per example, and divide by motif len 
    num_motifs = tf.divide(
        tf.reduce_sum(
            tf.cast(tf.not_equal(features, 0), tf.float32), 
            axis=[1,2,3]),
        motif_len) # {N, 1}
    num_motifs = tf.minimum(num_motifs, max_hits) # heuristic for now
    num_motifs_list = tf.unstack(num_motifs)

    # convolve with PWMs
    pwm_scores = pwm_convolve_v3(features, labels, {"pwms": pwm_list, "pool": pool}) # {N, 1, pos, motif}
    
    # max pool - this accounts for hits that are offset because 
    # the motifs are not aligned to each other
    #pwm_scores_pooled = slim.max_pool2d(pwm_scores, [1, 10], stride=[1, 10])

    # grab abs val max at each position
    pwm_scores_max_vals = tf.reduce_max(tf.abs(pwm_scores), axis=3, keep_dims=True) # {N, 1, pos, 1}

    # then only keep the max at each position. multiply by conditional on > 0 to keep clean
    # TODO - change this thresholding, to keep max val (whether pos or negative)
    pwm_max_mask = tf.multiply(
        tf.add(
            tf.cast(tf.greater_equal(pwm_scores, pwm_scores_max_vals), tf.float32),
            tf.cast(tf.less_equal(pwm_scores, -pwm_scores_max_vals), tf.float32)), # add two sided threshold masks
        tf.cast(tf.not_equal(pwm_scores, 0), tf.float32)) # and then make sure none are zero. {N, 1, pos, motif}
    pwm_scores_max = tf.multiply(pwm_scores, pwm_max_mask)

    # separate into each example
    pwm_scores_max_list = tf.unstack(pwm_scores_max) # list of {1, pos, motif}
    print tf.reshape(pwm_scores_max[0], [-1]).shape

    # and then top k - TODO factor out
    pwm_scores_topk = []
    for i in xrange(len(num_motifs_list)):
        top_k_vals, top_k_indices = tf.nn.top_k(tf.reshape(tf.abs(pwm_scores_max_list[i]), [-1]), k=tf.cast(num_motifs_list[i], tf.int32))
        thresholds = tf.reduce_min(top_k_vals, keep_dims=True)
        
        # threshold both pos and neg
        greaterthan_w_location = tf.cast(tf.greater_equal(pwm_scores_max_list[i], thresholds), tf.float32) # this is a threshold, so a count
        lessthan_w_location = tf.cast(tf.less_equal(pwm_scores_max_list[i], -thresholds), tf.float32) # this is a threshold, so a count
        threshold_mask = tf.add(greaterthan_w_location, lessthan_w_location)
        
        # and mask
        top_scores_w_location = tf.multiply(pwm_scores_max_list[i], threshold_mask) # {1, pos, motif}
        pwm_scores_topk.append(top_scores_w_location)

    # and restack
    pwm_final_scores = tf.stack(pwm_scores_topk) # {N, 1, pos, motif}

    # and reduce
    pwm_final_counts = tf.squeeze(tf.reduce_sum(pwm_final_scores, axis=2)) # {N, motif}

    return pwm_final_counts, labels, model_params


def multitask_motif_assignment(features, labels, config, is_training=False):
    """Multitask motif assignment
    """
    features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)]

    motif_assignments = []
    for i in xrange(len(features)):
        # TODO(dk) give unique PWM names for weights
        with tf.variable_scope("task_{}".format(i)):
            task_features, _, _ = motif_assignment(features[i], labels, config)
            motif_assignments.append(task_features)
    features = tf.stack(motif_assignments, axis=1)

    return features, labels, config


def featurize_motifs(features, pwm_list=None, is_training=False):
    '''
    All this model does is convolve with PWMs and get top k pooling to output
    a example by motif matrix.
    '''
    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)

    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]

    # make the convolution net
    conv1_filter_size = [1, max_size]
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        net = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='conv1/conv')

    # Then get top k values across the correct axis
    net = tf.transpose(net, perm=[0, 1, 3, 2])
    top_k_val, top_k_indices = tf.nn.top_k(net, k=3)

    # Do a summation
    motif_tensor = tf.squeeze(tf.reduce_sum(top_k_val, 3)) # 3 is the axis

    return motif_tensor


# get rid of all these below when clear it's not used


def pwm_convolve_old(features, labels, pwm_list):
    '''
    All this model does is convolve with PWMs and get top k pooling to output
    a example by motif matrix.
    '''

    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)

    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]

    # make the convolution net
    with slim.arg_scope([slim.conv2d], padding='VALID',
                        activation_fn=None, trainable=False):
        conv1_filter_size = [1, max_size]
        net = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='conv1/conv')

    # Then get top k values across the correct axis
    net = tf.transpose(net, perm=[0, 1, 3, 2])
    top_k_val, top_k_indices = tf.nn.top_k(net, k=3)

    # Do a summation
    motif_tensor = tf.squeeze(tf.reduce_sum(top_k_val, 3)) # 3 is the axis

    # Then adjust the filters by putting in PWM info
    # note that there should actually only be 1 set of weights, the first layer
    weights = [v for v in tf.global_variables() if ('weights' in v.name)] 
    weights_list = []
    for i in range(len(pwm_list)):
        pwm = pwm_list[i]
        pad_length = max_size - pwm.weights.shape[1]
        padded_weights = np.concatenate((pwm.weights,
                                         np.zeros((4, pad_length))),
                                        axis=1)
        weights_list.append(padded_weights)

    # stack into weights tensor and assign
    pwm_all_weights = np.stack(weights_list, axis=0).transpose(2, 1, 0)
    pwm_np_tensor = np.expand_dims(pwm_all_weights, axis=0)
    load_pwm_update = weights[0].assign(pwm_np_tensor)

    return motif_tensor, load_pwm_update


def pwm_convolve_v2(features, labels, model_params, is_training=False):
    '''
    All this model does is convolve with PWMs and get top k pooling to output
    a example by motif matrix.
    '''

    pwm_list = model_params["pwms"]

    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)

    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]

    # make the convolution net
    conv1_filter_size = [1, max_size]
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        net = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='conv1/conv')

    # Then get top k values across the correct axis
    net = tf.transpose(net, perm=[0, 1, 3, 2])
    top_k_val, top_k_indices = tf.nn.top_k(net, k=3)

    # Do a summation
    motif_tensor = tf.squeeze(tf.reduce_sum(top_k_val, 3)) # 3 is the axis

    return motif_tensor


def top_motifs_w_distances(features, pwm_list, top_k_val=2):
    '''
    This extracts motif scores with associated distances
    '''

    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)

    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]

    # make the convolution net
    with slim.arg_scope([slim.conv2d], padding='VALID',
                        activation_fn=None, trainable=False):
        conv1_filter_size = [1, max_size]
        net = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='conv1/conv')

    # Then get top k values across the correct axis
    net = tf.squeeze(tf.transpose(net, perm=[0, 1, 3, 2]))
    mat_topkval, mat_topkval_indices = tf.nn.top_k(net, k=top_k_val)
    
    # Get mean and var to do a zscore on the scores for each sequence
    seq_mean, seq_var = tf.nn.moments(mat_topkval, [1, 2])

    # Extra operations because broadcasting is finicky
    mat_mean_intermediate = tf.stack([seq_mean for i in range(num_filters)], axis=1)
    mat_mean = tf.stack([mat_mean_intermediate for i in range(top_k_val)], axis=2)

    mat_var_intermediate = tf.stack([seq_var for i in range(num_filters)], axis=1)
    mat_var = tf.stack([mat_var_intermediate for i in range(top_k_val)], axis=2)

    mat_topkval_zscore = tf.multiply(tf.subtract(mat_topkval, mat_mean), tf.rsqrt(mat_var))
    
    # TOP SCORES: reshape, outer product, reshape (NOTE: broadcasting not fully working in this tf version)
    mat1_topkval_motif_x_motif = tf.stack([mat_topkval_zscore for i in range(num_filters)], axis=2)
    mat1_topkval_full = tf.stack([mat1_topkval_motif_x_motif for i in range(top_k_val)], axis=4)
    mat2_topkval_motif_x_motif = tf.stack([mat_topkval_zscore for i in range(num_filters)], axis=1)
    mat2_topkval_full = tf.stack([mat2_topkval_motif_x_motif for i in range(top_k_val)], axis=3)

    motif_x_motif_scores = tf.multiply(mat1_topkval_full, mat2_topkval_full)
    score_dims = motif_x_motif_scores.get_shape().as_list()
    new_dims = score_dims[:-2] + [score_dims[-2] * score_dims[-1]]
    motif_x_motif_scores_redux = tf.reshape(motif_x_motif_scores, new_dims)
    print "Motif score matrix dims:", motif_x_motif_scores_redux.get_shape()

    # TOP INDICES: reshape, outer product, reshape
    mat1_topkval_idx_x_idx = tf.stack([mat_topkval_indices for i in range(num_filters)], axis=2)
    mat1_topkval_indices_full = tf.stack([mat1_topkval_idx_x_idx for i in range(top_k_val)], axis=4)
    mat2_topkval_idx_x_idx = tf.stack([mat_topkval_indices for i in range(num_filters)], axis=1)
    mat2_topkval_indices_full = tf.stack([mat2_topkval_idx_x_idx for i in range(top_k_val)], axis=3)

    motif_x_motif_indices = tf.abs(tf.subtract(mat1_topkval_indices_full, mat2_topkval_indices_full))
    motif_x_motif_indices_redux = tf.reshape(motif_x_motif_indices, new_dims)
    print "Motif indices matrix dims:", motif_x_motif_indices_redux.get_shape()

    # --------------------
    # Loading PWMs into the first layer convolutions
    
    # Then adjust the filters by putting in PWM info
    # note that there should actually only be 1 set of weights, the first layer
    weights = [v for v in tf.global_variables() if ('weights' in v.name)] 
    weights_list = []
    for i in range(len(pwm_list)):
        pwm = pwm_list[i]
        pad_length = max_size - pwm.weights.shape[1]
        padded_weights = np.concatenate((pwm.weights,
                                         np.zeros((4, pad_length))),
                                        axis=1)
        weights_list.append(padded_weights)

    # stack into weights tensor and assign
    pwm_all_weights = np.stack(weights_list, axis=0).transpose(2, 1, 0)
    pwm_np_tensor = np.expand_dims(pwm_all_weights, axis=0)
    load_pwm_update = weights[0].assign(pwm_np_tensor)

    return motif_x_motif_scores_redux, motif_x_motif_indices_redux, load_pwm_update
