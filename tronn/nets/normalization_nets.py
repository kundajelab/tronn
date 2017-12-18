"""Description: nets for normalizing results
"""

import tensorflow as tf


def normalize_w_probability_weights(features, labels, config, is_training=False):
    """Normalize features on a per example basis. Requires a weights vector,
    normally the probabilities at the final point of the output
    (ie, think of if you had a total weight
    of 1, how should it be spread, and then weight that by the final
    probability value)

    Rationale (for clarifying my thoughts to myself): there are usually more
    positive than negative base pairs, with input_x_grad. This means that
    probs better than logits for normalization. This also means that there
    is usually a positive gap between sum(pos bp) and sum (neg bp). So normalize
    that gap to the prob val.

    Possible failure modes:
    1) sum(features) < 0. Strong negative features. (Though do we care, these are negatives)
    2) sum(features) very close to 0. will make features explode, mostly problem
    if strong prob score towards 1.

    """
    assert is_training == False

    probs = config.get("normalize_probs", None)
    assert probs is not None

    # split out by task
    probs = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(probs, axis=1)]
    
    # use probs as a confidence measure. further from 0.5 the stronger - so increase the importance weights accordingly.
    # for the timepoints, key is to explain the dominant prediction, so multiply by neg for negative scores.
    # for global, use abs value.
    # something w probs + values?
    #probs = [tf.subtract(tensor, 0.5) for tensor in probs] # 0.5 is technically not confident
    features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)]

    # normalize
    # for normalization, just use total sum of importance scores to be 1
    # this makes probs just a confidence measure.
    normalized_features = []
    for i in xrange(len(features)):
        weight_sums = tf.reduce_sum(tf.abs(features[i]), axis=[1, 2, 3], keep_dims=True)
        #weight_sums = tf.reduce_sum(features[i], axis=[1, 2, 3], keep_dims=True)
        task_features = tf.multiply(
            tf.divide(features[i], weight_sums), # TODO add some weight to make sure doesnt explode?
            tf.reshape(probs[i], weight_sums.get_shape()))
        normalized_features.append(task_features)

    # and concat back into a block
    features = tf.concat(normalized_features, axis=1)

    return features, labels, config


def normalize_to_logits(features, labels, config, is_training=False):
    """Normalize to logits? Most likely not best way to do it
    """
    assert is_training == False

    logits = config.get("logits", None)
    assert logits is not None
    
    # split out into tasks to normalize by task probs
    features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)]
    normalized_features = []
    for i in xrange(len(features)):
        weight_sums = tf.reduce_sum(features[i], axis=[1, 2, 3], keep_dims=True) # impt edge case - balanced pos neg
        task_features = tf.multiply(
            tf.divide(features[i], tf.abs(weight_sums)), # TODO add some weight to make sure doesnt explode?
            tf.reshape(tf.abs(logits[i]), weight_sums.get_shape()))
        normalized_features.append(task_features)

    # and concat back into a block
    features = tf.concat(normalized_features, axis=1)
    
    return features, labels, config


def normalize_to_probabilities(features, labels, config, is_training=False):
    """Normalize features such that total weight
    is the final probability value (ie, think of if you had a total weight
    of 1, how should it be spread, and then weight that by the final
    probability value)
    """
    assert is_training == False

    weight_sums = tf.reduce_sum(features, axis=[1, 2, 3], keep_dims=True)
    features = tf.divide(features, weight_sums)
    
    return features, labels, config


def zscore(features, labels, config, is_training=False):
    """Zscore the features.
    """
    assert is_training == False
    
    # get mean and stdev
    signal_mean, signal_var = tf.nn.moments(features, axes=[1, 2, 3])
    signal_stdev = tf.sqrt(signal_var)
    
    # and zscore
    features = tf.divide(
        tf.subtract(features, tf.reshape(signal_mean, [features.get_shape().as_list()[0], 1, 1, 1])),
        tf.reshape(signal_stdev, [features.get_shape().as_list()[0], 1, 1, 1]))

    return features, labels, config


def zscore_and_scale_to_weights(features, labels, config, is_training=False):
    """Zscore such that the standard dev is not 1 but {weight}
    """
    assert is_training == False

    #weights = config.get("logits", None)
    weights = config.get("probs", None)
    assert weights is not None

    weights = tf.subtract(weights, 0.5)
    
    # zscore and multiply by abs(logit) - ie, the bigger the logit (more confident) the stronger
    # the contributions should be
    features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)]
    normalized_features = []
    for i in xrange(len(features)):
        features_z, labels, config = zscore(features[i], labels, config, is_training=is_training)
        task_features = tf.multiply(
            features_z,
            tf.reshape(weights[i], [features_z.get_shape().as_list()[0], 1, 1, 1])) # just multiply by logits, not absolute val?
        normalized_features.append(task_features)

    # and concat
    features = tf.concat(normalized_features, axis=1)
    
    return features, labels, config





