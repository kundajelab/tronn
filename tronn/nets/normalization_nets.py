"""Description: nets for normalizing results
"""

import tensorflow as tf


def normalize_w_probability_weights(features, labels, config, is_training=False):
    """Normalize features on a per example basis. Requires a weights vector,
    normally the probabilities at the final point of the output
    (ie, think of if you had a total weight
    of 1, how should it be spread, and then weight that by the final
    probability value)
    """
    assert is_training == False

    probs = config.get("probs", None)
    assert probs is not None

    # split out into tasks to normalize by task probs
    features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)]
    normalized_features = []
    for i in xrange(len(features)):
        weight_sums = tf.reduce_sum(features[i], axis=[1, 2, 3], keep_dims=True)
        task_features = tf.multiply(
            tf.divide(features[i], weight_sums),
            tf.reshape(probs[i], weight_sums.get_shape()))
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
    
    num_stdev = config.get("normalize.zscore.num_stdev", 3)
    
    # get mean and stdev
    signal_mean, signal_var = tf.nn.moments(features, axes=[1, 2, 3])
    signal_stdev = tf.sqrt(signal_var)
    
    # and zscore
    features = (features - signal_mean) / signal_stdev

    return features, labels, config






