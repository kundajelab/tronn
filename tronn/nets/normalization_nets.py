"""Description: nets for normalizing results
"""

import logging

import tensorflow as tf

from tronn.util.utils import DataKeys


def normalize_to_importance_logits(inputs, params):
    """normalize to the logits
    """
    # assertions
    assert params.get("weight_key") is not None
    assert params.get("importance_task_indices") is not None
    assert inputs.get(params["weight_key"]) is not None
    assert inputs.get("features") is not None

    logging.info("LAYER: normalize to the logits")
    
    # features
    features = inputs[DataKeys.FEATURES]
    features_shape = features.get_shape().as_list()
    outputs = dict(inputs)

    # weights
    weights = tf.gather(
        inputs[params["weight_key"]],
        params["importance_task_indices"],
        axis=1) # {N,.., task}
    weights_shape = weights.get_shape().as_list() # {N, shuf, logit} or {N, logit}
    diff_dims = len(features_shape) - len(weights_shape)
    weights = tf.reshape(weights, weights_shape + [1 for i in xrange(diff_dims)])
    logging.debug("...weights: {}".format(weights.get_shape()))
    
    axes = range(len(features_shape))[len(weights_shape):]
    logging.debug("...reduction over axes: {}".format(axes))
    
    # summed importances
    importance_sums = tf.reduce_sum(tf.abs(features), axis=axes, keepdims=True)
    logging.debug("...importance_sums: {}".format(importance_sums.get_shape()))
    
    # normalize
    features = tf.divide(features, importance_sums)
    features = tf.multiply(weights, features)

    # guard against inf
    features = tf.multiply(
        features,
        tf.cast(tf.greater(importance_sums, 1e-7), tf.float32))
    
    outputs[DataKeys.FEATURES] = features

    logging.debug("RESULTS: {}".format(outputs[DataKeys.FEATURES].get_shape()))
    
    return outputs, params

# TODO deprecate
def normalize_to_weights(inputs, params):
    """Normalize features on a per example basis. Requires a weights vector,
    normally the probabilities at the final point of the output
    (ie, think of if you had a total weight
    of 1, how should it be spread, and then weight that by the final
    probability value)
    """
    # assertions
    assert params.get("weight_key") is not None
    assert params.get("importance_task_indices") is not None
    assert inputs.get(params["weight_key"]) is not None
    assert inputs.get("features") is not None
    
    # get features and pass rest through
    features = inputs["features"] # {N, task, ...}
    weights = inputs[params["weight_key"]]
    importance_task_indices = params["importance_task_indices"]
    outputs = dict(inputs)
    
    # split out by task
    features = [tf.expand_dims(tensor, axis=1)
                for tensor in tf.unstack(features, axis=1)] # {N, 1, ...}
    weights = [tf.expand_dims(tensor, axis=1)
               for tensor in tf.unstack(weights, axis=1)] # {N, 1}?
    
    # normalize
    # for normalization, just use total sum of importance scores to be 1
    # this makes probs just a confidence measure.
    normalized_features = []
    for i in xrange(len(features)):
        weight_sums = tf.reduce_sum(tf.abs(features[i]), axis=[1, 2, 3], keepdims=True)
        task_features = tf.multiply(
            tf.divide(features[i], weight_sums),
            tf.reshape(tf.abs(weights[importance_task_indices[i]]),
                       weight_sums.get_shape()))
        normalized_features.append(task_features)

    

    # and concat back into a block
    features = tf.concat(normalized_features, axis=1)

    outputs["features"] = features
    
    return outputs, params


def normalize_to_weights_w_shuffles(inputs, params):
    """normalize to weights for both the features and the shuffles
    """
    # first the features
    outputs = normalize_to_weights(inputs, params)[0]

    # then the shuffles
    shuffles = inputs[DataKeys.WEIGHTED_SEQ_SHUF]
    shuf_shape = shuffles.get_shape().as_list()
    shuffles = tf.reshape(
        shuffles,
        [shuf_shape[0]*shuf_shape[1],
         shuf_shape[2],
         shuf_shape[3],
         shuf_shape[4]])

    shuf_logits_key = "{}.{}".format(DataKeys.LOGITS, DataKeys.SHUFFLE_SUFFIX)
    shuf_logits = inputs[shuf_logits_key]
    shuf_logits_shape = shuf_logits.get_shape().as_list()
    shuf_logits = tf.reshape(
        shuf_logits,
        [shuf_logits_shape[0]*shuf_logits_shape[1],
         shuf_logits_shape[2]])
    
    inputs.update({
        DataKeys.FEATURES: shuffles,
        shuf_logits_key: shuf_logits})
    params.update({"weight_key": shuf_logits_key})
    shuffles = normalize_to_weights(inputs, params)[0][DataKeys.FEATURES]
    outputs[DataKeys.WEIGHTED_SEQ_SHUF] = tf.reshape(shuffles, shuf_shape)

    return outputs, params


def normalize_to_absolute_one(inputs, params):
    """Normalize features on a per example basis. Requires a weights vector,
    normally the probabilities at the final point of the output
    (ie, think of if you had a total weight
    of 1, how should it be spread, and then weight that by the final
    probability value)
    """
    # assertions
    assert inputs.get(DataKeys.FEATURES) is not None
    
    # get features and pass rest through
    features = inputs[DataKeys.FEATURES] # {N, task, ...}
    mask = tf.cast(tf.equal(inputs[DataKeys.MUT_MOTIF_POS], 0), tf.float32)
    blanked_features = tf.multiply(mask, features) # {N, mutM, task, pos, 4}
    outputs = dict(inputs)

    # divide
    importance_sums = tf.reduce_sum(tf.abs(blanked_features), axis=[-2, -1], keepdims=True)
    features = tf.divide(features, importance_sums)

    # guard against inf
    features = tf.multiply(
        features,
        tf.cast(tf.greater(importance_sums, 1e-7), tf.float32))

    # save out
    outputs[DataKeys.FEATURES] = features
    
    return outputs, params


def normalize_to_delta_logits(inputs, params):
    """Normalize features on a per example basis. Requires a weights vector,
    normally the probabilities at the final point of the output
    (ie, think of if you had a total weight
    of 1, how should it be spread, and then weight that by the final
    probability value)
    """
    # assertions
    assert inputs.get("delta_logits") is not None
    assert inputs.get("features") is not None
    
    # get features and pass rest through
    features = inputs["features"] # {N, task, ...}
    weights = inputs["delta_logits"]
    outputs = dict(inputs)
    
    # split out by example (each is a different mutation)
    features = tf.unstack(features, axis=0) # list of {task, seqlen, 4}

    # split out by example
    weights = tf.unstack(weights, axis=0) # list of {task, mut}
    
    # ignore the first one
    normalized_features = [features[0]]
    
    # normalize
    for i in xrange(1,len(features)):
        weight_sums = tf.reduce_sum(tf.abs(features[i]), axis=[1, 2], keepdims=True) # {task}
        task_features = tf.multiply(
            tf.divide(features[i], weight_sums),
            tf.reshape(tf.abs(weights[i-1][:,i-1]),
                       weight_sums.get_shape()))
        normalized_features.append(task_features)

    # and concat back into a block
    features = tf.stack(normalized_features, axis=0)

    outputs["features"] = features
    
    return outputs, params




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





