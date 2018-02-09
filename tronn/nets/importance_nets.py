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
    use_relu = config.get("relu", False)
    
    anchor = config.get("anchor")
    [feature_grad] = tf.gradients(anchor, [features])
    features = tf.multiply(features, feature_grad, 'input_x_grad')

    if use_relu:
        features = tf.nn.relu(features)
    
    return features, labels, config


# TODO basic deeplift
# TODO integrated gradients?


def multitask_importances(features, labels, config, is_training=False):
    """Set up importances coming from multiple tasks
    """
    assert is_training == False

    # get configs
    anchors = config.get("anchors")
    task_indices = config.get("importance_task_indices")
    importances_fn = config.get("importances_fn", input_x_grad)
    #keep_key = config.get("keep_onehot_sequence")

    #if keep_key is not None:
        #config["outputs"][keep_key] = features
        #config["keep_features"] = None
    
    assert anchors is not None
    assert task_indices is not None
    assert importances_fn is not None
    
    # split out anchors by task
    anchors = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(anchors, axis=1)] # {N, 1, pos, C}

    # get task specific importances
    task_importances = []
    for anchor_idx in task_indices:
        config["anchor"] = anchors[anchor_idx]
        task_importance, _, _ = importances_fn(
            features, labels, config)
        task_importances.append(task_importance)

    features = tf.concat(task_importances, axis=1) # {N, task, pos, C}

    return features, labels, config


def multitask_global_importance(features, labels, config, is_training=False):
    """Also get global importance. does a check to see that the feature is at least
    observed twice (count_thresh)
    """
    assert is_training == False
    append = config.get("append", True)
    count_thresh = config.get("count_thresh", 2)
    
    # per example, only keep positions that have been seen more than once
    features_by_example = [tf.expand_dims(tensor, axis=0) for tensor in tf.unstack(features)] # {1, task, M}

    # for each example, get sum across tasks
    # TODO separate this out as different function
    masked_features_list = []
    for example_features in features_by_example:
        motif_counts = tf.reduce_sum(
            tf.cast(tf.not_equal(example_features, 0), tf.float32),
            axis=1, keep_dims=True) # sum across tasks {1, 1, M}
        #motif_max = tf.reduce_max(
        #    motif_counts, axis=[1, 3], keep_dims=True) # {1, 1, pos, 1}
        # then mask based on max position
        motif_mask = tf.cast(tf.greater_equal(motif_counts, count_thresh), tf.float32)
        # and mask
        masked_features = tf.multiply(motif_mask, example_features)
        masked_features_list.append(masked_features)

    # stack
    features = tf.concat(masked_features_list, axis=0)
    
    # TODO could do a max - min scoring system?
    reduce_type = config.get("reduce_type", "sum")

    if reduce_type == "sum":
        features_max = tf.reduce_sum(features, axis=1, keep_dims=True) # TODO add abs? probably not
    elif reduce_type == "max":
        features_max = tf.reduce_max(features, axis=1, keep_dims=True)
    
    #features_max = tf.reduce_max(tf.abs(features), axis=1, keep_dims=True)

    if append:
        features = tf.concat([features, features_max], axis=1)
    else:
        features = features_max

    if config.get("keep_global_pwm_scores") is not None:
        # attach to config
        config["outputs"][config["keep_global_pwm_scores"]] = features_max #{N, pos, motif}
        config["keep_global_pwm_scores"] = None # TODO fix this, this is because there is overwriting
        
    return features, labels, config

