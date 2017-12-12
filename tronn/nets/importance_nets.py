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

    features = tf.concat(task_importances, axis=1)

    return features, labels, config


def multitask_global_importance(features, labels, config, is_training=False):
    """Also get global importance
    """
    assert is_training == False
    append = config.get("append", True)
    
    #features_max = tf.reduce_sum(tf.abs(features), axis=1, keep_dims=True)
    features_max = tf.reduce_max(tf.abs(features), axis=1, keep_dims=True)

    if append:
        features = tf.concat([features, features_max], axis=1)
    else:
        features = features_max

    return features, labels, config

