"""Description: graphs that transform importance scores to other representations
"""

import tensorflow as tf

from tronn.nets.filter_nets import rebatch
from tronn.nets.filter_nets import remove_shuffles


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


def integrated_gradients(features, labels, config, is_training=False):
    """Integrated gradients as proposed by Sundararajan 2017
    
    NOTE: the current set up is such that a batch is basically a scaled set of sequences
    so just reduce mean on them.
    """
    batch_size = config.get("batch_size")
    assert batch_size is not None

    # run input_x_grad
    features, _, _ = input_x_grad(features, labels, config, is_training=False)

    # for features, reduce mean
    features = tf.reduce_mean(features, axis=0, keep_dims=True)

    # NOTE: DONT adjust here, it references the original dict
    # for everything else, just grab the LAST of the batch (which is the non-scaled sequence)
    labels = tf.expand_dims(tf.unstack(labels, axis=0)[-1], axis=0)
    for key in config["outputs"].keys():
        config["outputs"][key] = tf.expand_dims(
            tf.unstack(config["outputs"][key], axis=0)[-1], axis=0)

    return features, labels, config


def get_diff_from_ref(features, shuffle_num=7):
    """Get the diff from reference, but maintain batch
    only remove shuffles at the end of the process. For deeplift
    """
    batch_size = features.get_shape().as_list()[0]
    assert batch_size % (shuffle_num + 1) == 0

    example_num = batch_size / (shuffle_num + 1)

    # unstack to get diff from ref
    features = [tf.expand_dims(example, axis=0)
                for example in tf.unstack(features, axis=0)]
    for i in xrange(example_num):
        idx = (shuffle_num + 1) * (i)
        actual = features[idx]
        references = features[idx+1:idx+shuffle_num+1]
        diff = tf.subtract(actual, tf.reduce_mean(references, axis=0))
        features[idx] = diff

    # restack
    features = tf.concat(features, axis=0)
    
    return features


def build_deeplift_multiplier(x, y, multiplier=None, shuffle_num=7):
    """Takes input and activations to pass down
    """
    # TODO figure out how to use sets for name matching
    linear_names = set(["Conv", "conv", "fc"])
    nonlinear_names = set(["Relu", "relu"])

    if "Relu" in y.name:
        # rescale rule
        delta_y = get_diff_from_ref(y, shuffle_num=shuffle_num)
        delta_x = get_diff_from_ref(x, shuffle_num=shuffle_num)
        multiplier = tf.divide(
            delta_y, delta_x)
    elif "Conv" in y.name or "fc" in y.name:
        # linear rule
        [weights] = tf.gradients(y, x, grad_ys=multiplier)
        #delta_x = get_diff_from_ref(x, shuffle_num=shuffle_num)
        #multiplier = tf.multiply(
        #    weights, delta_x)
        multiplier = weights
    else:
        # TODO implement reveal cancel rule?
        print y.name, "not recognized"
        quit()

    return multiplier


def deeplift(features, labels, config, is_training=False):
    """Basic deeplift in raw tensorflow
    """
    assert is_training == False
    assert config.get("anchor") is not None
    anchor = config.get("anchor")
    shuffle_num = config.get("shuffle_num", 7)
    
    # deepnet needs to be adding activations (for EVERY layer)
    # to the DEEPLIFT_ACTIVATIONS collection
    activations = tf.get_collection("DEEPLIFT_ACTIVATIONS")
    activations = [features] + activations
    
    # go backwards through the variables
    activations.reverse()
    for i in xrange(len(activations)):
        current_activation = activations[i]
        
        if i == 0:
            previous_activation = tf.identity(
                anchor, name="fc.anchor")
            multiplier = None
        else:
            previous_activation = activations[i-1]

        # function here to build multiplier and pass down
        multiplier = build_deeplift_multiplier(
            current_activation,
            previous_activation,
            multiplier=multiplier,
            shuffle_num=shuffle_num)
        
    features = multiplier
 
    return features, labels, config


def multitask_importances(features, labels, config, is_training=False):
    """Set up importances coming from multiple tasks
    """
    assert is_training == False

    # get configs
    anchors = config["outputs"].get("logits")
    task_indices = config.get("importance_task_indices")
    backprop = config.get("backprop", "input_x_grad")

    if backprop == "input_x_grad":
        importances_fn = input_x_grad
    elif backprop == "integrated_gradients":
        importances_fn = integrated_gradients
    elif backprop == "deeplift":
        importances_fn = deeplift
    else:
        print "method does not exist/not implemented!"
        quit()

    assert anchors is not None
    assert task_indices is not None
    assert importances_fn is not None
    print importances_fn
    
    # split out anchors by task
    anchors = [tf.expand_dims(tensor, axis=1)
               for tensor in tf.unstack(anchors, axis=1)] # {N, 1}

    # get task specific importances
    task_importances = []
    for anchor_idx in task_indices:
        config["anchor"] = anchors[anchor_idx]
        task_importance, _, _ = importances_fn(
            features, labels, config)
        task_importances.append(task_importance)

    features = tf.concat(task_importances, axis=1) # {N, task, pos, C}
    
    # adjust labels and configs as needed here
    if backprop == "integrated_gradients":
        labels = tf.expand_dims(tf.unstack(labels, axis=0)[-1], axis=0)
        for key in config["outputs"].keys():
            config["outputs"][key] = tf.expand_dims(
                tf.unstack(config["outputs"][key], axis=0)[-1], axis=0)
        #features, labels, config = rebatch(features, labels, config)
    elif backprop == "deeplift":
        # TODO - actually remove shuffles later, make dependent on shuffle null and data input loader
        features, labels, config = remove_shuffles(features, labels, config)
        #features, labels, config = rebatch(features, labels, config)
        
    return features, labels, config


def multitask_global_importance(features, labels, config, is_training=False):
    """Also get global importance. does a check to see that the feature is at least
    observed twice (count_thresh)
    """
    assert is_training == False
    append = config.get("append", True)
    count_thresh = config.get("count_thresh", 2)
    
    # per example, only keep positions that have been seen more than once
    features_by_example = [tf.expand_dims(tensor, axis=0)
                           for tensor in tf.unstack(features)] # {1, task, M}

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
        features_max = tf.reduce_sum(features, axis=1, keep_dims=True)
    elif reduce_type == "max":
        features_max = tf.reduce_max(features, axis=1, keep_dims=True)
    elif reduce_type == "mean":
        features_max = tf.reduce_mean(features, axis=1, keep_dims=True)

    # append or replace
    if append:
        features = tf.concat([features, features_max], axis=1)
    else:
        features = features_max

    # things to keep
    if config.get("keep_global_pwm_scores") is not None:
        # attach to config
        config["outputs"][config["keep_global_pwm_scores"]] = features_max #{N, pos, motif}
        config["keep_global_pwm_scores"] = None # TODO fix this, this is because there is overwriting
        
    return features, labels, config

