# a way to filter in the middle of processing.
# take outputs, filter, gather to queue and keep going

import tensorflow as tf
import tensorflow.contrib.slim as slim


def rebatch(features, labels, config, is_training=False):
    """Re-batch after "breaking" a batch
    """
    batch_size = config.get("batch_size")
    assert batch_size is not None
    
    tensor_keys = ["features", "labels"]
    tensors = [features, labels]
    
    # attach the other outputs
    for output_key in config["outputs"].keys():
        tensor_keys.append(output_key)
        tensors.append(config["outputs"][output_key])
        
    outputs = tf.train.batch(
        tensors,
        batch_size,
        capacity=100000,
        #capacity=batch_size*10 + 100,
        num_threads=4,
        enqueue_many=True,
        name="rebatch_queue")

    # separate things back out from queue
    features = outputs[0]
    labels = outputs[1]

    new_outputs = {}
    for i in xrange(2, len(outputs)):
        new_outputs[tensor_keys[i]] = outputs[i]
    
    # replace old outputs with new ones
    config["outputs"] = new_outputs

    return features, labels, config


def remove_shuffles(features, labels, config, is_training=False):
    """Given shuffled sequences interspersed, remove them
    """
    shuffle_num = config.get("shuffle_num", 7)
    batch_size = features.get_shape().as_list()[0]
    assert batch_size % (shuffle_num + 1) == 0

    example_num = batch_size / (shuffle_num + 1)
    
    # remove shuffles from features and labels
    features = [tf.expand_dims(example, axis=0)
                for example in tf.unstack(features, axis=0)]
    labels = [tf.expand_dims(example, axis=0)
              for example in tf.unstack(labels, axis=0)]
    final_features = []
    final_labels = []
    for i in xrange(example_num):
        idx = (shuffle_num + 1) * (i)
        final_features.append(features[idx])
        final_labels.append(labels[idx])
    features = tf.concat(final_features, axis=0)
    labels = tf.concat(final_labels, axis=0)

    # remove shuffles from config
    for key in config["outputs"].keys():
        key_tensors = [tf.expand_dims(example, axis=0)
                       for example in tf.unstack(config["outputs"][key], axis=0)]
        new_outputs = []
        for i in xrange(example_num):
            idx = (shuffle_num + 1) * (i)
            new_outputs.append(key_tensors[idx])
        new_outputs = tf.concat(new_outputs, axis=0)
        config["outputs"][key] = new_outputs
        
    return features, labels, config


def filter_through_mask(
        features,
        labels,
        config,
        condition_mask,
        num_threads=1,
        use_queue=True):
    """Given a precalculated condition mask, filter variables
    """
    batch_size = config.get("batch_size")
        
    assert batch_size is not None
    
    # filter examples
    keep_indices = tf.reshape(tf.where(condition_mask), [-1])
    
    # set up tensor keys to track tensor names
    # set up tensor list to push tensors through queue in one (ordered) list
    tensor_keys = ["features", "labels"]
    tensors = [
        tf.gather(features, keep_indices),
        tf.gather(labels, keep_indices)]

    # attach the other outputs
    for output_key in config["outputs"].keys():
        tensor_keys.append(output_key)
        tensors.append(
            tf.gather(config["outputs"][output_key], keep_indices))
        
    # set up queue
    if use_queue:
        outputs = tf.train.batch(
            tensors,
            batch_size,
            #capacity=128+5, # keep capacity small to move faster?
            capacity=100000,
            #capacity=batch_size*10 + 100,
            num_threads=num_threads,
            enqueue_many=True,
            name="filtering_queue")
    else:
        outputs = tensors

    # separate things back out from queue
    features = outputs[0]
    labels = outputs[1]

    new_outputs = {}
    for i in xrange(2, len(outputs)):
        new_outputs[tensor_keys[i]] = outputs[i]
    
    # replace old outputs with new ones
    config["outputs"] = new_outputs
    
    return features, labels, config


# TODO: filters for:
# 1) accuracy
# 2) num importance basepairs
# 3) motif matches: motif vector must have a positive hit in specific locations
# 4) grammar matches

def filter_by_accuracy(features, labels, config, is_training=False):
    """Filter by accuracy
    """
    # get params needed
    probs = config["outputs"].get("probs")
    task_indices = config.get("importance_task_indices")
    batch_size = config.get("batch_size")
    acc_threshold = config.get("acc_threshold", 0.8)

    # assertions
    assert probs is not None
    assert task_indices is not None
    assert batch_size is not None
    
    # collect filtering tasks and probs
    labels_tmp = tf.unstack(labels, axis=1)
    probs_tmp = tf.unstack(probs, axis=1)

    filter_labels = []
    filter_probs = []
    for task_idx in task_indices:
        filter_labels.append(labels_tmp[task_idx])
        filter_probs.append(probs_tmp[task_idx])
    filter_labels = tf.stack(filter_labels, axis=1)
    filter_probs = tf.stack(filter_probs, axis=1)

    # calculate accuracy
    correct_predictions = tf.logical_not(tf.logical_xor(
        tf.cast(filter_labels, tf.bool),
        tf.greater_equal(filter_probs, 0.5))) # {N, tasks}
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=1) # {N}
    
    # set condition
    condition_mask = tf.greater_equal(accuracy, acc_threshold)

    # filter
    with tf.variable_scope("accuracy_filter"):
        features, labels, config = filter_through_mask(
            features, labels, config, condition_mask, use_queue=True, num_threads=1)

    return features, labels, config


def filter_by_importance(features, labels, config, is_training=False):
    """Filter out low importance examples, not interesting
    """
    # get params
    batch_size = config.get("batch_size")
    cutoff = config.get("cutoff", 20)
    positive_only = config.get("positive_only", False)
    
    # assertions
    assert batch_size is not None

    if positive_only:
        # get condition mask
        feature_sums = tf.reduce_max(
            tf.reduce_sum(
                tf.cast(tf.greater(features, 0), tf.float32),
                axis=[2, 3]),
            axis=1) # shape {N}
    else:
        # get condition mask
        feature_sums = tf.reduce_max(
            tf.reduce_sum(
                tf.cast(tf.not_equal(features, 0), tf.float32),
                axis=[2, 3]),
            axis=1) # shape {N}
    
    condition_mask = tf.greater(feature_sums, cutoff)

    # filter
    with tf.variable_scope("importance_filter"):
        features, labels, config = filter_through_mask(
            features, labels, config, condition_mask, num_threads=1)
    
    return features, labels, config


# TODO - is this used?
def filter_by_motif_presence(features, labels, config, is_training=False):
    """Given desired motif vector, filter through that mask
    """
    # get_params
    grammars = config.get("grammars") # {M, G}

    # assertions
    assert grammars is not None
    
    # binarize motif scores and reshape for grammar axis
    features_present = tf.expand_dims(tf.cast(tf.not_equal(features, [0]), tf.float32), axis=-1) # {N, M, 1}

    # TODO separate this out into a grammar layer in nets
    
    # set up grammars
    grammars = tf.stack([grammars for i in xrange(features.shape[0])], axis=0) # {N, M, G}
    grammar_num_motifs = tf.reduce_sum(grammars, axis=1) # {N, G}
    
    # multiply
    grammar_scores = tf.reduce_sum(tf.multiply(features_present, grammars), axis=1) # {N, G}

    # mark hits
    grammar_presence = tf.cast(tf.equal(grammar_scores, grammar_num_motifs), tf.float32) # {N, G}

    # condition mask
    condition_mask = tf.cast(tf.greater(tf.reduce_sum(grammar_presence, axis=1), [0]), tf.float32)

    # and run filter
    with tf.variable_scope("motif_filter"):
        features, labels, config = filter_through_mask(
            features, labels, config, condition_mask)

    return features, labels, config


def filter_by_grammar_presence(features, labels, config, is_training=False):
    """given grammar vector, filter. here, want to filter by presence of key motifs
    """
    grammars_present = config["outputs"].get("grammars_present") #{N}
    assert grammars_present is not None
    
    # reduce sum
    condition_mask = tf.greater(tf.squeeze(grammars_present), [0]) # {N}
    
    # and run filter
    with tf.variable_scope("grammar_filter"):
        features, labels, config = filter_through_mask(
            features, labels, config, condition_mask)

    return features, labels, config


def filter_singles(features, labels, config, is_training=False):
    """Filter out singlets to remove noise
    """
    window = config.get("window", 5)
    min_features = config.get("min_fract", 0.4)
    
    features_present = tf.cast(tf.not_equal(features, 0), tf.float32)

    # check the windows
    feature_counts_in_window = slim.avg_pool2d(features_present, [1, window], stride=[1, 1], padding="SAME")
    feature_mask = tf.cast(tf.greater_equal(feature_counts_in_window, min_features), tf.float32)
    
    # wherever the window is 0, blank out that region
    features = tf.multiply(features, feature_mask)
    
    return features, labels, config


def filter_singles_twotailed(features, labels, config, is_training=False):
    """Filter out singlets, removing positive and negative singlets separately
    """
    # split features
    pos_features = tf.cast(tf.greater(features, 0), tf.float32)
    neg_features = tf.cast(tf.less(features, 0), tf.float32)

    # get masks
    pos_mask, _, _ = filter_singles(pos_features, labels, config, is_training=is_training)
    neg_mask, _, _ = filter_singles(neg_features, labels, config, is_training=is_training)
    keep_mask = tf.add(pos_mask, neg_mask)

    # mask features
    features = tf.multiply(features, keep_mask)

    # output for later
    num_positive_features = tf.reduce_sum(
        tf.cast(
            tf.greater(
                tf.reduce_max(features, axis=[1,3]), [0]),
            tf.float32), axis=1, keep_dims=True)
    config["outputs"]["positive_importance_bp_sum"] = num_positive_features

    return features, labels, config
