# a way to filter in the middle of processing.
# take outputs, filter, gather to queue and keep going

import tensorflow as tf
import tensorflow.contrib.slim as slim


def filter_through_mask(
        features,
        labels,
        config,
        condition_mask):
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
    outputs = tf.train.batch(
        tensors,
        batch_size,
        capacity=batch_size*3 + 100,
        num_threads=1,
        enqueue_many=True,
        name="filtering_queue")

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

def filter_by_accuracy(features, labels, config, is_training=False):
    """Filter by accuracy
    """
    # get params needed
    probs = config.get("filter_probs")
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
        tf.cast(labels, tf.bool),
        tf.greater_equal(probs, 0.5))) # {N, tasks}
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=1, keep_dims=True) # {N}
    
    # set condition
    condition_mask = tf.greater_equal(accuracy, acc_threshold)

    # filter
    with tf.variable_scope("accuracy_filter"):
        features, labels, config = filter_through_mask(
            features, labels, config, condition_mask)

    return features, labels, config


def filter_by_importance(features, labels, config, is_training=False):
    """Filter out low importance examples, not interesting
    """
    # get params
    batch_size = config.get("batch_size")
    cutoff = config.get("cutoff", 10)

    # assertions
    assert batch_size is not None

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
            features, labels, config, condition_mask)
    
    return features, labels, config


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



def is_accurate(labels, probs, acc_threshold=0.8):
    """Determines which examples pass condition
    """
    # calculate accuracy
    correct_predictions = tf.logical_not(tf.logical_xor(
        tf.cast(labels, tf.bool),
        tf.greater_equal(probs, 0.5))) # {N, tasks}
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=1, keep_dims=True) # {N}
    #print accuracy.get_shape()
    
    # set condition
    #condition_met = tf.greater_equal(accuracy, [acc_threshold])
    condition_met = tf.greater_equal(accuracy, acc_threshold)
    #print condition_met.get_shape()
    
    return condition_met


def filter_by_fn(features, labels, config, is_training=False):
    """Filter by specific tasks and probabilities
    """
    # if going to filter, have to filter features, labels, metadata, probs, logits
    # also requires probs. but separate this out into fn
    probs = config.get("filter_probs")
    task_indices = config.get("importance_task_indices")
    batch_size = config.get("batch_size")
    acc_threshold = config.get("acc_threshold", 0.8)
    filter_fn = config.get("filter_fn")
    
    assert probs is not None
    assert task_indices is not None
    assert batch_size is not None
    assert filter_fn is not None
    
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
    
    # run condition
    condition_mask = filter_fn(filter_labels, filter_probs, acc_threshold=acc_threshold)
    selected_examples = tf.reshape(tf.where(condition_mask), [-1])
    
    # make a list
    tensor_keys = ["features", "labels"]
    tensors = [
        tf.gather(features, selected_examples),
        tf.gather(labels, selected_examples)]

    for output_key in config["outputs"].keys():
        tensor_keys.append(output_key)
        tensors.append(
            tf.gather(config["outputs"][output_key], selected_examples))
        
    # set up a second queue
    outputs = tf.train.batch(
        tensors,
        batch_size,
        capacity=batch_size*3 + 100,
        num_threads=1,
        enqueue_many=True,
        name="accuracy_filter")

    features = outputs[0]
    labels = outputs[1]

    new_outputs = {}
    for i in xrange(2, len(outputs)):
        new_outputs[tensor_keys[i]] = outputs[i]
    
    # replace old outputs with new ones
    config["outputs"] = new_outputs

    return features, labels, config


def filter_by_accuracy_old(features, labels, config, is_training=False):
    """Filter by accuracy
    """
    config["filter_fn"] = is_accurate

    features, labels, config = filter_by_fn(
        features, labels, config, is_training=False)

    return features, labels, config


def is_low_importance(features, cutoff=10):
    """Remove examples that really don't have importance scores
    """
    feature_sums = tf.reduce_max(
        tf.reduce_sum(
            tf.cast(tf.not_equal(features, 0), tf.float32),
            axis=[2, 3]),
        axis=1) # shape {N}

    condition_met = tf.greater(feature_sums, cutoff)
    
    return condition_met


def filter_by_importance_old(features, labels, config, is_training=False):
    """Filter out low importance examples, not interesting
    """
    # if going to filter, have to filter features, labels, metadata, probs, logits
    # also requires probs. but separate this out into fn
    batch_size = config.get("batch_size")
    cutoff = config.get("cutoff", 10)
    assert batch_size is not None

    # run condition
    condition_mask = is_low_importance(features, cutoff=cutoff)
    selected_examples = tf.reshape(tf.where(condition_mask), [-1])

    # TODO: here to below should be factored out
    # make a list
    tensor_keys = ["features", "labels"]
    tensors = [
        tf.gather(features, selected_examples),
        tf.gather(labels, selected_examples)]

    for output_key in config["outputs"].keys():
        tensor_keys.append(output_key)
        tensors.append(
            tf.gather(config["outputs"][output_key], selected_examples))
        
    # set up a second queue
    outputs = tf.train.batch(
        tensors,
        batch_size,
        capacity=batch_size*3 + 100,
        num_threads=1,
        enqueue_many=True,
        name="importance_filter")

    features = outputs[0]
    labels = outputs[1]

    new_outputs = {}
    for i in xrange(2, len(outputs)):
        new_outputs[tensor_keys[i]] = outputs[i]
    
    # replace old outputs with new ones
    config["outputs"] = new_outputs
    
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

    return features, labels, config












# OLD STUFF


def setup_fifo_queue(tensors, batch_size):
    """
    """
    # set up queue
    tensor_dtypes = [tensor.dtype for tensor in tensors]
    tensor_shapes = [tensor.get_shape()[1:] for tensor in tensors]
    
    fifo_queue = tf.FIFOQueue(
        batch_size*3 + 100,
        dtypes=tensor_dtypes,
        shapes=tensor_shapes)
    enqueue = fifo_queue.enqueue_many(tensors)
    outputs = fifo_queue.dequeue_many(batch_size)

    # set up queue runner
    queue_runner = tf.train.QueueRunner(
        fifo_queue,
        enqueue_ops=[enqueue])
    tf.train.add_queue_runner(queue_runner)

    return outputs


def filter_by_accuracy_test(features, labels, config, is_training=False):
    """testing
    """ 
    batch_size = config.get("batch_size")

    # make a list
    tensor_keys = ["features", "labels"]
    tensors = [features, labels]

    print config["outputs"]
    
    for output_key in config["outputs"].keys():
        tensor_keys.append(output_key)
        tensors.append(config["outputs"][output_key])

    #tensors = [
    #    features,
    #    labels,
    #    config["outputs"]["logits"],
    #    config["outputs"]["probs"],
     #   config["outputs"]["example_metadata"]]

    print ""
    print tensors

    # set up a second queue
    #outputs = setup_fifo_queue(tensors, batch_size)

    outputs = tf.train.batch(
        tensors,
        batch_size,
        capacity=batch_size*3 + 100,
        num_threads=1,
        enqueue_many=True,
        name="accuracy_filter")


    print ""
    print outputs
    print ""
    
    features = outputs[0]
    labels = outputs[1]

    new_outputs = {}
    for i in xrange(2, len(outputs)):
        new_outputs[tensor_keys[i]] = outputs[i]

    print new_outputs
    
    # replace old outputs with new ones
    config["outputs"] = new_outputs

    print config["outputs"]


    return features, labels, config
