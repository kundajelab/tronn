"""description: filtering utils
"""

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim


def rebatch(inputs, params):
    """Re-batch after "breaking" a batch
    """
    # assertions
    assert params.get("name") is not None
    assert params.get("batch_size") is not None

    # params
    name = params["name"]
    batch_size = params["batch_size"]
    num_threads = params.get("num_queue_threads", 1)
    
    # set up the train batch
    with tf.variable_scope(name):
        outputs = tf.train.batch(
            inputs,
            batch_size,
            capacity=batch_size*3,
            num_threads=num_threads,
            enqueue_many=True,
            name="rebatch_queue")

    # delete name to make sure queues stay
    # in separate scopes
    del params["name"]

    return outputs, params


def filter_and_rebatch(inputs, params):
    """filter through condition mask and rebatch
    """
    # assertions
    assert inputs.get("condition_mask") is not None
    assert params.get("name") is not None

    # params
    name = params["name"]
    condition_mask = inputs["condition_mask"]
    batch_size = inputs["condition_mask"].get_shape().as_list()[0]
    use_queue = params.get("use_queue", True)
    num_threads = params.get("num_queue_threads", 1)
    
    # get indices
    keep_indices = tf.reshape(tf.where(condition_mask), [-1])

    # and adjust data accordingly
    for key in inputs.keys():
        inputs[key] = tf.gather(inputs[key], keep_indices)
        
    # set up queue
    if use_queue:
        params.update({"batch_size": batch_size})
        outputs, _ = rebatch(inputs, params)
        
    # and delete the condition mask and name
    del outputs["condition_mask"]
    
    return outputs, params


def filter_by_labels(inputs, params):
    """Given specific filter tasks, only push through examples 
    if they are positive in these tasks
    """
    # assertions
    assert params.get("labels_key") is not None
    assert params.get("filter_tasks") is not None

    # params
    labels_key = params.get("labels_key", "labels")
    labels = inputs[labels_key]
    filter_tasks = params.get("filter_tasks")
    batch_size = params["batch_size"]
    
    # set up labels mask for filter tasks
    labels_mask_np = np.zeros((labels.get_shape()[1]))
    for task_idx in filter_tasks:
        labels_mask_np[task_idx] = 1
    labels_mask = tf.cast(
        tf.stack([tf.constant(labels_mask_np) for i in xrange(batch_size)], axis=0),
        tf.float32)
        
    # run a conditional on the labels and get indices
    pos_labels = tf.multiply(labels, labels_mask)
    inputs["condition_mask"] = tf.greater(
        tf.reduce_sum(pos_labels, axis=1), [0])

    # run through queue
    params.update({"num_queue_threads": 4})
    outputs, params = filter_and_rebatch(inputs, params)
    params.update({"num_queue_threads": 1})
    
    return outputs, params


def filter_singleton_labels(inputs, params):
    """Remove examples that are positive only in a single case
    """
    # assertions
    assert params.get("labels_key") is not None
    assert params.get("filter_tasks") is not None

    # params
    labels_key = params.get("labels_key", "labels")
    labels = inputs[labels_key]
    filter_tasks = params.get("filter_tasks")
    batch_size = params["batch_size"]
    outputs = dict(inputs)

    # set up task subset
    labels = [tf.expand_dims(tensor, axis=1)
              for tensor in tf.unstack(labels, axis=1)]
    label_subset = tf.concat(
        [labels[i] for i in filter_tasks],
        axis=1)

    # condition mask
    outputs["condition_mask"] = tf.greater(
        tf.reduce_sum(label_subset, axis=1), [1])

    # run through queue
    outputs, params = filter_and_rebatch(outputs, params)
    
    return outputs, params



# TODO adjust this
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


def filter_by_accuracy(inputs, params):
    """Filter by accuracy
    """
    # assertions
    assert inputs.get("probs") is not None
    assert params.get("importance_task_indices") is not None

    # features, send the rest through
    probs = inputs["probs"]
    labels = inputs["labels"]
    
    # params
    task_indices = params["importance_task_indices"]
    acc_threshold = params.get("acc_threshold", 0.8)
    
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
    inputs["condition_mask"] = tf.greater_equal(accuracy, acc_threshold)
    
    # run through queue
    params.update({"name": "accuracy_filter"})
    outputs, params = filter_and_rebatch(inputs, params)
    
    return outputs, params


def filter_by_activation_pattern(inputs, params):
    """given a specific input pattern, filter for those that match beyond a cutoff
    """
    # assertions
    assert params.get("activation_pattern") is not None
    assert params.get("importance_task_indices") is not None
    assert params.get("activation_pattern_corr_thresh") is not None
    assert inputs.get("logits") is not None

    # features
    logits = inputs["logits"]
    activation_pattern = params["activation_pattern"]
    corr_thresh = params["activation_pattern_corr_thresh"]
    outputs = dict(inputs)

    # adjust features for just importance task indices
    
    
    # do the pearson calculation
    pearson = None # {N}

    # threshold
    inputs["condition_mask"] = tf.greater_equal(pearson, corr_thresh)

    # run through queue
    params.update({"name": "activation_pattern_filter"})
    outputs, params = filter_and_rebatch(outputs, params)

    return outputs, params
