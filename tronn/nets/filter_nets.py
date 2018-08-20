"""description: filtering utils
"""

import numpy as np

import tensorflow as tf

from tronn.nets.util_nets import rebatch


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

    NOTE: this is ANY not ALL
    """
    # assertions
    assert params.get("labels_key") is not None
    assert params.get("filter_tasks") is not None

    # params
    labels_key = params.get("labels_key", "labels")
    labels = inputs[labels_key]
    filter_tasks = params.get("filter_tasks")
    batch_size = params["batch_size"]

    # adjust indices as needed
    if len(filter_tasks) == 0:
        filter_tasks = range(labels.get_shape()[1])
    
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
    """Remove examples that are positive only in a single case in the subset
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
    params.update({"num_queue_threads": 4})
    outputs, params = filter_and_rebatch(outputs, params)
    params.update({"num_queue_threads": 1})
    
    return outputs, params




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
