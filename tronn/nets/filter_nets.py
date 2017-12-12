# a way to filter in the middle of processing.
# take outputs, filter, gather to queue and keep going

import tensorflow as tf


def is_accurate(labels, probs, acc_threshold=0.7):
    """Determines which examples pass condition
    """
    # calculate accuracy
    correct_predictions = tf.logical_not(tf.logical_xor(
        tf.cast(labels, tf.bool),
        tf.greater_equal(probs, 0.5))) # {N, tasks}
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=1) # {N}
    
    # set condition
    condition_met = tf.greater_equal(accuracy, acc_threshold)

    return condition_met


def filter_by_accuracy(features, labels, config, is_training=False):
    """Filter by specific tasks and probabilities
    """
    # if going to filter, have to filter features, labels, metadata, probs, logits
    # also requires probs. but separate this out into fn
    probs = config.get("filter_probs")
    task_indices = config.get("importance_task_indices")
    batch_size = config.get("batch_size")
    acc_threshold = config.get("acc_threshold", 0.7)
    
    assert probs is not None
    assert task_indices is not None
    assert batch_size is not None
    
    # collect filtering tasks and probs
    labels = tf.unstack(labels, axis=1)
    probs = tf.unstack(probs, axis=1)

    filter_labels = []
    filter_probs = []
    for task_idx in task_indices:
        filter_labels.append(labels[task_idx])
        filter_probs.append(probs[task_idx])
    filter_labels = tf.stack(filter_labels, axis=1)
    filter_probs = tf.stack(filter_probs, axis=1)
    
    # run condition
    condition_mask = is_accurate(filter_labels, filter_probs, acc_threshold=acc_threshold)
    selected_examples = tf.reshape(tf.where(condition_mask), [-1])

    # gather in outputs (including features and labels)
    filtered_outputs = {}
    for output_key in config["outputs"].keys():
        filtered_outputs[output_key] = tf.gather(config["outputs"][output_key], selected_examples)
    filtered_outputs["features"] = tf.gather(features, selected_examples)
    filtered_outputs["labels"] = tf.gather(labels, selected_examples)
        
    # set up a second queue
    outputs = tf.train.batch(
        filtered_outputs,
        batch_size,
        capacity=100000,
        enqueue_many=True,
        name="accuracy_filter")

    # get back features and labels
    features = outputs["features"]
    del outputs["features"]
    labels = outputs["labels"]
    del outputs["labels"]
    
    # replace old outputs with new ones
    config["outputs"] = outputs

    return features, labels, config
