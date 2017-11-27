# Description: joins various smaller nets to run analyses after getting predictions

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.nets.motif_nets import pwm_convolve_v3
from tronn.nets.motif_nets import motif_assignment


def threshold_top_k(input_tensor, k_val):
    """Get top k hits across all axes
    """
    # grab the top k and threshold out the rest
    input_shape = input_tensor.get_shape().as_list()
    top_k_shape = [input_shape[0], input_shape[2]*input_shape[3]]
    top_k_vals, top_k_indices = tf.nn.top_k(tf.reshape(input_tensor, top_k_shape), k=k_val)
    
    thresholds = tf.reshape(tf.reduce_min(top_k_vals, axis=1, keep_dims=True), [input_shape[0], 1, 1, 1])
    # TODO(dk) fix below - need to get actual scores, not presence
    top_hits_w_location = tf.cast(tf.greater_equal(input_tensor, thresholds), tf.float32) # out: (example, 1, bp_pos, motif)

    top_scores = tf.multiply(input_tensor, top_hits_w_location) # this gives you back actual scores

    # and aggregate
    # TODO(dk) fix this, need to reduce mean correctly.... or just use max?
    #top_hits_per_example = tf.squeeze(tf.reduce_mean(top_hits_w_location, axis=2)) # out: (example, motif)
    top_hits_per_example = tf.squeeze(tf.reduce_max(top_hits_w_location, axis=2)) # out: (example, motif)

    return top_hits_per_example


def get_top_k_motif_hits(
        features, 
        labels, 
        model_params, 
        is_training=False):
    """Get motif hits and then get top k
    """
    # convolve with PWMs
    pwm_scores = pwm_convolve_v3(features, labels, model_params) # out: (example, 1, bp_pos, motif)

    # max pool?
    pwm_scores = slim.max_pool2d(pwm_scores, [1,10], stride=[1,10])
    
    # TODO(dk) maybe just do the MAX instead?
    pwm_scores_thresholded = threshold_top_k(pwm_scores, model_params["k_val"])

    return pwm_scores_thresholded


def get_importance_weighted_motif_hits(
        features,
        labels,
        model_params,
        is_training=False,
        k_val=10):
    """Goes from one-hot sequence to global pwm hits. 
    Use as standard addition on the inference graph, assumes trained model
    has been set up by tronn graph so that you have logits

    Args:
      features: the anchor point on which you want importances
      logits_list: per task, the logit tensors
    """
    assert is_training == False
    outputs = {}

    # pull out from model params
    pwm_list = model_params["pwm_list"]
    importances_fn = model_params["importances_fn"]
    logits_tensors = model_params["logits"]
    probs_tensors = model_params["probs"]
    normalize = model_params["normalize"]
    
    # first set up importances
    per_task_importances = []
    for logits_tensor_idx in xrange(len(logits_tensors)):
        # get importances (note: importance fn does the thresholding and normalization)
        per_task_importance = importances_fn(
            logits_tensors[logits_tensor_idx],
            features,
            probs=probs_tensors[logits_tensor_idx] if normalize else None,
            normalize=normalize)
        
        # append
        per_task_importances.append(per_task_importance)
        
    # stack
    importances = tf.squeeze(tf.stack(per_task_importances, axis=1)) # out: (example, task, bp_pos, bp/channel)
    outputs["importances"] = importances
    
    # reduce max (just want regions where importance was found across the board)
    global_importances = tf.reduce_max(importances, axis=1, keep_dims=True) # out: (example, 1, bp_pos, bp/channel)

    # get top k hits
    #outputs["pwm_hits"] = get_top_k_motif_hits(global_importances, labels, {"pwms": pwm_list, "k_val": k_val})
    outputs["pwm_hits"] = motif_assignment(global_importances, labels, {"pwms": pwm_list, "k_val": k_val})
    
    return outputs


def get_motif_assignments(
        features,
        labels,
        model_params,
        is_training=False):
    """This procedure assigns 1 motif to (approx) 1 importance seqlet.
    """
    assert is_training == False
    outputs = {}

    # pull out from model params
    pwm_list = model_params["pwm_list"]
    importances_fn = model_params["importances_fn"]
    logits_tensors = model_params["logits"]
    probs_tensors = model_params["probs"]
    normalize = model_params["normalize"]
    absolute_val = model_params.get("abs_val", False)
    
    # first set up importances
    per_task_importances = []
    for logits_tensor_idx in xrange(len(logits_tensors)):
        # get importances (note: importance fn does the thresholding and normalization)
        per_task_importance = importances_fn(
            logits_tensors[logits_tensor_idx],
            features,
            probs=probs_tensors[logits_tensor_idx] if normalize else None,
            normalize=normalize)
        
        # append
        per_task_importances.append(per_task_importance)
        
    # stack
    importances = tf.squeeze(tf.stack(per_task_importances, axis=1)) # out: (example, task, bp_pos, bp/channel)

    # TODO(dk) convert all to pos ONLY for when getting motifs - want negatives for grammars.
    if absolute_val:
        importances = tf.abs(importances)
    
    # reduce max (just want regions where importance was found across the board)
    global_importances = tf.reduce_max(importances, axis=1, keep_dims=True) # out: (example, 1, bp_pos, bp/channel)

    # now add global importances to the mix
    per_task_importances.append(global_importances) # {N, tasks+1, pos, channel}

    # get top k hits
    for i in xrange(len(per_task_importances)):
        with tf.variable_scope("task_{}".format(i)):
            outputs["pwm-counts.taskidx-{}".format(i)] = motif_assignment(per_task_importances[i], labels, {"pwms": pwm_list})

    return outputs
