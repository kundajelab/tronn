# Description: joins various smaller nets to run analyses after getting predictions

import tensorflow as tf

from tronn.nets.motif_nets import pwm_convolve_v3


def get_motif_hits(
        features,
        labels,
        model_params,
        is_training=False,
        k_val=5):
    """Goes from one-hot sequence to global pwm hits. 
    Use as standard addition on the inference graph, assumes trained model
    has been set up by tronn graph so that you have logits

    Args:
      features: the anchor point on which you want importances
      logits_list: per task, the logit tensors
    """
    assert is_training == False
    outputs = {}

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
    
    # then convolve with PWMs
    pwm_scores = pwm_convolve_v3(features, labels, {"pwms": pwm_list[0:3]}) # out: (example, 1, bp_pos, motif)
    
    # grab the top k hits (across ALL motifs) and threshold out the rest
    pwm_scores_shape = pwm_scores.get_shape().as_list()
    top_k_shape = [pwm_scores_shape[0], pwm_scores_shape[2]*pwm_scores_shape[3]]
    top_k_vals, top_k_indices = tf.nn.top_k(tf.reshape(pwm_scores, top_k_shape), k=k_val)
    
    thresholds = tf.reshape(tf.reduce_min(top_k_vals, axis=1, keep_dims=True), [pwm_scores_shape[0], 1, 1, 1])
    pwm_hits_w_location = tf.cast(tf.greater_equal(pwm_scores, thresholds), tf.float32) # out: (example, 1, bp_pos, motif)

    # and aggregate
    pwm_hits_per_example = tf.squeeze(tf.reduce_mean(pwm_hits_w_location, axis=2)) # out: (example, motif)
    outputs["pwm_hits"] = pwm_hits_per_example
    
    return outputs
