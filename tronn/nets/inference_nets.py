# Description: joins various smaller nets to run analyses after getting predictions

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.nets.importance_nets import input_x_grad
from tronn.nets.importance_nets import multitask_importances
from tronn.nets.importance_nets import multitask_global_importance

from tronn.nets.normalization_nets import normalize_w_probability_weights

from tronn.nets.threshold_nets import threshold_topk
from tronn.nets.threshold_nets import threshold_gaussian

from tronn.nets.motif_nets import pwm_convolve_v3
from tronn.nets.motif_nets import multitask_motif_assignment
from tronn.nets.motif_nets import motif_assignment


def get_importances(features, labels, config, is_training=False):
    """Get importance scores
    """
    # set up stack
    inference_stack = [
        (multitask_importances, {"anchors": tf.unstack(config["logits"], axis=1), "importances_fn": input_x_grad}),
        (multitask_global_importance, {"append": True}),
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (normalize_w_probability_weights, {"probs": tf.unstack(config["probs"], axis=1)}),
    ]

    # stack the transforms
    for transform_fn, config in inference_stack:
        features, labels, config = transform_fn(features, labels, config)
        # TODO - if needed, pass on additional configs through

    # TODO: allow option for unstacking importances as needed
        
        
    return features, labels, config


def sequence_to_motif_assignments(features, labels, config, is_training=False):
    """Go straight from raw sequence to motif scores. Do not pass (through) NN.
    """
    # set up stack
    inference_stack = [
        (motif_assignment, {"pmws": config["pwms"], "k_val": 4, "motif_len": 5, "pool": True}),
        (threshold_topk, {"k_val": 4})
    ]

    # stack the transforms
    for transform_fn, config in inference_stack:
        features, labels, config = transform_fn(features, labels, config)
        # TODO - if needed, pass on additional configs through
        
    return features, labels, config


def importances_to_motif_assignments(features, labels, config, is_training=False):
    """Update to motif assignments
    
    Returns:
      dict of results
    """
    # set up stack
    inference_stack = [
        (multitask_importances, {"anchors": config["importance_logits"], "importances_fn": input_x_grad}),
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (normalize_w_probability_weights, {"probs": config["importance_probs"]}),
        (multitask_global_importance, {"append": False}),
        (motif_assignment, {"pwms": config["pwms"], "k_val": 4, "motif_len": 5, "pool": True}),
        #(threshold_topk, {"k_val": 4})
    ]

    # stack the transforms
    for transform_fn, config in inference_stack:
        print transform_fn
        features, labels, config = transform_fn(features, labels, config)
        # TODO - if needed, pass on additional configs through

    features = {"pwm-counts.global": features}
            
    return features, labels, config


def get_top_k_motif_hits(features, labels, config, is_training=False):
    """Top k style, just on global importances
    """
    # set up stack
    inference_stack = [
        (multitask_importances, {"anchors": tf.unstack(config["logits"], axis=1), "importances_fn": input_x_grad}),
        (multitask_global_importance, {"append": False}),
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (normalize_w_probability_weights, {"probs": tf.unstack(config["probs"], axis=1)}),
        (pwm_convolve3, {"pwms": config["pwms"], "pool": True}),
        (threshold_topk, {"k_val": 4})
    ]

    # stack the transforms
    for transform_fn, config in inference_stack:
        features, labels, config = transform_fn(features, labels, config)
        # TODO - if needed, pass on additional configs through
    

    return features, labels, config

