# Description: joins various smaller nets to run analyses after getting predictions

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.nets.importance_nets import input_x_grad
from tronn.nets.importance_nets import multitask_importances
from tronn.nets.importance_nets import multitask_global_importance

from tronn.nets.normalization_nets import normalize_w_probability_weights
from tronn.nets.normalization_nets import normalize_to_logits
from tronn.nets.normalization_nets import zscore_and_scale_to_weights

from tronn.nets.threshold_nets import threshold_topk_by_example
from tronn.nets.threshold_nets import multitask_threshold_topk_by_example
from tronn.nets.threshold_nets import add_per_example_kval
from tronn.nets.threshold_nets import threshold_gaussian
from tronn.nets.threshold_nets import threshold_shufflenull
from tronn.nets.threshold_nets import clip_edges


from tronn.nets.motif_nets import pwm_convolve_v3
from tronn.nets.motif_nets import pwm_convolve_inputxgrad
from tronn.nets.motif_nets import pwm_maxpool
from tronn.nets.motif_nets import pwm_positional_max
from tronn.nets.motif_nets import multitask_motif_assignment
from tronn.nets.motif_nets import motif_assignment


def get_importances(features, labels, config, is_training=False):
    """Get importance scores
    """
    # set up stack
    inference_stack = [
        (multitask_importances, {"anchors": config["importance_logits"], "importances_fn": input_x_grad}),
        (threshold_shufflenull, {"num_shuffles": 200, "pval": 0.02, "two_tailed": True}), # be more stringent
        #(normalize_w_probability_weights, {"probs": config["importance_probs"]}), # this normalization is weak, normalize pos and neg separately?
        #(normalize_to_logits, {"logits": config["importance_logits"]}),
        (multitask_global_importance, {"append": True}),
    ]

    # stack the transforms
    for transform_fn, config in inference_stack:
        print transform_fn
        features, labels, config = transform_fn(features, labels, config)

    # TODO: allow option for unstacking importances as needed
    features = tf.unstack(features, axis=1)
    
    outputs = {}
    for i in xrange(len(features)):
        outputs["importances.taskidx-{}".format(i)] = tf.expand_dims(features[i], axis=1)
        
    return outputs, labels, config


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
        (multitask_global_importance, {"append": True}),
        # TODO - split out pwm convolution from motif assignment
        (multitask_motif_assignment, {"pwms": config["pwms"], "k_val": 4, "motif_len": 5, "pool": True}),
        #(threshold_topk, {"k_val": 4})
    ]

    # stack the transforms
    for transform_fn, config in inference_stack:
        print transform_fn
        features, labels, config = transform_fn(features, labels, config)
        # TODO - if needed, pass on additional configs through

    # TODO separate out results into separate task sets
    features = tf.unstack(features, axis=1)
    
    outputs = {}
    for i in xrange(len(features)):
        outputs["pwm-counts.taskidx-{}".format(i)] = features[i]
        
    return outputs, labels, config


def importances_to_motif_assignments_v2(features, labels, config, is_training=False):
    """Update to motif assignments
    
    1) get importances
    2) threshold with shuffled null (pvals)
    3) normalize to probs (absolute val)
    4) and take max (abs val) for global
    5) work through motif assignment

    note: important how to think about a normalized importance score. in classification sense,
    probabilities near 1 and 0 are more certain than near 0.5. So importance scores should be
    scaled to those ends.

    Returns:
      dict of results
    """
    # set up stack
    inference_stack = [
        (multitask_importances, {"anchors": config["importance_logits"], "importances_fn": input_x_grad}), # importances
        (threshold_shufflenull, {"num_shuffles": 100, "pval": 0.05, "two_tailed": True}), # threshold
        (normalize_w_probability_weights, {"probs": config["importance_probs"]}), # normalize
        # TODO - just get middle 200? If so do this after thresholding and normalization
        (clip_edges, {"left_clip": 400, "right_clip": 600}),
        #(zscore_and_scale_to_weights, {"logits": config["importance_logits"]}), # normalize
        #(zscore_and_scale_to_weights, {"probs": config["importance_probs"]}), # normalize
        (multitask_global_importance, {"append": True}), # get global (abs val)
        # replace this part of stack with get_importances
        
        (multitask_motif_assignment, {"pwms": config["pwms"], "k_val": 4, "motif_len": 5, "pool": True}),
        #(threshold_topk, {"k_val": 4})
    ]

    # stack the transforms
    for transform_fn, config in inference_stack:
        print transform_fn
        features, labels, config = transform_fn(features, labels, config)
        # TODO - if needed, pass on additional configs through

    # TODO separate out results into separate task sets
    features = tf.unstack(features, axis=1)
    
    outputs = {}
    for i in xrange(len(features)):
        outputs["pwm-counts.taskidx-{}".format(i)] = features[i]
        
    return outputs, labels, config


def importances_to_motif_assignments_v3(features, labels, config, is_training=False):
    """Update to motif assignments

    Returns:
      dict of resuults
    """
    # set up stack
    inference_stack = [
        (multitask_importances, {"anchors": config["importance_logits"], "importances_fn": input_x_grad}), # importances
        (threshold_shufflenull, {"num_shuffles": 100, "pval": 0.05, "two_tailed": True}), # threshold
        (normalize_w_probability_weights, {"probs": config["importance_probs"]}), # normalize
        (clip_edges, {"left_clip": 400, "right_clip": 600}), # clip for active center
        (multitask_global_importance, {"append": True}), # get global (abs val)
        #(add_per_example_kval, {"k_val": 4, "motif_len": 5}), # get a kval for each example, use with multitask_threshold_topk_by_example
        (pwm_convolve_inputxgrad, {"pwms": config["pwms"]}),
        (pwm_maxpool, {"pool_width": 10}),
        (pwm_positional_max, {}),
        (multitask_threshold_topk_by_example, {"splitting_axis": 0, "position_axis": 2}) # just keep top k
    ]

    # stack the transforms
    master_config = {}
    for transform_fn, config in inference_stack:
        print transform_fn
        master_config.update(config) # update config before and after
        features, labels, config = transform_fn(features, labels, master_config)
        master_config.update(config)
    
    # unstack features by task
    features = tf.unstack(features, axis=1)
    outputs = {}
    for i in xrange(len(features)):
        outputs["pwm-counts.taskidx-{}".format(i)] = features[i]
        
    return outputs, labels, config


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

