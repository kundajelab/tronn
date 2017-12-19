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
from tronn.nets.motif_nets import pwm_consistency_check
from tronn.nets.motif_nets import pwm_positional_max
from tronn.nets.motif_nets import multitask_motif_assignment
from tronn.nets.motif_nets import motif_assignment

from tronn.nets.filter_nets import filter_by_accuracy
from tronn.nets.filter_nets import filter_by_importance
from tronn.nets.filter_nets import filter_singles


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


def importances_to_motif_assignments_v3(features, labels, config, is_training=False):
    """Update to motif assignments

    Returns:
      dict of results
    """
    # set up stack
    inference_stack = [
        (multitask_importances, {"anchors": config["outputs"]["logits"], "importances_fn": input_x_grad, "relu": False}), # importances
        (filter_by_accuracy, {"filter_probs": config["outputs"]["probs"], "acc_threshold": 0.7}), # filter out low accuracy examples
        # TODO - build a correct shuffle null
        #(threshold_shufflenull, {"num_shuffles": 100, "pval": 0.05, "two_tailed": True}), # threshold
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (filter_singles, {"window": 5, "min_fract": 0.4}), # needs to have 3bp within a 5bp window
        # TODOMAYBE stabilize here too? ie filter out things that dont match well across time?
        
        (filter_by_importance, {"cutoff": 10}),
        (normalize_w_probability_weights, {"normalize_probs": config["outputs"]["probs"]}), # normalize, never use logits (too much range) unless clip it
        
        #(clip_edges, {"left_clip": 400, "right_clip": 600}), # clip for active center
        (add_per_example_kval, {"max_k": 4, "motif_len": 5}), # get a kval for each example, use with multitask_threshold_topk_by_example
        (pwm_convolve_inputxgrad, {"pwms": config["pwms"]}),

        # maxpool to deal with slightly shifted motifs and check which motifs most consistent across time
        (pwm_maxpool, {"pool_width": 10}),
        (pwm_consistency_check, {}),

        # take max and threshold
        (pwm_positional_max, {}),
        (multitask_threshold_topk_by_example, {"splitting_axis": 0, "position_axis": 2}), # just keep top k
        
        # moved from after filtration
        (multitask_global_importance, {"append": True}), # get global (abs val)
    ]

    # stack the transforms
    master_config = config
    for transform_fn, config in inference_stack:
        print transform_fn
        master_config.update(config) # update config before and after
        features, labels, config = transform_fn(features, labels, master_config)
        master_config.update(config)
        
    # unstack features by task and attach to config
    features = tf.unstack(features, axis=1)
    outputs = {}
    for i in xrange(len(features)):
        outputs["pwm-counts.taskidx-{}".format(i)] = features[i]
        master_config["outputs"]["pwm-counts.taskidx-{}".format(i)] = features[i]

    # and add labels
    master_config["outputs"]["labels"] = labels
    
    return outputs, labels, master_config


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

