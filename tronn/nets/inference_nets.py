# Description: joins various smaller nets to run analyses after getting predictions

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.nets.importance_nets import multitask_importances
from tronn.nets.importance_nets import multitask_global_importance

from tronn.nets.normalization_nets import normalize_w_probability_weights
from tronn.nets.normalization_nets import normalize_to_logits
from tronn.nets.normalization_nets import zscore_and_scale_to_weights

from tronn.nets.threshold_nets import threshold_gaussian
from tronn.nets.threshold_nets import threshold_shufflenull
from tronn.nets.threshold_nets import clip_edges

from tronn.nets.motif_nets import pwm_convolve_inputxgrad
from tronn.nets.motif_nets import pwm_maxpool
from tronn.nets.motif_nets import pwm_consistency_check
from tronn.nets.motif_nets import pwm_positional_max
from tronn.nets.motif_nets import pwm_position_squeeze
from tronn.nets.motif_nets import pwm_relu
from tronn.nets.motif_nets import pwm_match_filtered_convolve

from tronn.nets.grammar_nets import multitask_score_grammars

from tronn.nets.filter_nets import filter_by_accuracy
from tronn.nets.filter_nets import filter_by_importance
from tronn.nets.filter_nets import filter_singles_twotailed
from tronn.nets.filter_nets import filter_by_grammar_presence

from tronn.nets.mutate_nets import motif_ism

from tronn.nets.util_nets import remove_global_task


def build_inference_stack(features, labels, config, inference_stack):
    """Given the inference stack, build the graph
    """
    master_config = config
    for transform_fn, config in inference_stack:
        print transform_fn
        master_config.update(config) # update config before and after
        features, labels, config = transform_fn(features, labels, master_config)
        print features.get_shape()
        master_config.update(config)

    return features, labels, config


def unstack_tasks(features, labels, config, prefix="features", task_axis=1):
    """Unstack by task (axis=1)
    """
    features = tf.unstack(features, axis=task_axis)
    #outputs = {}
    for i in xrange(len(features)):
        task_features_key = "{}.taskidx-{}".format(prefix, i)
        #outputs[task_features_key] = features[i]
        config["outputs"][task_features_key] = features[i]

    # and add labels
    config["outputs"]["labels"] = labels

    return config


def sequence_to_importance_scores(
        features,
        labels,
        config,
        is_training=False):
    """Go from sequence (N, 1, pos, 4) to importance scores (N, 1, pos, 4)
    """
    method = config.get("importances_fn")
    
    inference_stack = [
        (multitask_importances, {"backprop": method, "relu": False}), # importances
        (filter_by_accuracy, {"acc_threshold": 0.7}), # filter out low accuracy examples TODO use FDR instead
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}), # needs to have 2bp within a 7bp window.
        (filter_by_importance, {"cutoff": 10, "positive_only": True}), # TODO - change this to positive cutoff?
        (normalize_w_probability_weights, {}), # normalize, never use logits (too much range) unless clip it
    ]
    
    # set up inference stack
    features, labels, config = build_inference_stack(
        features, labels, config, inference_stack)

    # unstack
    if config.get("keep_importances") is not None:
        config = unstack_tasks(features, labels, config, prefix=config["keep_importances"])
        
    return features, labels, config


def sequence_to_motif_scores(
        features,
        labels,
        config,
        is_training=False):
    """Go from sequence (N, 1, pos, 4) to motif hits (N, motif)
    """
    use_importances = config.get("use_importances", True)
    count_thresh = config.get("count_thresh", 1)
    assert use_importances is not None

    keep_key = config.get("keep_onehot_sequence")
    if keep_key is not None:
        config["outputs"][keep_key] = features
    
    # if using NN, convert features to importance scores first
    if use_importances:
        features, labels, config = sequence_to_importance_scores(
            features, labels, config, is_training=is_training)
        count_thresh = 2 # there's time info, so can filter across tasks
        
    # set up inference stack
    inference_stack = [
        (pwm_match_filtered_convolve, {"pwms": config["pwms"]}), # double filter: raw seq match and impt weighted seq match
        (multitask_global_importance, {"append": True, "count_thresh": count_thresh}), # get global (abs val), keep_features = global-pwm-scores
        (pwm_position_squeeze, {"squeeze_type": "max"}), # get the max across positions {N, motif} # TODO - give an option for counts vs max (homotypic grammars)
        (pwm_relu, {}), # for now - since we dont really know how to deal with negative sequences yet
    ]

    # build inference stack
    features, labels, config = build_inference_stack(
        features, labels, config, inference_stack)

    # unstack
    if config.get("keep_pwm_scores") is not None:
        config = unstack_tasks(features, labels, config, prefix=config["keep_pwm_scores"])

    return features, labels, config


def sequence_to_grammar_scores(
        features,
        labels,
        config,
        is_training=False):
    """Go from sequence (N, 1, pos, 4) to grammar hits (N, grammar)

    Use this inference stack to get:
      - viz importance scores on grammars
      - run modisco
      - look at pairwise motif positions

    """
    # first go from sequence to motifs
    features, labels, config = sequence_to_motif_scores(
        features, labels, config, is_training=is_training)
    
    # set up inference stack
    inference_stack = [
        (remove_global_task, {}),
        (multitask_score_grammars, {}),
        (multitask_global_importance, {"append": True, "reduce_type": "mean"}), # for now, just average across tasks for final score
        #(filter_by_grammar_presence, {}) # filter stage, keeps last outputs (N, task+1, G) DO NOT FILTER ALWAYS
    ]

    # build inference stack
    features, labels, config = build_inference_stack(
        features, labels, config, inference_stack)

    if config.get("keep_grammar_scores") is not None:
        config = unstack_tasks(features, labels, config, prefix=config["keep_grammar_scores"])

    return features, labels, config


def sequence_to_grammar_ism(features, labels, config, is_training=False):
    """Go from sequence (N, 1, pos, 4) to ism/deltadeeplift results (N, 1, motif), where 1=1 motif

    Use this inference stack to get:
      -- deltadeeplift

    """
    # use sequence_to_grammar_scores above
    features, labels, config = sequence_to_grammar_scores(
        features, labels, config, is_training=is_training)

    inference_stack = [
        (motif_ism, {})
        
    ]

    # TODO - deltadeeplift is probably a separate function?
    # 3) then subtract reference from broken - this is the delta deeplift part
    # 4) then run the motif scan again. {N, M}. positive AND negative are informative
    # 5) reduce_sum to calculate the summed delta for each motif (relative to the master motif) {N, M}


    # build inference stack
    features, labels, config = build_inference_stack(
        features, labels, config, inference_stack)

    config = unstack_tasks(features, labels, config, prefix="grammar-scores")
    
    return features, labels, config

# TODO - somewhere (datalayer?) build module for generating synthetic sequences

