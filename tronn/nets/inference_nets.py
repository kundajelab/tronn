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
from tronn.nets.grammar_nets import score_distance_to_motifspace_point
from tronn.nets.grammar_nets import check_motifset_presence

from tronn.nets.filter_nets import filter_by_accuracy
from tronn.nets.filter_nets import filter_by_importance
from tronn.nets.filter_nets import filter_singles_twotailed
from tronn.nets.filter_nets import filter_by_motifset_presence

from tronn.nets.mutate_nets import generate_mutation_batch
from tronn.nets.mutate_nets import run_model_on_mutation_batch
from tronn.nets.mutate_nets import dfim
from tronn.nets.mutate_nets import motif_dfim
from tronn.nets.mutate_nets import delta_logits

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
    task_indices = config.get("importance_task_indices")
    new_task_indices = list(task_indices)
    if len(features) == len(task_indices) + 1:
        new_task_indices.append("global") # for the situations with a global score
    assert task_indices is not None
    for i in xrange(len(features)):
        task_features_key = "{}.taskidx-{}".format(
            prefix, new_task_indices[i])
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
        (multitask_importances, {"backprop": method, "relu": False}),
        (filter_by_accuracy, {"acc_threshold": 0.7}), # filter out low accuracy examples TODO use FDR instead
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}),
        (normalize_w_probability_weights, {}), 
        (clip_edges, {"left_clip": 400, "right_clip": 600}),
        (filter_by_importance, {"cutoff": 10, "positive_only": True}), 
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
        config["outputs"]["{}_clipped".format(keep_key)] = features
    
    # if using NN, convert features to importance scores first
    if use_importances:
        features, labels, config = sequence_to_importance_scores(
            features, labels, config, is_training=is_training)
        count_thresh = 2 # there's time info, so can filter across tasks
        
    # set up inference stack
    inference_stack = [
        (pwm_match_filtered_convolve, {"pwms": config["pwms"]}),
        (multitask_global_importance, {"append": True, "count_thresh": count_thresh}),
        (pwm_position_squeeze, {"squeeze_type": "max"}),
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

    features = tf.expand_dims(config["outputs"]["pwm-scores-raw"], axis=1)
    
    # set up inference stack
    inference_stack = [
        #(remove_global_task, {}),
        (multitask_score_grammars, {}),
        (multitask_global_importance, {"append": True, "reduce_type": "max"}),
        #(filter_by_grammar_presence, {}) # filter stage, keeps last outputs (N, task+1, G) DO NOT FILTER ALWAYS
    ]

    # build inference stack
    features, labels, config = build_inference_stack(
        features, labels, config, inference_stack)

    if config.get("keep_grammar_scores") is not None:
        config = unstack_tasks(features, labels, config, prefix=config["keep_grammar_scores"])

    # TODO do this right later
    del config["outputs"]["onehot_sequence"]

    return features, labels, config


def sequence_to_motif_ism(features, labels, config, is_training=False):
    """Go from sequence (N, 1, pos, 4) to ism results (N, task, mutation)
    """
    # get motif scores
    features, labels, config = sequence_to_motif_scores(
        features, labels, config, is_training=is_training)

    # start from raw pwm scores to filter sequences
    features = tf.expand_dims(config["outputs"]["pwm-scores-raw"], axis=1)

    # set up inference stack
    inference_stack = [
        (score_distance_to_motifspace_point, {"filter_motifspace": True}),
        (check_motifset_presence, {"filter_motifset": True}),
        (generate_mutation_batch, {}), # note that these use importance weighted position maps
        (run_model_on_mutation_batch, {"pairwise_mutate": True}),
        # TODO from here just need to extract the logits
        (delta_logits, {}),
    ]

    # build inference stack
    features, labels, config = build_inference_stack(
        features, labels, config, inference_stack)

    #if config.get("keep_ism_results") is not None:
    if True:
        config = unstack_tasks(features, labels, config, prefix=config["keep_ism_results"])

    print config["outputs"].keys()

    quit()
        
    return features, labels, config


def sequence_to_dmim(features, labels, config, is_training=False):
    """For a grammar, get back the delta deeplift results on motifs, another way
    to extract dependencies at the motif level
    """
    # get motif scores
    features, labels, config = sequence_to_motif_scores(
        features, labels, config, is_training=is_training)

    # start from the raw pwm scores to filter sequences
    features = tf.expand_dims(config["outputs"]["pwm-scores-raw"], axis=1)

    # set up inference stack
    inference_stack = [
        (score_distance_to_motifspace_point, {"filter_motifspace": True}),
        (check_motifset_presence, {"filter_motifset": True}),
        (generate_mutation_batch, {}), # note that these use importance weighted position maps
        (run_model_on_mutation_batch, {}), 
        (sequence_to_importance_scores, {}), # {N, task, 200, 4}
        (dfim, {}), # {N, task, 200, 4}
        (sequence_to_motif_scores, {"use_importances": False}),
        (remove_global_task, {}),
        (motif_dfim, {})
    ]

    # build inference stack
    features, labels, config = build_inference_stack(
        features, labels, config, inference_stack)

    if True:
    #if config.get("keep_delta_deeplift_scores") is not None:
        config = unstack_tasks(features, labels, config, prefix="deltadeeplift-results")

    del config["outputs"]["onehot_sequence"]

    quit()
    
    return features, labels, config


# TODO another function to take outputs from either dmim or motif_ism
# and filter and then output sequences in original ACGT format

# TODO - somewhere (datalayer?) build module for generating synthetic sequences

