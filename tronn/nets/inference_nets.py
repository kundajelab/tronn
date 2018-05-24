# Description: joins various smaller nets to run analyses after getting predictions

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.nets.importance_nets import multitask_importances
#from tronn.nets.importance_nets import multitask_global_importance
from tronn.nets.importance_nets import filter_by_importance
from tronn.nets.importance_nets import filter_singles_twotailed

from tronn.nets.normalization_nets import normalize_w_probability_weights
from tronn.nets.normalization_nets import normalize_to_weights
from tronn.nets.normalization_nets import normalize_to_delta_logits
#from tronn.nets.normalization_nets import normalize_to_logits
#from tronn.nets.normalization_nets import zscore_and_scale_to_weights

from tronn.nets.threshold_nets import threshold_gaussian
from tronn.nets.threshold_nets import threshold_shufflenull
from tronn.nets.threshold_nets import clip_edges

#from tronn.nets.motif_nets import pwm_convolve_inputxgrad
#from tronn.nets.motif_nets import pwm_maxpool
from tronn.nets.motif_nets import pwm_consistency_check
from tronn.nets.motif_nets import pwm_positional_max
from tronn.nets.motif_nets import pwm_position_squeeze
from tronn.nets.motif_nets import pwm_relu
from tronn.nets.motif_nets import pwm_match_filtered_convolve
from tronn.nets.motif_nets import multitask_global_pwm_scores

from tronn.nets.grammar_nets import multitask_score_grammars
from tronn.nets.grammar_nets import score_distance_to_motifspace_point
from tronn.nets.grammar_nets import check_motifset_presence

from tronn.nets.filter_nets import filter_by_accuracy
from tronn.nets.filter_nets import filter_singleton_labels


#from tronn.nets.filter_nets import filter_by_motifset_presence

from tronn.nets.mutate_nets import generate_mutation_batch
from tronn.nets.mutate_nets import run_model_on_mutation_batch
from tronn.nets.mutate_nets import dfim
from tronn.nets.mutate_nets import motif_dfim
from tronn.nets.mutate_nets import delta_logits
from tronn.nets.mutate_nets import filter_mutation_directionality
from tronn.nets.mutate_nets import blank_motif_sites

from tronn.nets.util_nets import remove_global_task

from tronn.nets.manifold_nets import score_manifold_distances
from tronn.nets.manifold_nets import filter_by_manifold_distance
from tronn.nets.manifold_nets import filter_by_sig_pwm_presence

from tronn.nets.sequence_nets import onehot_to_string


def build_inference_stack(inputs, params, inference_stack):
    """Given the inference stack, build the graph
    """
    outputs = inputs
    master_params = params
    for transform_fn, params in inference_stack:
        print transform_fn
        master_params.update(params) # update config before and after
        outputs, params = transform_fn(outputs, master_params)
        print outputs["features"].get_shape()
        master_params.update(params)

    return outputs, master_params


def unstack_tasks(inputs, params):
    """Unstack by task
    """
    features = inputs.get("features")
    task_axis = params.get("task_axis", 1)
    outputs = dict(inputs)
    
    # unstack
    features = tf.unstack(features, axis=task_axis)
    
    # params
    task_indices = params.get("importance_task_indices")
    assert task_indices is not None
    new_task_indices = list(task_indices)
    if len(features) == len(task_indices) + 1:
        new_task_indices.append("global") # for the situations with a global score
    name = params.get("name", "features")

    # save out with appropriate index    
    for i in xrange(len(features)):
        task_features_key = "{}.taskidx-{}".format(
            name, new_task_indices[i])
        outputs[task_features_key] = features[i]
    
    return outputs, params


def sequence_to_importance_scores(inputs, params):
    """Go from sequence (N, 1, pos, 4) to importance scores (N, 1, pos, 4)
    """
    # params
    params["is_training"] = False
    unstack = params.get("unstack_importances", True)
    
    # set up importance logits
    inputs["importance_logits"] = inputs["logits"]
    
    # set up inference stack
    inference_stack = [
        (multitask_importances, {"relu": False}),
        #(threshold_shufflenull, {"pval_thresh": 0.05}),
        (filter_by_accuracy, {"acc_threshold": 0.7}), # TODO use FDR instead
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}),
        (normalize_to_weights, {"weight_key": "probs"}), 
        (clip_edges, {"left_clip": 400, "right_clip": 600}),
        (filter_by_importance, {"cutoff": 10, "positive_only": True}), 
    ]

    # build inference stack
    outputs, params = build_inference_stack(
        inputs, params, inference_stack)

    # unstack
    if unstack:
        params["name"] = "importances"
        outputs, params = unstack_tasks(outputs, params)
        
    return outputs, params


def sequence_to_importance_scores_unfiltered(inputs, params):
    """Go from sequence (N, 1, pos, 4) to importance scores (N, 1, pos, 4)
    """
    # params
    params["is_training"] = False
    unstack = params.get("unstack_importances", True)
    params["keep_gradients"] = True
    
    # set up importance logits
    inputs["importance_logits"] = inputs["logits"]
    
    # set up inference stack
    inference_stack = [
        (multitask_importances, {"relu": False}),
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}),
        (normalize_to_weights, {"weight_key": "probs"}), 
        (clip_edges, {"left_clip": 400, "right_clip": 600}),
    ]

    # build inference stack
    outputs, params = build_inference_stack(
        inputs, params, inference_stack)

    # unstack
    if unstack:
        params["name"] = "importances"
        outputs, params = unstack_tasks(outputs, params)
        
    return outputs, params


def sequence_to_motif_scores(inputs, params):
    """Go from sequence (N, 1, pos, 4) to motif hits (N, motif)
    """
    # params
    params["is_training"] = False
    #params["unstack_importances"] = False # normally, adjust for debugging
    params["unstack_importances"] = True # normally, adjust for debugging
    params["raw-sequence-key"] = "raw-sequence"
    params["raw-sequence-clipped-key"] = "raw-sequence-clipped"
    params["raw-pwm-scores-key"] = "raw-pwm-scores"
    
    # params
    use_importances = params.get("use_importances", True)
    count_thresh = params.get("count_thresh", 1)
    unstack = params.get("unstack_pwm_scores", True)
    
    if params.get("raw-sequence-key") is not None:
        inputs[params["raw-sequence-key"]] = inputs["features"]
    if params.get("raw-sequence-clipped-key") is not None:
        inputs[params["raw-sequence-clipped-key"]] = inputs["features"]
    
    # if using NN, convert features to importance scores first
    if use_importances:
        inputs, params = sequence_to_importance_scores(inputs, params)
        count_thresh = 2 # there's time info, so can filter across tasks
        
    # set up inference stack
    inference_stack = [
        (pwm_match_filtered_convolve, {}),
        (multitask_global_pwm_scores, {"append": True, "count_thresh": count_thresh}),
        (pwm_position_squeeze, {"squeeze_type": "max"}),
        (pwm_relu, {}), # for now - since we dont really know how to deal with negative sequences yet
    ]

    # build inference stack
    outputs, params = build_inference_stack(
        inputs, params, inference_stack)

    # unstack
    if unstack:
        params["name"] = "pwm-scores"
        outputs, params = unstack_tasks(outputs, params)

    return outputs, params




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
        (multitask_score_grammars, {}),
        (multitask_global_importance, {"append": True, "reduce_type": "max"}),
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

    if config.get("keep_ism_scores") is not None:
        config = unstack_tasks(features, labels, config, prefix=config["keep_ism_scores"])

    del config["outputs"]["onehot_sequence"]
    
    quit()
        
    return features, labels, config



def sequence_to_dmim_old(inputs, params):
    """For a grammar, get back the delta deeplift results on motifs, another way
    to extract dependencies at the motif level
    """
    # params:
    params["raw-sequence-key"] = "raw-sequence"
    params["raw-sequence-clipped-key"] = "raw-sequence-clipped"
    params["raw-pwm-scores-key"] = "raw-pwm-scores"
    params["positional-pwm-scores-key"] = "positional-pwm-scores"
    params["filter_motifspace"] = True
    params["filter_motifset"] = True
    params["dmim-scores-key"] = "dmim-scores"

    # maybe keep
    params["keep_importances"] = None
    
    # get motif scores
    #outputs, params = sequence_to_motif_scores(inputs, params)
    
    method = params.get("importances_fn")

    # set up inference stack
    inference_stack = [
        (sequence_to_motif_scores, {}),
        (score_distance_to_motifspace_point, {"filter_motifspace": True}),
        (check_motifset_presence, {"filter_motifset": True}),
        (generate_mutation_batch, {}),
        (run_model_on_mutation_batch, {}),
        (delta_logits, {"logits_to_features": False}),

        (multitask_importances, {"backprop": method, "relu": False}),
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}), # TODO - some shuffle null here? if so need to generate shuffles
        (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}),
        
        (dfim, {}), # {N, task, 1000, 4}
        

        (normalize_to_delta_logits, {}),

        (blank_motif_sites, {}),
        
        (clip_edges, {"left_clip": 400, "right_clip": 600}),


        
        (pwm_match_filtered_convolve, {"positional-pwm-scores-key": None}),
        (pwm_position_squeeze, {"squeeze_type": "max"}),
        (motif_dfim, {}), # TODO - somewhere here, keep the mutated sequences to read out if desired
        # TODO normalize by probability here?
        (filter_mutation_directionality, {}) # check if this makes sense (in the right order) in the context of blanking things out
    ]

    # build inference stack
    outputs, params = build_inference_stack(
        outputs, params, inference_stack)

    # unstack
    if params.get("dmim-scores-key") is not None:
        params["name"] = params["dmim-scores-key"]
        outputs, params = unstack_tasks(outputs, params)
        
    return outputs, params


def sequence_to_dmim(inputs, params):
    """For a grammar, get back the delta deeplift results on motifs, another way
    to extract dependencies at the motif level
    """
    # params:
    params["raw-sequence-key"] = "raw-sequence"
    params["raw-sequence-clipped-key"] = "raw-sequence-clipped"
    params["raw-pwm-scores-key"] = "raw-pwm-scores"
    params["positional-pwm-scores-key"] = "positional-pwm-scores"
    params["filter_motifspace"] = True
    params["filter_motifset"] = True
    params["dmim-scores-key"] = "dmim-scores"

    # maybe keep
    #params["keep_importances"] = None
    params["keep_importances"] = "importances"
    
    method = params.get("importances_fn")

    # set up inference stack
    inference_stack = [
        # save normal sequence
        (onehot_to_string, {}),
        
        # score motifs on importance scores
        (sequence_to_motif_scores, {}),

        # filter by manifold locations
        (score_manifold_distances, {}),
        (filter_by_manifold_distance, {}),
        (filter_by_sig_pwm_presence, {}),

        # generate mutations and run model and get dfim
        (generate_mutation_batch, {}),
        (run_model_on_mutation_batch, {}),
        (onehot_to_string, {"string_key": "mut_features.string"}),
        (delta_logits, {"logits_to_features": False}),
        (multitask_importances, {"backprop": method, "relu": False}), # check relu - should this be done later?
        
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}), # TODO - some shuffle null here? if so need to generate shuffles
        (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}),
        (normalize_to_weights, {"weight_key": "mut_probs"}), 

        # only keep positives
        (pwm_relu, {}),
        
        #(normalize_to_delta_logits, {}),
        (dfim, {}), # {N, task, 1000, 4}

        (blank_motif_sites, {}),
        (clip_edges, {"left_clip": 400, "right_clip": 600, "clip_string": True}),

        # NOTE: can't filter here - will throw off balance
        #(filter_by_importance, {"cutoff": 10, "positive_only": True}), 

        # scan motifs
        (pwm_match_filtered_convolve, {"positional-pwm-scores-key": None}),
        (pwm_position_squeeze, {"squeeze_type": "max"}),
        (motif_dfim, {}), # TODO - somewhere here, keep the mutated sequences to read out if desired?

        # TODO some kind of filter here on the dmim scores (remove those with no delta dmim)?
        
        #(filter_mutation_directionality, {}) # check if this makes sense (in the right order) in the context of blanking things out
    ]

    # build inference stack
    outputs, params = build_inference_stack(
        inputs, params, inference_stack)

    # unstack
    if params.get("dmim-scores-key") is not None:
        params["name"] = params["dmim-scores-key"]
        outputs, params = unstack_tasks(outputs, params)
        
    return outputs, params

