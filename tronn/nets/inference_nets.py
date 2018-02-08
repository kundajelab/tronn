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
from tronn.nets.motif_nets import pwm_position_squeeze
from tronn.nets.motif_nets import pwm_relu
from tronn.nets.motif_nets import pwm_match_filtered_convolve

from tronn.nets.motif_nets import multitask_motif_assignment
from tronn.nets.motif_nets import motif_assignment

from tronn.nets.grammar_nets import score_grammars

from tronn.nets.filter_nets import filter_by_accuracy
from tronn.nets.filter_nets import filter_by_importance
from tronn.nets.filter_nets import filter_singles_twotailed
from tronn.nets.filter_nets import filter_by_grammar_presence


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
    inference_stack = [
        (multitask_importances, {"anchors": config["outputs"]["logits"], "importances_fn": input_x_grad, "relu": False}), # importances
        (filter_by_accuracy, {"filter_probs": config["outputs"]["probs"], "acc_threshold": 0.7}), # filter out low accuracy examples TODO use FDR instead
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

    # if using NN, convert features to importance scores first
    if use_importances:
        features, labels, config = sequence_to_importance_scores(
            features, labels, config, is_training=is_training) # (1) TRUE for viz, FALSE otherwise
        count_thresh = 2 # there's time info, so can filter across tasks
        
    # set up inference stack
    inference_stack = [
        (pwm_match_filtered_convolve, {"pwms": config["pwms"]}), # double filter: raw seq match and impt weighted seq match
        (pwm_consistency_check, {}), # get consistent across time scores
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


# set up grammars net
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
    # TODO - make sure to keep importance scores to run modisco
    features, labels, config = sequence_to_motif_scores(
        features, labels, config, is_training=is_training)

    # for grammar scan, want to start from an intermediate
    features = config["outputs"]["pwm-scores-full"] # (N, task, pos, motif) <- keep this output (for visualizing positions)
    del config["outputs"]["pwm-scores-full"] # remove after setting up as features
    
    # set up inference stack
    inference_stack = [
        # TODO - may need to zero out the hits that overlap by N bp (they are covering the same importance scores)
        
        (pwm_position_squeeze, {"squeeze_type": "max"}), # squeeze = (N, task, M)
        # TODO - this is actually a really bad filter, bc it will grab all the motifs around the top sequence
        # really want to do something more spaced out - like assign the importances to just one motif....
        (multitask_threshold_topk_by_example, {}), # cutoff for top 10 motifs to reduce noise <- probably this bit is slow?
        (score_grammars, {"grammars": config["grammars"], "pwms": config["pwms"]}), #  {N, task, G} 
        (multitask_global_importance, {"append": True, "reduce_type": "max"}), # N, task+1, G <- keep this output (when does grammar turn on)
        (filter_by_grammar_presence, {}) # filter stage, keeps last outputs (N, task+1, G)
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
        features, labels, config, is_training=is_training, keep_outputs=False)

    inference_stack = [
        
    ]

    # then with the set of positive hits, run ISM centered on a key motif
    # 1) break the motif, using info from the motif x pos matrix info
    # 2) pass through and get importances -
    #    probably need to change variable scope to prevent tensorflow confusion
    # 3) then subtract reference from broken - this is the delta deeplift part
    # 4) then run the motif scan again. {N, M}. positive AND negative are informative
    # 5) reduce_sum to calculate the summed delta for each motif (relative to the master motif) {N, M}


    # build inference stack
    features, labels, config = build_inference_stack(
        features, labels, config, inference_stack)

    if keep_outputs:
        config = unstack_tasks(features, labels, config, prefix="grammar-scores")
    
    return features, labels, config

# TODO - somewhere (datalayer?) build module for generating synthetic sequences


# OLD CODE BELOW TO DELETE



def importances_to_motif_assignments_v3(features, labels, config, is_training=False):
    """Update to motif assignments

    Returns:
      dict of results
    """

    # important principles: utilizing consistency across tasks, filtering out noise, proper normalization
    
    # short todos:
    # 2) normalizing motifs by info content - number of bases that are actually nonzero?
    # 3) increase number of importance base pairs needed (to account for noise) - maybe up to 20 needed?
    
    # set up stack
    inference_stack = [
        (multitask_importances, {"anchors": config["outputs"]["logits"], "importances_fn": input_x_grad, "relu": False}), # importances
        (filter_by_accuracy, {"filter_probs": config["outputs"]["probs"], "acc_threshold": 0.7}), # filter out low accuracy examples
        # TODO - build a correct shuffle null
        # ^ do this by adding on extra shuffled sequences to a batch (pass along the ratio and indices)
        # then after using the extra shuffled sequences, can discard and only pass along real sequences into a queue
        ##(threshold_shufflenull, {"num_shuffles": 100, "pval": 0.05, "two_tailed": True}), # threshold
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}), # needs to have 2bp within a 5bp window.
        # TODOMAYBE stabilize here too? ie filter out things that dont match well across time?
        
        (filter_by_importance, {"cutoff": 20}),
        (normalize_w_probability_weights, {"normalize_probs": config["outputs"]["probs"]}), # normalize, never use logits (too much range) unless clip it
        
        ##(clip_edges, {"left_clip": 400, "right_clip": 600}), # clip for active center
        #(add_per_example_kval, {"max_k": 4, "motif_len": 5}), # get a kval for each example, use with multitask_threshold_topk_by_example, tighten this up?
        #(add_sumpool_threshval, {"width": 10, "stride": 1, "fract": 2/25.}),
        
        #(pwm_convolve_inputxgrad, {"pwms": config["pwms"]}),
        (pwm_match_filtered_convolve, {"pwms": config["pwms"]}),

        # maxpool to deal with slightly shifted motifs and check which motifs most consistent across time
        #(pwm_maxpool, {"pool_width": 10}), # from 10
        (pwm_consistency_check, {}),

        (multitask_global_importance, {"append": True}), # get global (abs val)
        
        # take max and threshold
        # this is an important step to reduce noise
        # sources of noise - confounding of match with the importance score strength
        # solving this still leaves the possibility of a match simply due to noise - maybe weight by num of nonzero bp in the filter range?
        # maybe a double filter - first select position with best match on binary sequence (no weighting, just 1 or 0)
        # also consider a minimum match threshold. since log likelihood, greater than zero?
        # and then with that, filter?

        # filter steps to add:
        # 1) PWM raw sequence filter (ie on raw sequence, score > 0) - do this in the pwm step above. don't do max because what if multiple sites in region?
        #    actually max probably ok for now (even top 3, for later)
        # 2) spread the score across the filter and intersect with nonzero basepairs, and re-sum - this gives you a weighted score
        #    another way to do this is to just do a sumpool in the filter range, divide by filter size, and multiply the score by that val
        # 3) then from those scores take max val across positions (below: the pwm_position_squeeze)
        
        #(pwm_positional_max, {}), # change squeeze back to True
        (pwm_position_squeeze, {"squeeze_type": "max"}),
        (pwm_relu, {}), # for now - since we dont really know how to deal with negative sequences yet
        #(multitask_threshold_topk_by_example, {"splitting_axis": 0, "position_axis": 2}), # just keep top k
        #(apply_sumpool_thresh, {}),
        
        # moved from after filtration

    ]

    # stack the transforms
    master_config = config
    for transform_fn, config in inference_stack:
        print transform_fn
        master_config.update(config) # update config before and after
        features, labels, config = transform_fn(features, labels, master_config)
        print features.get_shape()
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

