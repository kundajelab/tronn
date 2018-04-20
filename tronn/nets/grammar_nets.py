"""Contains nets to help run grammars
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_utils import get_fan_in
from tronn.util.initializers import pwm_simple_initializer

from tronn.nets.filter_nets import filter_and_rebatch


def score_distance_to_motifspace_point(inputs, params):
    """Given grammars, pull out the motifspace information and score distance
    to the motifspace vector, attach a threshold mask to config as needed
    """
    # assertions
    assert params.get("raw-pwm-scores-key") is not None
    assert inputs.get(params["raw-pwm-scores-key"]) is not None
    assert params.get("pwms") is not None
    assert params.get("grammars") is not None

    # get features and pass on rest
    print inputs[params["raw-pwm-scores-key"]]
    features = tf.expand_dims(inputs[params["raw-pwm-scores-key"]], axis=1)
    print features
    #import ipdb
    #ipdb.set_trace()
    
    grammars = params["grammars"]
    pwms = params["pwms"]
    grammars = grammars[0] # for now. TODO fix this later
    outputs = dict(inputs)
    
    # set up vectors
    motifspace_vectors = np.zeros((1, 1, len(pwms), len(grammars))) # {1, 1, M, G}
    motifspace_weights = np.zeros((1, 1, len(pwms), len(grammars))) # {1, 1, M, G}
    motifspace_thresholds = np.zeros((1, 1, len(grammars))) # {1, 1, G}

    # store values
    for grammar_idx in xrange(len(grammars)):
        motifspace_vectors[:, :, :, grammar_idx] = grammars[grammar_idx].motifspace_vector
        motifspace_weights[:, :, :, grammar_idx] = grammars[grammar_idx].motifspace_weights
        motifspace_thresholds[:, :, grammar_idx] = grammars[grammar_idx].motifspace_threshold
    
    # convert to tensors
    motifspace_vectors = tf.convert_to_tensor(motifspace_vectors, dtype=tf.float32)
    motifspace_weights = tf.convert_to_tensor(motifspace_weights, dtype=tf.float32)
    motifspace_thresholds = tf.convert_to_tensor(motifspace_thresholds, dtype=tf.float32)

    # get distance (dot product) on the sequence
    weighted_features = tf.multiply(
        tf.expand_dims(features, axis=3),
        motifspace_weights)
    similarities = tf.reduce_sum(
        tf.multiply(weighted_features, motifspace_vectors), axis=2)
    
    #differences = tf.subtract(weighted_features, motifspace_vectors)
    #similarities = tf.norm(differences, axis=2)
    #similarities_thresholded = tf.cast(
    #    tf.less_equal(similarities, motifspace_thresholds), tf.float32) # {N, 1, G}
    
    similarities_thresholded = tf.cast(
        tf.greater_equal(similarities, motifspace_thresholds), tf.float32) # {N, 1, G}

    # save to config outputs
    outputs["motifspace_dists"] = similarities
    outputs["motifspace_mask"] = similarities_thresholded
    outputs["features"] = features
    
    if params.get("filter_motifspace", False) == True:
        # filter
        outputs["condition_mask"] = tf.greater(
            tf.reduce_max(similarities_thresholded, axis=[1,2]), [0])
        params["name"] = "motifspace_filter"
        outputs, params = filter_and_rebatch(outputs, params)
    
    return outputs, params


def check_motifset_presence(inputs, params):
    """Given grammars check motifs in the grammars to make sure they're all there
    """
    # assertions
    assert params.get("grammars") is not None
    assert params.get("pwms") is not None

    # features and pass rest through
    features = inputs["features"]
    outputs = dict(inputs)
    
    # params
    grammars = params["grammars"][0] # for now, can only run 1 at a time
    pwms = params.get("pwms")
    print "WARNING ONLY RUNNING ONE GRAMMAR"

    # set up features
    motifset_features = tf.expand_dims(features, axis=3) # {N, 1, M, 1}
    
    # input - {N, 1, M}, ie 1 cell state
    pwm_thresholds = np.zeros((1, 1, len(pwms), len(grammars)))

    #motif_counts_by_grammar = []
    for grammar_idx in xrange(len(grammars)):
        # add pwm thresholds and total counts
        pwm_thresholds[:,:,:,grammar_idx] = grammars[grammar_idx].pwm_thresholds
        #motif_counts_by_grammar.append(
        #    np.sum(grammars[grammar_idx].pwm_thresholds > 0).astype(np.int32))
        
    # convert to tensors
    pwm_thresholds = tf.convert_to_tensor(pwm_thresholds, dtype=tf.float32)
    
    # mask unless had presence of all key motifs ABOVE a certain threshold
    grammar_motif_presence = tf.cast(tf.greater(pwm_thresholds, [0]), tf.float32) #  {1, 1, M, G}
    grammar_motif_count = tf.reduce_sum(grammar_motif_presence, axis=2) # {1, 1, G}
    pointwise_presence = tf.cast(tf.greater(
        tf.multiply(
            grammar_motif_presence,
            motifset_features), pwm_thresholds), tf.float32) # {N, 1, M, G}
    motif_present_counts = tf.reduce_sum(pointwise_presence, axis=2) #  {N, 1, G}
    score_mask = tf.cast(tf.equal(motif_present_counts, grammar_motif_count), tf.float32) # {N, 1, G}

    outputs["motifset_total"] = motif_present_counts
    
    if params.get("filter_motifset", False) == True:
        # filter
        outputs["condition_mask"] = tf.greater(
            tf.reduce_max(score_mask, axis=[1,2]), [0])
        params["name"] = "motifset_filter"
        outputs, params = filter_and_rebatch(outputs, params)
    
    return outputs, params


def generalized_jaccard_similarity(features, compare_tensor):
    """Given a batch of features, compare to the tensor and 
    get back generalized jaccard similarity
    """
    # compare tensor: {N, 1, M, G}
    features = tf.expand_dims(features, axis=3) # {N, 1, M, 1}
    
    min_vals = tf.reduce_sum(
        tf.minimum(features, compare_tensor), axis=2) # {N, 1, G}
    max_vals = tf.reduce_sum(
        tf.maximum(features, compare_tensor), axis=2) # {N, 1, G}

    # jaccard
    similarity = tf.divide(min_vals, max_vals) # {N, 1, G}
    
    return similarity


# write a single task version, then compile into multitask
def score_grammars(features, labels, config, is_training=False):
    """load in grammar
    """
    # features {N, 1, M}
    grammars = config.get("grammars")
    pwms = config.get("pwms")
    assert grammars is not None
    assert pwms is not None
    
    # input - {N, 1, M}, ie 1 cell state
    # generate two array maps - (1, 1, M, G), and (1, 1, M, M, G)
    pointwise_weights = np.zeros((1, 1, len(pwms), len(grammars)))
    pairwise_weights = np.zeros((1, 1, len(pwms), len(pwms), len(grammars)))

    # TODO assertions to make sure grammars were created in the same ways
    motif_counts_by_grammar = []
    for grammar_idx in xrange(len(grammars)):
        # add pointwise weights
        pointwise_weights[:,:,:,grammar_idx] = grammars[grammar_idx].pwm_vector
        motif_counts_by_grammar.append(
            np.sum(grammars[grammar_idx].pwm_vector).astype(np.int32))
        
        # add pairwise weights
        pairwise_weights[:,:,:,:,grammar_idx] = grammars[grammar_idx].adjacency_matrix
        
    # convert to tensors
    pointwise_weights = tf.convert_to_tensor(pointwise_weights, dtype=tf.float32)
    pairwise_weights = tf.convert_to_tensor(pairwise_weights, dtype=tf.float32)

    # adjust feature tensor dimensions
    pointwise_features = tf.expand_dims(features, axis=3) # {N, 1, M, 1}
    pairwise_features = tf.expand_dims(
        tf.multiply(
            tf.expand_dims(features, axis=3),
            tf.expand_dims(features, axis=2)), axis=4) # {N, 1, M, M, 1}
    
    # linear model
    pointwise_scores = tf.reduce_sum(
        tf.multiply(
            pointwise_weights,
            pointwise_features), axis=2) # {N, 1, G}
    pairwise_scores = tf.reduce_sum(
        tf.multiply(
            pairwise_weights,
            pairwise_features), axis=[2, 3]) # {N, 1, G}
    features = tf.add(pointwise_scores, pairwise_scores) # {N, 1, G}

    # ALSO, take union of motifs and only keep those in motif mat
    # at this stage, append to config and keep separate
    output_maps = {}
    if config.get("pos_x_pwm") is not None:
        position_map = config["pos_x_pwm"] # {N, 1, pos, M}
        task_idx = config["taskidx"]

        motifs_present_by_grammar = tf.unstack(
            tf.squeeze(grammar_motif_presence, axis=0), axis=1)
        
        for grammar_idx in xrange(len(grammars)):
            key = "grammaridx-{}.motif_x_pos.taskidx-{}".format(grammar_idx, task_idx)
            keep_motifs = tf.greater(motifs_present_by_grammar[grammar_idx], [0])
            motif_indices = tf.reshape(tf.where(keep_motifs), [-1])
            # TODO fix this?
            #position_map_filt = tf.reshape(
            #    tf.gather(position_map, motif_indices, axis=3),
            #    position_map.get_shape().as_list()[0:3] + [motif_counts_by_grammar[grammar_idx]])
            #output_maps[key] = position_map_filt

    return features, labels, output_maps


def multitask_score_grammars(features, labels, config, is_training=False):
    """Run multitask version
    """
    grammar_sets = config.get("grammars")
    assert grammar_sets is not None
    
    # split features by task
    #features = [tf.expand_dims(tensor, axis=1)
    #            for tensor in tf.unstack(features, axis=1)] # {N, 1, M}
    features = [features for i in xrange(10)] # {N, 10, M}
    
    if config.get("keep_pwm_scores_full") is not None:
        position_maps = [tf.expand_dims(tensor, axis=1)
                         for tensor in tf.unstack(
                                 config["outputs"][config["keep_pwm_scores_full"]], axis=1)]
        position_maps = [
            tf.expand_dims(config["outputs"][config["keep_pwm_scores_full"]], axis=1)
            for i in xrange(10)]
        
    else:
        position_maps = None
    
    # score grammars by task
    grammar_scores = []
    for i in xrange(len(features)):
        # get the right grammars
        task_grammars = []
        for grammar_set in grammar_sets:
            task_grammars.append(grammar_set[i])
        # set up config
        task_config = {
            "pwms": config["pwms"],
            "grammars": task_grammars,
            "pos_x_pwm": position_maps[i],
            "taskidx": i}
        # score
        scores, _, output_maps = score_grammars(
            features[i], labels, task_config, is_training=is_training)
        grammar_scores.append(scores)
        config["outputs"].update(output_maps)

    # stack
    features = tf.concat(grammar_scores, axis=1)

    # delete output
    if config.get("keep_pwm_scores_full") is not None:
        pass
        #del config["outputs"][config["keep_pwm_scores_full"]]
    
    return features, labels, config


def score_grammars_old(features, labels, config, is_training=False):
    """load in grammars
    """
    grammars = config.get("grammars")
    pwms = config.get("pwms")
    assert grammars is not None
    assert pwms is not None
    
    # input - {N, task, M}, ie assume before reduction to global and position
    # adjust this in inference stack

    grammar_array = np.zeros((len(pwms), len(grammars)))
    for grammar_idx in xrange(len(grammars)):
        grammar_array[:,grammar_idx] = grammars[grammar_idx].pwm_vector

    grammar_tensor = tf.convert_to_tensor(grammar_array, dtype=tf.float32) # {M, G}
    grammar_presence_threshold = tf.reduce_sum(
        tf.cast(
            tf.greater(grammar_tensor, [0]), # eventually, look at negative scores too
            tf.float32), axis=0, keep_dims=True) # {1, G}

    # adjust grammar threshold?
    #grammar_threshold = tf.multiply(grammar_threshold, [0.75]) # at least 75% of hits
    
    # adjust score tensor dimensions
    grammar_tensor = tf.expand_dims(grammar_tensor, axis=0) # {1, M, G}
    grammar_tensor = tf.expand_dims(grammar_tensor, axis=0) # {1, 1, M, G}

    # adjust feature tensor dimensions
    features = tf.expand_dims(features, axis=3) # {N, task, M, 1}

    # multiply
    grammar_scores = tf.multiply(grammar_tensor, features) # {N, task, M, G} 
    features = tf.reduce_sum(grammar_scores, axis=2) # {N, task, G}

    # keep for filtering
    keep_key = config.get("keep_grammar_scores_full")
    if keep_key is not None:
        config["outputs"][keep_key] = grammar_scores # {N, task, M, G}

        # also whether it passed the threshold
        motif_count_scores = tf.reduce_sum(
            tf.cast(
                tf.greater(grammar_scores, [0]),
                tf.float32), axis=2) # {N, task, G}

        max_motif_scores = tf.reduce_max(motif_count_scores, axis=1) # {N, G}
        
        # was a grammar present
        grammars_present = tf.reduce_max(
            tf.cast(tf.greater_equal(
                max_motif_scores,
                grammar_presence_threshold), tf.float32),
            axis=1, keep_dims=True) # {N}
        config["outputs"]["grammars_present"] = grammars_present

    return features, labels, config
