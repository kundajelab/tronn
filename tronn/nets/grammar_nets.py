"""Contains nets to help run grammars
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_utils import get_fan_in
from tronn.util.initializers import pwm_simple_initializer


# write a single task version, then compile into multitask
def score_grammars(features, labels, config, is_training=False):
    """load in grammar
    """
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

    # only give it a score if it had presence of all key motifs
    grammar_motif_presence = tf.cast(tf.greater(pointwise_weights, [0]), tf.float32) #  {1, 1, M, G}
    grammar_motif_count = tf.reduce_sum(grammar_motif_presence, axis=2) # {1, 1, G}
    pointwise_presence = tf.cast(tf.greater(
        tf.multiply(
            grammar_motif_presence,
            pointwise_features), [0]), tf.float32) # {N, 1, M, G}
    motif_present_counts = tf.reduce_sum(pointwise_presence, axis=2) #  {N, 1, G}
    score_mask = tf.cast(tf.equal(motif_present_counts, grammar_motif_count), tf.float32) # {N, 1, G}
    
    # pass through mask
    features = tf.multiply(score_mask, features)

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
    features = [tf.expand_dims(tensor, axis=1)
                for tensor in tf.unstack(features, axis=1)] # {N, 1, M}
    if config.get("keep_pwm_scores_full") is not None:
        position_maps = [tf.expand_dims(tensor, axis=1)
                         for tensor in tf.unstack(
                                 config["outputs"][config["keep_pwm_scores_full"]], axis=1)]
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
