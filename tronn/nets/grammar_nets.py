"""Contains nets to help run grammars
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_utils import get_fan_in
from tronn.util.initializers import pwm_simple_initializer


def score_grammars(features, labels, config, is_training=False):
    """load in grammar
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
