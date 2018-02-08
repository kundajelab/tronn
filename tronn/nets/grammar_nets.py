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
    grammar_threshold = tf.reduce_sum(
        tf.cast(
            tf.greater(grammar_tensor, [0]), # eventually, look at negative scores too
            tf.float32), axis=0) # {G}

    # adjust grammar threshold?
    grammar_threshold = tf.multiply(grammar_threshold, [0.75]) # at least 75% of hits
    
    # adjust score tensor dimensions
    grammar_tensor = tf.expand_dims(grammar_tensor, axis=0) # {1, M, G}
    grammar_tensor = tf.expand_dims(grammar_tensor, axis=0) # {1, 1, M, G}

    # adjust feature tensor dimensions
    features = tf.expand_dims(features, axis=3) # {N, task, M, 1}

    # multiply
    grammar_scores = tf.reduce_sum(
        tf.cast(
            tf.greater(
                tf.multiply(grammar_tensor, features), [0]),
            tf.float32), axis=2) # {N, task, G}

    # may need to adjust dims here too
    features = tf.cast(tf.greater_equal(grammar_scores, grammar_threshold), tf.float32) # {N, task, G}

    # outside of this, condense features to global score
    
    return features, labels, config
