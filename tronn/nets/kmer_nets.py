# Description: use graphs to quickly scan for kmers and sum up


import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.initializers import pwm_simple_initializer
from tronn.util.tf_utils import get_fan_in
from tronn.interpretation.kmers import kmer_hash_to_array
from tronn.interpretation.motifs import PWM

def gkmerize(features, labels, model_params, is_training=False):
    """Given a (batch, 1, seq_len, 4) sequence, kmerize it
    Allow gapped kmers
    
    INCOMPLETE: stopped because I realized that it's faster
    to parallely process kmers on CPUs instead of 1 GPU... ha

    """
    kmer_len = model_params["kmer_len"]
    base_pairs = 5
    # set up kmers to scan
    kmer_pwm_list = []
    kmer_match_totals = []
    total_kmers = base_pairs**kmer_len
    for kmer_hash in xrange(total_kmers):
        kmer_array = kmer_hash_to_array(kmer_hash)
        kmer_pwm_list.append(PWM(kmer_array))
        kmer_match_totals.append(np.sum(kmer_array))

    kmer_match_tensor = tf.cast(tf.convert_to_tensor(np.array(kmer_match_totals)), tf.float32)
        
    # make the conv filter layer
    kmer_filter_size = [1, kmer_len]
    with slim.arg_scope(
            [slim.conv2d],
            padding="VALID",
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                kmer_filter_size, kmer_pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        net = slim.conv2d(
            features, total_kmers, kmer_filter_size,
            scope="conv1/conv")

    # then need to only keep those that fully match
    kmer_matches = tf.cast(tf.equal(net, kmer_match_tensor), tf.int32)
    kmer_vector = tf.squeeze(tf.reduce_sum(kmer_matches, axis=2))

    return kmer_vector



def featurize_kmers(features, kmer_len=6, is_training=False):
    """Given a (batch, 1, seq_len, 4) sequence, kmerize it
    Allow gapped kmers

    Use this to featurize on the fly
    """
    base_pairs = 5
    # set up kmers to scan
    kmer_pwm_list = []
    kmer_match_totals = []
    total_kmers = base_pairs**kmer_len
    for kmer_hash in xrange(total_kmers):
        kmer_array = kmer_hash_to_array(kmer_hash)
        kmer_pwm_list.append(PWM(kmer_array))
        kmer_match_totals.append(np.sum(kmer_array))

    kmer_match_tensor = tf.cast(tf.convert_to_tensor(np.array(kmer_match_totals)), tf.float32)
        
    # make the conv filter layer
    kmer_filter_size = [1, kmer_len]
    with slim.arg_scope(
            [slim.conv2d],
            padding="VALID",
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                kmer_filter_size, kmer_pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        net = slim.conv2d(
            features, total_kmers, kmer_filter_size,
            scope="conv1/conv")

    # then need to only keep those that fully match
    kmer_matches = tf.cast(tf.equal(net, kmer_match_tensor), tf.int32)
    kmer_vector = tf.squeeze(tf.reduce_sum(kmer_matches, axis=2))

    return kmer_vector


def tensor_to_hash(kmer_tensor, base5_converter):
    """Take in an array and convert to hash value
    """
    indices = tf.argmax(kmer_tensor, axis=3)

    


    return


def featurize_kmers_v2(features, kmer_len=6, is_training=False):
    """Given a (batch, 1, seq_len, 4) sequence, kmerize it
    Allow gapped kmers

    Use this to featurize on the fly
    """
    base_pairs = 5
    base5_converter = tf.constant([5**i for i in xrange(kmer_len)])

    # set up a sparse tensor representation
    seq_len = features.get_shape()[2]

    indices = []
    for pos_idx in xrange(seq_len):
        # here, extract kmer, mutate, and add to list of indices
        kmer_array = features[:, 1, pos_idx:pos_idx+kmer_len, :]

        
        pass

    # then stack the indices to a tensor

    # instantiate a tensor of 1s to be values


    # and then also set up dense tensor


    # and merge into sparse tensor


    # then convert to dense tensor
    
    


    
    # set up kmers to scan
    kmer_pwm_list = []
    kmer_match_totals = []
    total_kmers = base_pairs**kmer_len
    for kmer_hash in xrange(total_kmers):
        kmer_array = kmer_hash_to_array(kmer_hash)
        kmer_pwm_list.append(PWM(kmer_array))
        kmer_match_totals.append(np.sum(kmer_array))

    kmer_match_tensor = tf.cast(tf.convert_to_tensor(np.array(kmer_match_totals)), tf.float32)
        
    # make the conv filter layer
    kmer_filter_size = [1, kmer_len]
    with slim.arg_scope(
            [slim.conv2d],
            padding="VALID",
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                kmer_filter_size, kmer_pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        net = slim.conv2d(
            features, total_kmers, kmer_filter_size,
            scope="conv1/conv")

    # then need to only keep those that fully match
    kmer_matches = tf.cast(tf.equal(net, kmer_match_tensor), tf.int32)
    kmer_vector = tf.squeeze(tf.reduce_sum(kmer_matches, axis=2))

    return kmer_vector
