"""Contains nets that perform PWM convolutions
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.initializers import pwm_simple_initializer
from tronn.util.tf_utils import get_fan_in


def pwm_convolve_v2(features, labels, model_params, is_training=False):
    '''
    All this model does is convolve with PWMs and get top k pooling to output
    a example by motif matrix.
    '''

    pwm_list = model_params["pwms"]

    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)

    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]

    # make the convolution net
    conv1_filter_size = [1, max_size]
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        net = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='conv1/conv')

    # Then get top k values across the correct axis
    net = tf.transpose(net, perm=[0, 1, 3, 2])
    top_k_val, top_k_indices = tf.nn.top_k(net, k=3)

    # Do a summation
    motif_tensor = tf.squeeze(tf.reduce_sum(top_k_val, 3)) # 3 is the axis

    return labels, motif_tensor, motif_tensor



def featurize_motifs(features, pwm_list=None, is_training=False):
    '''
    All this model does is convolve with PWMs and get top k pooling to output
    a example by motif matrix.
    '''
    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)

    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]

    # make the convolution net
    conv1_filter_size = [1, max_size]
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size, pwm_list, get_fan_in(features)),
            biases_initializer=None,
            trainable=False):
        net = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='conv1/conv')

    # Then get top k values across the correct axis
    net = tf.transpose(net, perm=[0, 1, 3, 2])
    top_k_val, top_k_indices = tf.nn.top_k(net, k=3)

    # Do a summation
    motif_tensor = tf.squeeze(tf.reduce_sum(top_k_val, 3)) # 3 is the axis

    return motif_tensor


def pwm_convolve(features, labels, pwm_list):
    '''
    All this model does is convolve with PWMs and get top k pooling to output
    a example by motif matrix.
    '''

    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)

    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]

    # make the convolution net
    with slim.arg_scope([slim.conv2d], padding='VALID',
                        activation_fn=None, trainable=False):
        conv1_filter_size = [1, max_size]
        net = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='conv1/conv')

    # Then get top k values across the correct axis
    net = tf.transpose(net, perm=[0, 1, 3, 2])
    top_k_val, top_k_indices = tf.nn.top_k(net, k=3)

    # Do a summation
    motif_tensor = tf.squeeze(tf.reduce_sum(top_k_val, 3)) # 3 is the axis

    # Then adjust the filters by putting in PWM info
    # note that there should actually only be 1 set of weights, the first layer
    weights = [v for v in tf.global_variables() if ('weights' in v.name)] 
    weights_list = []
    for i in range(len(pwm_list)):
        pwm = pwm_list[i]
        pad_length = max_size - pwm.weights.shape[1]
        padded_weights = np.concatenate((pwm.weights,
                                         np.zeros((4, pad_length))),
                                        axis=1)
        weights_list.append(padded_weights)

    # stack into weights tensor and assign
    pwm_all_weights = np.stack(weights_list, axis=0).transpose(2, 1, 0)
    pwm_np_tensor = np.expand_dims(pwm_all_weights, axis=0)
    load_pwm_update = weights[0].assign(pwm_np_tensor)

    return motif_tensor, load_pwm_update


def top_motifs_w_distances(features, pwm_list, top_k_val=2):
    '''
    This extracts motif scores with associated distances
    '''

    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)

    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]

    # make the convolution net
    with slim.arg_scope([slim.conv2d], padding='VALID',
                        activation_fn=None, trainable=False):
        conv1_filter_size = [1, max_size]
        net = slim.conv2d(
            features, num_filters, conv1_filter_size,
            scope='conv1/conv')

    # Then get top k values across the correct axis
    net = tf.squeeze(tf.transpose(net, perm=[0, 1, 3, 2]))
    mat_topkval, mat_topkval_indices = tf.nn.top_k(net, k=top_k_val)
    
    # Get mean and var to do a zscore on the scores for each sequence
    seq_mean, seq_var = tf.nn.moments(mat_topkval, [1, 2])

    # Extra operations because broadcasting is finicky
    mat_mean_intermediate = tf.stack([seq_mean for i in range(num_filters)], axis=1)
    mat_mean = tf.stack([mat_mean_intermediate for i in range(top_k_val)], axis=2)

    mat_var_intermediate = tf.stack([seq_var for i in range(num_filters)], axis=1)
    mat_var = tf.stack([mat_var_intermediate for i in range(top_k_val)], axis=2)

    mat_topkval_zscore = tf.multiply(tf.subtract(mat_topkval, mat_mean), tf.rsqrt(mat_var))
    
    # TOP SCORES: reshape, outer product, reshape (NOTE: broadcasting not fully working in this tf version)
    mat1_topkval_motif_x_motif = tf.stack([mat_topkval_zscore for i in range(num_filters)], axis=2)
    mat1_topkval_full = tf.stack([mat1_topkval_motif_x_motif for i in range(top_k_val)], axis=4)
    mat2_topkval_motif_x_motif = tf.stack([mat_topkval_zscore for i in range(num_filters)], axis=1)
    mat2_topkval_full = tf.stack([mat2_topkval_motif_x_motif for i in range(top_k_val)], axis=3)

    motif_x_motif_scores = tf.multiply(mat1_topkval_full, mat2_topkval_full)
    score_dims = motif_x_motif_scores.get_shape().as_list()
    new_dims = score_dims[:-2] + [score_dims[-2] * score_dims[-1]]
    motif_x_motif_scores_redux = tf.reshape(motif_x_motif_scores, new_dims)
    print "Motif score matrix dims:", motif_x_motif_scores_redux.get_shape()

    # TOP INDICES: reshape, outer product, reshape
    mat1_topkval_idx_x_idx = tf.stack([mat_topkval_indices for i in range(num_filters)], axis=2)
    mat1_topkval_indices_full = tf.stack([mat1_topkval_idx_x_idx for i in range(top_k_val)], axis=4)
    mat2_topkval_idx_x_idx = tf.stack([mat_topkval_indices for i in range(num_filters)], axis=1)
    mat2_topkval_indices_full = tf.stack([mat2_topkval_idx_x_idx for i in range(top_k_val)], axis=3)

    motif_x_motif_indices = tf.abs(tf.subtract(mat1_topkval_indices_full, mat2_topkval_indices_full))
    motif_x_motif_indices_redux = tf.reshape(motif_x_motif_indices, new_dims)
    print "Motif indices matrix dims:", motif_x_motif_indices_redux.get_shape()

    # --------------------
    # Loading PWMs into the first layer convolutions
    
    # Then adjust the filters by putting in PWM info
    # note that there should actually only be 1 set of weights, the first layer
    weights = [v for v in tf.global_variables() if ('weights' in v.name)] 
    weights_list = []
    for i in range(len(pwm_list)):
        pwm = pwm_list[i]
        pad_length = max_size - pwm.weights.shape[1]
        padded_weights = np.concatenate((pwm.weights,
                                         np.zeros((4, pad_length))),
                                        axis=1)
        weights_list.append(padded_weights)

    # stack into weights tensor and assign
    pwm_all_weights = np.stack(weights_list, axis=0).transpose(2, 1, 0)
    pwm_np_tensor = np.expand_dims(pwm_all_weights, axis=0)
    load_pwm_update = weights[0].assign(pwm_np_tensor)

    return motif_x_motif_scores_redux, motif_x_motif_indices_redux, load_pwm_update
