"""Description: Contains tensorflow graphs

Currently implemented:

- Basset (Kelley et al Genome Research 2016)
- RNN module
- Resnet module

"""

import math
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

from tensorflow.contrib.distributions import Poisson

from tronn.util.tf_ops import maxnorm
from tronn.util.sample_stats import percentile

from tronn.initializers import pwm_initializer
from tronn.initializers import pwm_simple_initializer

from tronn.util.tf_utils import get_fan_in



def final_pool(net, pool):
    if pool == 'flatten':
        net = slim.flatten(net, scope='flatten')
    elif pool == 'mean':
        net = tf.reduce_mean(net, axis=[1,2], name='avg_pooling')
    elif pool == 'max':
        net = tf.reduce_max(net, axis=[1,2], name='max_pooling')
    elif pool == 'kmax':
        net = tf.squeeze(net, axis=1)#remove width that was used for conv2d; result is batch x time x dim
        net_time_last = tf.transpose(net, perm=[0,2,1])
        net_time_last = nn_ops.order_preserving_k_max(net_time_last, k=8)
        net = slim.flatten(net_time_last, scope='flatten')
    elif pool is not None:
        raise Exception('Unrecognized final_pooling: %s'% pool)
    return net


def mlp_module(features, num_tasks, fc_dim, fc_layers, dropout=0.0, l2=0.0, is_training=True):
    net = features
    with slim.arg_scope([slim.fully_connected], activation_fn=None, weights_regularizer=slim.l2_regularizer(l2)):
        for i in xrange(fc_layers):
            with tf.variable_scope('fc%d'%i):
                net = slim.fully_connected(net, fc_dim, biases_initializer=None)
                net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training)
                net = slim.dropout(net, keep_prob=1.0-dropout, is_training=is_training)
        logits = slim.fully_connected(net, num_tasks, scope='logits')
    return logits


def temporal_pred_module(features, num_days, share_logistic_weights, is_training=True):
    dim = features.shape.as_list()[1]
    day_nets = [slim.fully_connected(features, dim, activation_fn=tf.nn.relu) for day in xrange(num_days)]#remove relu?
    cell_fw = tf.contrib.rnn.LSTMBlockCell(dim)
    cell_bw = tf.contrib.rnn.LSTMBlockCell(dim)
    day_nets, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, day_nets, dtype=tf.float32)
    if share_logistic_weights:
        net = tf.concat([tf.expand_dims(day_nets, 1) for day_net in day_nets], 1)#batch X day X 2*dim
        net = tf.reshape(net, [-1, 2*dim])#batch*day X 2*dim
        logits_flat = slim.fully_connected(net, 1, activation_fn=None)#batch*day X 1
        logits = tf.reshape(logits_flat, [-1, num_days])#batch X num_days
    else:
        day_logits = [slim.fully_connected(day_net, 1, activation_fn=None) for day_net in day_nets]
        logits = tf.concat(day_logits, 1)
    return logits


def basset_conv_module(features, is_training=True):
    with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
        with slim.arg_scope([slim.conv2d], activation_fn=None, weights_initializer=layers.variance_scaling_initializer(), biases_initializer=None):
            net = slim.conv2d(features, 300, [1, 19])
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [1, 3], stride=[1, 3])

            net = slim.conv2d(net, 200, [1, 11])
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [1, 4], stride=[1, 4])

            net = slim.conv2d(net, 200, [1, 7])
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [1, 4], stride=[1, 4])
    return net


def basset(features, labels, config, is_training=True):
    '''
    Basset - Kelley et al Genome Research 2016
    '''
    config['temporal'] = config.get('temporal', False)
    config['final_pool'] = config.get('final_pool', 'flatten')
    config['fc_layers'] = config.get('fc_layers', 2)
    config['fc_dim'] = config.get('fc_dim', 1000)
    config['drop'] = config.get('drop', 0.3)

    net = basset_conv_module(features, is_training)
    net = final_pool(net, config['final_pool'])
    if config['temporal']:
        logits = temporal_pred_module(net, int(labels.get_shape()[-1]), share_logistic_weights=True, is_training=is_training)
    else:
        logits = mlp_module(net, 
                    num_tasks = int(labels.get_shape()[-1]), 
                    fc_dim = config['fc_dim'], 
                    fc_layers = config['fc_layers'],
                    dropout=config['drop'],
                    is_training=is_training)
    # Torch7 style maxnorm
    maxnorm(norm_val=7)

    return logits


def danq(features, labels, config, is_training=True):
    net = slim.conv2d(features, 320, kernel_size=[1,26], stride=[1,1], activation_fn=tf.nn.relu, padding='VALID')
    net = slim.max_pool2d(net, kernel_size=[1,13], stride=[1,13], padding='VALID')
    net = slim.dropout(net, keep_prob=0.8, is_training=is_training)

    net = tf.squeeze(net, axis=1)#remove extra dim that was added so we could use conv2d. Results in batchXtimeXdepth
    rnn_inputs = tf.unstack(net, axis=1, name='unpack_time_dim')

    cell_fw = tf.contrib.rnn.LSTMBlockCell(320)
    cell_bw = tf.contrib.rnn.LSTMBlockCell(320)
    outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
    net = tf.concat([state_fw[1], state_bw[1]], axis=1)
    net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
    net = slim.fully_connected(net, 925, activation_fn=tf.nn.relu)
    logits = slim.fully_connected(net, int(labels.get_shape()[-1]), activation_fn=None)
    return logits


def _residual_block(net_in, depth, pooling_info=(None, None), first=False):
    first_stride = 1
    depth_in = net_in.get_shape().as_list()[-1]

    if depth_in!=depth and not first:
        pooling, pooling_stride = pooling_info
        if pooling=='conv':
            shortcut = slim.avg_pool2d(net_in, stride=[1, pooling_stride])#downsample for both shortcut and conv branch
            net_preact = slim.batch_norm(net_in)
            first_stride = pooling_stride
        elif pooling=='max':
            net = slim.max_pool2d(net_in, stride=[1, pooling_stride])#downsample for both shortcut and conv branch
            shortcut = net
            net_preact = slim.batch_norm(net)
        else:
            raise Exception('unrecognized pooling: %s'%pooling_info)
    else:
        net_preact = slim.batch_norm(net_in)
        if first:
            shortcut = net_preact
        else:
            shortcut = net_in
    net = slim.conv2d(net_preact, depth, stride=[1, first_stride])
    net = slim.batch_norm(net)
    net = slim.conv2d(net, depth, stride=[1, 1])

    if depth_in != depth:
        paddings = [(0,0),(0,0),(0,0),((depth-depth_in)/2, int(math.ceil((depth-depth_in)/2)))]
        shortcut = tf.pad(net_preact, paddings)
    net = net + shortcut
    return net


def _resnet(features, initial_conv, kernel, stages, pooling_info, l2, is_training=True):
    print features.get_shape().as_list()
    with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], kernel_size=[1, kernel], padding='SAME'):
            with slim.arg_scope(
                    [slim.conv2d],
                    activation_fn=None,
                    weights_regularizer=slim.l2_regularizer(l2),
                    weights_initializer=layers.variance_scaling_initializer(),
                    biases_initializer=None):
                # We do not include batch normalization or activation functions in embed because the first ResNet unit will perform these.
                with tf.variable_scope('embed'):
                    initial_filters, initial_kernel, initial_stride = initial_conv
                    net = slim.conv2d(features, initial_filters, kernel_size=[1, initial_kernel])
                    net = slim.max_pool2d(net, kernel_size=[1, initial_stride], stride=[1, initial_stride])
                print net.get_shape().as_list()
                for i, stage in enumerate(stages):
                    with tf.variable_scope('stage%d'%i):
                        num_blocks, depth = stage
                        for j in xrange(num_blocks):
                            with tf.variable_scope('block%d'%j):
                                net = _residual_block(net, depth, pooling_info, first=(i==0 and j==0))
                                print net.get_shape().as_list()
        net = slim.batch_norm(net)
    return net


def resnet(features, labels, config, is_training=True):
    initial_conv = config.get('initial_conv', (16, 3, 1))
    kernel = config.get('kernel', 3)
    stages = config.get('stages', [(1, 32),(1, 64),(1, 128),(1, 256)])
    pooling_info = config.get('pooling', ('max', 2))
    final_pooling = config.get('final_pooling', 'mean')
    fc_layers, fc_dim = config.get('fc', (1, 1024))
    drop = config.get('drop', 0.0)
    l2 = config.get('l2', 0.0001)
    num_labels = int(labels.get_shape()[-1])

    net = _resnet(features, initial_conv, kernel, stages, pooling_info, l2, is_training)
    net = final_pool(net, final_pooling)
    logits = mlp_module(net, num_labels, fc_dim, fc_layers, drop, l2, is_training)
    return logits


models = {}
models['basset'] = basset
models['danq'] = danq
models['resnet'] = resnet


# ================================================
# MODELS USED IN INTERPRETATION BELOW
# ================================================

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



def old_basset(features, labels, is_training=True):
    '''
    Basset - Kelley et al Genome Research 2016
    '''

    with slim.arg_scope([slim.conv2d], padding='VALID',
                        activation_fn=None):

        # Layer 1: conv layer to batch norm to relu to max pool.
        conv1_filter_size = [1, 19]
        net = slim.conv2d(
            features, 300, conv1_filter_size,
            weights_initializer=initializers.torch_conv_initializer(
                conv1_filter_size, tf_utils.get_fan_in(features)),
            biases_initializer=initializers.torch_conv_initializer(
                conv1_filter_size, tf_utils.get_fan_in(features)),
            scope='conv1/conv')
        net = slim.batch_norm(net, center=True, scale=True,
                              activation_fn=nn.relu, 
                              is_training=is_training,
                              scope='conv1/batchnorm')
        net = slim.max_pool2d(net, [1, 3], stride=[1, 3], 
            scope='conv1/maxpool')

        # Layer 2: conv layer to batch norm to relu to max pool.
        conv2_filter_size = [1, 11]
        net = slim.conv2d(
            net, 200, conv2_filter_size,
            weights_initializer=initializers.torch_conv_initializer(
                conv2_filter_size, tf_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_conv_initializer(
                conv2_filter_size, tf_utils.get_fan_in(net)),
            scope='conv2/conv')
        net = slim.batch_norm(net, center=True, scale=True, 
                              activation_fn=nn.relu, 
                              is_training=is_training,
                              scope='conv2/batchnorm')
        net = slim.max_pool2d(net, [1, 4], stride=[1, 4], 
            scope='conv2/maxpool')

        # Layer 3: conv layer to batch norm to relu to max pool.
        conv3_filter_size = [1, 7]
        net = slim.conv2d(
            net, 200, conv3_filter_size,
            weights_initializer=initializers.torch_conv_initializer(
                conv3_filter_size, tf_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_conv_initializer(
                conv3_filter_size, tf_utils.get_fan_in(net)),
            scope='conv3/conv')
        net = slim.batch_norm(net, center=True, scale=True,
                              activation_fn=nn.relu, 
                              is_training=is_training,
                              scope='conv3/batchnorm')
        net = slim.max_pool2d(net, [1, 4], stride=[1, 4], 
            scope='conv3/maxpool')

    net = slim.flatten(net, scope='flatten')

    with slim.arg_scope([slim.fully_connected], activation_fn=None):

        # Layer 4: fully connected layer to relu to dropout
        net = slim.fully_connected(
            net, 1000, 
            weights_initializer=initializers.torch_fullyconnected_initializer(
                tf_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_fullyconnected_initializer(
                tf_utils.get_fan_in(net)),
            scope='fullyconnected1/fullyconnected')
        net = slim.batch_norm(net, center=True, scale=True,
                              activation_fn=nn.relu, 
                              is_training=is_training,
                              scope='fullyconnected1/batchnorm')
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training,
            scope='fullyconnected1/dropout')

        # Layer 5: fully connected layer to relu to dropout
        net = slim.fully_connected(
            net, 1000, 
            weights_initializer=initializers.torch_fullyconnected_initializer(
                tf_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_fullyconnected_initializer(
                tf_utils.get_fan_in(net)),
            scope='fullyconnected2/fullyconnected')
        net = slim.batch_norm(net, center=True, scale=True,
                              activation_fn=nn.relu, 
                              is_training=is_training,
                              scope='fullyconnected2/batchnorm')
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training,
            scope='fullyconnected2/dropout')

        # OUT
        logits = slim.fully_connected(
            net, int(labels.get_shape()[-1]), activation_fn=None,
            weights_initializer=initializers.torch_fullyconnected_initializer(
                tf_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_fullyconnected_initializer(
                tf_utils.get_fan_in(net)),
            scope='out')

    # Torch7 style maxnorm
    nn_ops.maxnorm(norm_val=7)

    return logits



# ==================================
# weighted kmers
# ==================================

def pois_cutoff(signal, pval):
    """Given importance scores, calculates poisson pval
    and thresholds at that pval

    Args:
      signal: input tensor
      pval: the pval threshold

    Returns:
      out_tensor: output thresholded tensor
    """
    percentile_val = 1. - pval
    signal_shape = signal.get_shape()
    print "feature shape", signal_shape
    
    # get mean
    signal_mean = tf.reduce_mean(signal, axis=[1, 2, 3])
    print "signal mean", signal_mean.get_shape()

    # calculate poisson
    pois_distributions = tf.random_poisson(signal_mean, [1000])
    thresholds = percentile(pois_distributions, percentile_val, axis=0)

    for dim_idx in range(1, len(signal_shape)):
        to_stack = [thresholds for i in range(signal_shape[dim_idx])] # here?
        thresholds = tf.stack(to_stack, dim_idx)

    print "thresh full", thresholds.get_shape()
        
    # and threshold
    greaterthan_tensor = tf.cast(tf.greater(signal, thresholds), tf.float32)
    thresholded_tensor = signal * greaterthan_tensor

    out_tensor = tf.transpose(tf.squeeze(thresholded_tensor), [0, 2, 1])

    return out_tensor, signal_mean


def stdev_cutoff(signal, num_stdev=3):
    """Given importance scores, calculates poisson pval
    and thresholds at that pval

    Args:
      signal: input tensor
      pval: the pval threshold

    Returns:
      out_tensor: output thresholded tensor
    """
    #percentile_val = 1. - pval
    signal_shape = signal.get_shape()
    print "feature shape", signal_shape
    
    # get mean and stdev
    signal_mean, signal_var = tf.nn.moments(signal, axes=[1, 2, 3])
    signal_stdev = tf.sqrt(signal_var)
    thresholds = tf.add(signal_mean, tf.scalar_mul(num_stdev, signal_stdev))

    for dim_idx in range(1, len(signal_shape)):
        to_stack = [thresholds for i in range(signal_shape[dim_idx])] # here?
        thresholds = tf.stack(to_stack, dim_idx)

    print "thresh full", thresholds.get_shape()
        
    # and threshold
    greaterthan_tensor = tf.cast(tf.greater(signal, thresholds), tf.float32)
    thresholded_tensor = signal * greaterthan_tensor

    out_tensor = tf.transpose(tf.squeeze(thresholded_tensor), [0, 2, 1])

    return out_tensor


# ===============================
# grammars
# ===============================



def _grammar_module(features, grammar, threshold=False, nonlinear=False):
    """Set up grammar part of graph
    This is separated out in case we do more complicated grammar analyses 
    (such as involving distances, make decision trees, etc)
    """

    # bad: clean up later
    from tronn.interpretation.motifs import PWM
    
    # get various sizes needed to instantiate motif matrix
    num_filters = len(grammar.pwms)

    max_size = 0
    for pwm in grammar.pwms:
        # TODO write out order so that we have it
        
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]


    # run through motifs and select max score of each
    conv1_filter_size = [1, max_size]
    with slim.arg_scope([slim.conv2d],
                        padding='VALID',
                        activation_fn=None,
                        weights_initializer=pwm_initializer(conv1_filter_size, grammar.pwms, get_fan_in(features), num_filters),
                        biases_initializer=None):
        net = slim.conv2d(features, num_filters, conv1_filter_size)

        width = net.get_shape()[2]
        net = slim.max_pool2d(net, [1, width], stride=[1, 1])

        # and squeeze it
        net = tf.squeeze(net)

        if len(net.get_shape()) == 1:
            return net

        # and then just do a summed score for now? averaged score?
        # TODO there needs to be some sort of cooperative nonlinearity?
        # consider a joint multiplication of everything
        if nonlinear:
            net = tf.reduce_prod(net, axis=1)
        else:
            net = tf.reduce_mean(net, axis=1)
        
        # TODO throw in optional threshold
        if threshold:
            print "not yet implemented!"

    return net


def grammar_scanner(features, grammars, normalize=True):
    """Sets up grammars to run

    Args:
      features: input sequence (one hot encoded)
      grammars: dictionary, grammar name to grammar object

    Returns:
      out_tensor: vector of output values after running grammars
    """

    if normalize:
        features_norm = (features - 0.25) / 0.4330127
    else:
        features_norm = features
    
    grammar_out = []
    for grammar_key in sorted(grammars.keys()):
        grammar_out.append(_grammar_module(features_norm, grammars[grammar_key]))
        
    # then pack it into a vector and return vector
    out_tensor = tf.stack(grammar_out, axis=1)

    return out_tensor


def _mutate(feature_tensor, position_tensor, zero_out_ref=False):
    """Small module to perform 3 mutations from reference base pair
    Input goes from
    """
    mutated_examples = []
    # build the index position
    positions_tiled = tf.reshape(tf.stack([position_tensor for i in range(4)]), (4, 1))
    indices_list = [tf.cast(tf.zeros((4, 1)), 'int64'),
                    tf.cast(tf.zeros((4, 1)), 'int64'),
                    tf.reshape(tf.cast(positions_tiled, 'int64'), (4, 1)),
                    tf.reshape(tf.cast(tf.range(4), 'int64'), (4, 1))]
    indices_tensor = tf.squeeze(tf.stack(indices_list, axis=1))
    
    # go through all 4 positions, and ignore (zero out) the one that is reference
    for mut_idx in range(4):
        updates_array = np.ones((4))
        updates_array[mut_idx] = 0
        updates_tensor = tf.constant(updates_array)
        mutated_mask_inv = tf.cast(
            tf.scatter_nd(indices_tensor, updates_tensor, feature_tensor.get_shape()),
            'bool')
        mutated_mask = tf.cast(
            tf.logical_not(mutated_mask_inv),
            'float32')
        mutated_example = tf.squeeze(tf.multiply(feature_tensor, mutated_mask), axis=0)
        mutated_examples.append(mutated_example)

    mutated_batch = tf.stack(mutated_examples, axis=0)
    #print mutated_batch.get_shape()
    
    if zero_out_ref:
        # zero out the one that is reference
        
        slice_start_position = tf.stack([tf.constant(0),
                                         tf.constant(0),
                                         tf.cast(position_tensor, 'int32'),
                                         tf.constant(0)])

        #print slice_start_position.get_shape()
        #print feature_tensor.get_shape()
        mut_position_slice = tf.slice(feature_tensor,
                                      slice_start_position,
                                      [1, 1, 1, 4])

        # and flip
        mut_position_slice_bool = tf.cast(mut_position_slice, 'bool')
        mut_position_mask = tf.cast(tf.logical_not(mut_position_slice_bool), 'float32')
    
    
        #print mut_position_slice.get_shape()
        
        mask_per_bp = [tf.expand_dims(tf.squeeze(mut_position_mask), axis=1) for i in range(feature_tensor.get_shape()[2])]
        mask_per_bp_tensor = tf.stack(mask_per_bp, axis=2)
        #print mask_per_bp_tensor.get_shape()
        
        mask_list = [mask_per_bp_tensor for i in range(feature_tensor.get_shape()[3])]
        mask_batch = tf.stack(mask_list, axis=3)
    
        #print mask_batch.get_shape()
        mutated_filtered_examples = tf.multiply(mutated_batch, mask_batch)
        #print mutated_filtered_examples.get_shape()
    else:
        mutated_filtered_examples = mutated_batch
    
        
    return tf.unstack(mutated_filtered_examples)


def generate_point_mutant_batch(features, max_idx, num_mutations):
    """Given a position in a sequence (one hot encoded) generate mutants
    """
    half_num_mutations = num_mutations / 2
    sequence_batch_list = []    
    for mutation_idx in range(-half_num_mutations, half_num_mutations):
        offset_tensor = tf.cast(mutation_idx, 'int64')
        final_position_tensor = tf.add(max_idx, offset_tensor)
        nonnegative_mask = tf.squeeze(tf.greater(final_position_tensor,
                                                 tf.cast(tf.constant([0]), 'int64')))
        final_position_tensor_filt = tf.multiply(final_position_tensor, tf.cast(nonnegative_mask, 'int64'))
        sequence_batch_list = sequence_batch_list + _mutate(features, final_position_tensor_filt)
    # then stack
    return tf.stack(sequence_batch_list, axis=0)


def generate_paired_mutant_batch(features, max_idx1, max_idx2, num_mutations):
    """Given two positions, generate mutants
    """
    single_mutant_batch = generate_point_mutant_batch(features, max_idx1, num_mutations)
    single_mutants = tf.unstack(single_mutant_batch)
    paired_mutant_batch = []
    for single_mutant in single_mutants:
        single_mutant_extended = tf.expand_dims(single_mutant, axis=0)
        paired_mutant_batch = paired_mutant_batch + tf.unstack(
            generate_point_mutant_batch(single_mutant_extended, max_idx2, num_mutations))
    return tf.stack(paired_mutant_batch)
    

def ism_for_grammar_dependencies(
        features,
        labels,
        model_params, 
        num_mutations=6,
        num_tasks=43,
        current_task=0):
    """Run a form of in silico mutagenesis to get dependencies between motifs
    Remember that the input to this should be onehot encoded sequence
    NOT importance scores, and only those for the subtask set you care about
    """
    model = model_params["trained_net"]
    pwm_a = model_params["pwm_a"]
    pwm_b = model_params["pwm_b"]
    
    # first layer - instantiate motifs and scan for best match in sequence
    max_size = max(pwm_a.weights.shape[1], pwm_b.weights.shape[1])
    conv1_filter_size = [1, max_size]
    with slim.arg_scope([slim.conv2d],
                        padding='VALID',
                        activation_fn=None,
                        weights_initializer=pwm_simple_initializer(
                            conv1_filter_size, [pwm_a, pwm_b], get_fan_in(features)),
                        biases_initializer=None,
                        scope='mutate'):
        net = slim.conv2d(features, 2, conv1_filter_size)

    # get max positions for each motif
    max_indices = tf.argmax(net, axis=2) # TODO need to adjust to midpoint
    max_idx1_tensor = tf.squeeze(tf.slice(max_indices, [0, 0, 0], [1, 1, 1]))
    max_idx2_tensor = tf.squeeze(tf.slice(max_indices, [0, 0, 1], [1, 1, 1]))

    # generate mutant sequences for motif 1
    motif1_mutants_batch = generate_point_mutant_batch(
        features, max_idx1_tensor, num_mutations)
    motif1_batch_size = motif1_mutants_batch.get_shape().as_list()[0]
    print "motif 1 total mutants:", motif1_mutants_batch.get_shape()

    # for motif 2: do the same as motif 1
    motif2_mutants_batch = generate_point_mutant_batch(
        features, max_idx2_tensor, num_mutations)
    motif2_batch_size = motif2_mutants_batch.get_shape().as_list()[0]
    print "motif 2 total mutants:", motif2_mutants_batch.get_shape()

    # for joint
    joint_mutants_batch = generate_paired_mutant_batch(
        features, max_idx1_tensor, max_idx2_tensor, num_mutations)
    joint_batch_size = joint_mutants_batch.get_shape().as_list()[0]
    print "joint motif total mutants:", joint_mutants_batch.get_shape()

    # stack original sequence, motif 1 mutations, motif 2 mutations
    all_mutants_batch = tf.stack(
        tf.unstack(features) +
        tf.unstack(motif1_mutants_batch) +
        tf.unstack(motif2_mutants_batch) +
        tf.unstack(joint_mutants_batch)
    )

    # check batch size
    print "full total batch size:", all_mutants_batch.get_shape()
    batch_size = all_mutants_batch.get_shape()[0]

    # use labels to set output size THEN need to select the correct output logit node
    multilabel = tf.stack([labels for i in range(num_tasks)], axis=1)
    labels_list = []
    for i in range(batch_size):
        labels_list = labels_list + tf.unstack(multilabel)
    labels_extended = tf.squeeze(tf.stack(labels_list))

    # pass through model
    logits_alltasks = model(all_mutants_batch,
                        labels_extended,
                        model_config,
                        is_training=False)

    # Now need to select the correct logit position
    logits = tf.slice(logits_alltasks, [0, current_task], [tf.cast(batch_size, 'int32'), 1])
    print logits.get_shape()
    
    # might want to do it on logits actually (see larger synergy score?)
    # like: logit(orig) / (logit(a)/2 + logit (b)/2)
    # this tells you how much the score is versus only having 1 of each
    # actually you'll get two scores back - synergy dependent on one vs other motif
    single_mutant_total = num_mutations*4
    if False:
        example_logits_list = tf.unstack(logits)
        logit_orig = example_logits_list[0]
            
        logits_mutant_motif1 = tf.stack(example_logits_list[1:motif1_batch_size])
        logit_mutant_motif1_min = tf.reduce_min(logits_mutant_motif1, axis=0)
        logits_mutant_motif2 = tf.stack(example_logits_list[(1 + single_mutant_total):])
        logit_mutant_motif2_min = tf.reduce_min(logits_mutant_motif2, axis=0)

        synergy_score = tf.divide(logit_orig, tf.divide(tf.add(logit_mutant_motif1_min, logit_mutant_motif2_min), 2))

    # =============================
    # probabilities
    probabilities = tf.nn.sigmoid(logits)

    example_probs_list = tf.unstack(probabilities)
    prob_orig = example_probs_list[0]
    probs_mutant_motif1 = tf.stack(example_probs_list[1:motif1_batch_size])
    prob_mutant_motif1_min = tf.reduce_min(probs_mutant_motif1, axis=0)
    
    probs_mutant_motif2 = tf.stack(example_probs_list[(1 + motif1_batch_size):(1 + motif1_batch_size + motif2_batch_size)])
    print probs_mutant_motif2.get_shape()
    prob_mutant_motif2_min = tf.reduce_min(probs_mutant_motif2, axis=0)
    
    probs_mutant_joint = tf.stack(example_probs_list[(1 + motif1_batch_size + motif2_batch_size):])
    print probs_mutant_joint.get_shape()
    prob_mutant_joint_min = tf.reduce_min(probs_mutant_joint, axis=0)
    
    synergy_score = tf.divide(prob_orig, tf.divide(tf.add(prob_mutant_motif1_min, prob_mutant_motif2_min), 2))

    synergy_score2 = tf.divide(prob_orig - prob_mutant_joint_min,
                               tf.add(tf.subtract(prob_mutant_motif1_min, prob_mutant_joint_min),
                                      tf.subtract(prob_mutant_motif2_min, prob_mutant_joint_min)))


    # TODO(dk) individual motif scores
    # this gives you the multiplier (coefficient) for each motif, relative to each other
    motif1_score = tf.divide(prob_mutant_motif2_min, prob_mutant_joint_min)
    motif2_score = tf.divide(prob_mutant_motif1_min, prob_mutant_joint_min)
    
    # debug tool
    #interesting_outputs = [synergy_score, prob_orig, logit_orig, logit_mutant_motif1_min, logit_mutant_motif2_min, logits, all_mutants_batch, max_indices]
    
    # output the ratio
    return [synergy_score2, motif1_score, motif2_score]
