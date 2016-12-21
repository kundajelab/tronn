""" Contains gene regulation nets

Currently implemented:

- Basset (Kelley et al Genome Research 2016)

"""

import layers 
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.ops import nn


def maxnorm(model_state, norm_val=7):
    '''
    Torch7 style maxnorm. To be moved to user ops
    '''

    weights = [v for v in tf.all_variables()
               if ('weights' in v.name)]

    for weight in weights:
        maxnorm_update = weight.assign(tf.clip_by_norm(weight,
                                                       norm_val,
                                                       axes=[0]))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             maxnorm_update)
        
    return None

def calc_stdev(filter_height, filter_width, num_in_channels, style='conv'):
    if style == 'conv':
        return 1. / math.sqrt(filter_width * filter_height * num_in_channels)
    else:
        return 1. / math.sqrt(num_in_channels)


def basset(features, labels, model_state):
    '''
    Basset - Kelley et al Genome Research 2016
    '''

    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': model_state},
            weights_initializer=layers.conv_weight_initializer(),
            biases_initializer=layers.conv_bias_initializer()):

        # Layer 1: conv layer to batch norm to relu to max pool. 
        net = slim.conv2d(features, 300, [1, 19], scope='conv1')
        net = slim.max_pool2d(net, [1, 3], stride=3, scope='conv1')

        # Layer 2: conv layer to batch norm to relu to max pool. 
        net = slim.conv2d(net, 200, [1, 11], scope='conv2')
        net = slim.max_pool2d(net, [1, 4], stride=4, scope='conv2')

        # Layer 3: conv layer to batch norm to relu to max pool. 
        net = slim.conv2d(net, 200, [1, 7], scope='conv3')
        net = slim.max_pool2d(net, [1, 4], stride=4, scope='conv3')

    net = slim.flatten(net, scope='flatten')

    with slim.arg_scope(
        [slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        normalizer_params={'is_training': model_state},
        weights_initializer=layers.fc_weight_initializer(),
        biases_initializer=layers.fc_bias_initializer()):

        # Layer 4: fully connected layer to relu to dropout
        net = slim.fully_connected(net, 1000, scope='fc1')
        net = slim.dropout(net, keep_prob=0.7, is_training=model_state)

        # Layer 5: fully connected layer to relu to dropout
        net = slim.fully_connected(net, 1000, scope='fc2')
        net = slim.dropout(net, keep_prob=0.7, is_training=model_state)

    # OUT
    net = slim.fully_connected(net, int(labels.get_shape()[-1]), scope='out')

    # Make a maxnorm op and add to update ops
    # TODO check
    maxnorm(model_state, 7)

    return net


def basset_like(features, labels, model_state):
    '''
    Basset - Kelley et al Genome Research 2016
    Editing
    '''

    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            #normalizer_fn=slim.batch_norm,
            #normalizer_params={'is_training': model_state},
            weights_initializer=layers.conv_weight_initializer(),
            #biases_initializer=layers.conv_bias_initializer()
            ):

        # Layer 1: conv layer to batch norm to relu to max pool. 
        conv1_stdev = calc_stdev(19, 1, 4)
        net = slim.conv2d(features, 300, [1, 19], 
            activation_fn=None,
            biases_initializer=layers.slim_conv_bias_initializer(stdv=conv1_stdev),
            scope='conv1')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.max_pool2d(net, [1, 3], stride=3, scope='conv1')

        # Layer 2: conv layer to batch norm to relu to max pool. 
        conv2_stdev = calc_stdev(11, 1, 300)
        net = slim.conv2d(net, 200, [1, 11], 
            activation_fn=None,
            biases_initializer=layers.slim_conv_bias_initializer(stdv=conv2_stdev),
            scope='conv2')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.max_pool2d(net, [1, 4], stride=4, scope='conv2')

        # Layer 3: conv layer to batch norm to relu to max pool. 
        conv3_stdev = calc_stdev(7, 1, 200)
        net = slim.conv2d(net, 200, [1, 7], 
            activation_fn=None,
            biases_initializer=layers.slim_conv_bias_initializer(stdv=conv3_stdev),
            scope='conv3')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.max_pool2d(net, [1, 4], stride=4, scope='conv3')

    net = slim.flatten(net, scope='flatten')

    with slim.arg_scope(
        [slim.fully_connected],
        #normalizer_fn=slim.batch_norm,
        #normalizer_params={'is_training': model_state},
        weights_initializer=layers.fc_weight_initializer(),
        #biases_initializer=layers.fc_bias_initializer()
        ):

        # Layer 4: fully connected layer to relu to dropout
        fc1_stdev = calc_stdev(1, 1, 3600, style='fc')
        net = slim.fully_connected(net, 1000, 
            activation_fn=None,
            biases_initializer=layers.slim_conv_bias_initializer(stdv=fc1_stdev),
            scope='fc1')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.dropout(net, keep_prob=0.7, is_training=model_state)

        # Layer 5: fully connected layer to relu to dropout
        fc2_stdev = calc_stdev(1, 1, 1000, style='fc')
        net = slim.fully_connected(net, 1000, 
            activation_fn=None,
            biases_initializer=layers.slim_conv_bias_initializer(stdv=fc2_stdev),
            scope='fc2')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.dropout(net, keep_prob=0.7, is_training=model_state)

    # OUT
    out_stdev = calc_stdev(1, 1, 1000, style='fc')
    net = slim.fully_connected(net, int(labels.get_shape()[-1]), 
        activation_fn=None, # TODO watch this
        weights_initializer=layers.fc_weight_initializer(),
        biases_initializer=layers.slim_conv_bias_initializer(stdv=out_stdev),
        scope='out')

    # Make a maxnorm op and add to update ops
    # TODO check
    maxnorm(model_state, 7)

    return net


def basset_old(feature_batch, label_batch, model_state):
    '''
    Basset - Kelley et al 2016 Genome Research
    '''

    training_ops = []
    
    # --------------------------------------------------------------
    # Layer 1: conv layer to batch norm to relu to max pool. Fully connected convolution
    # --------------------------------------------------------------
    num_filters_1 = 300
    kernel_widths_1 = 19
    pool_widths_1 = 3
    pool_stride_widths_1 = 3

    convlayer_conv_1, regularization_conv_1, weights_conv_1 = layers.conv1d(feature_batch, kernel_widths_1, 1, num_filters_1, 'convlayer_conv_1', model_state, relu=False)

    convlayer_batchnorm_1, conv_bn_1_op = layers.batchnorm(convlayer_conv_1, 'convlayer_batchnorm_1', model_state, convolutional=True)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             conv_bn_1_op)

    convlayer_relu_1 = tf.nn.relu(convlayer_batchnorm_1, 'convlayer_relu_1') 
    convlayer_maxpool_1 = layers.maxpool(convlayer_relu_1, [1, pool_widths_1], [1, pool_stride_widths_1], 'convlayer_maxpool_1')
    
    # --------------------------------------------------------------
    # Layer 2: Add another conv layer series
    # --------------------------------------------------------------
    num_filters_2 = 200
    kernel_widths_2 = 11
    pool_widths_2 = 4
    pool_stride_widths_2 = 4

    convlayer_conv_2, regularization_conv_2, weights_conv_2 = layers.conv1d(convlayer_maxpool_1, kernel_widths_2, 1, num_filters_2, 'convlayer_conv_2', model_state, relu=False)

    convlayer_batchnorm_2, conv_bn_2_op = layers.batchnorm(convlayer_conv_2, 'convlayer_batchnorm_2', model_state, convolutional=True)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             conv_bn_2_op)

    convlayer_relu_2 = tf.nn.relu(convlayer_batchnorm_2, 'convlayer_relu_2')
    convlayer_maxpool_2 = layers.maxpool(convlayer_relu_2, [1, pool_widths_2], [1, pool_stride_widths_2], 'convlayer_maxpool_2')

    # --------------------------------------------------------------
    # Layer 3: Add another conv layer series
    # --------------------------------------------------------------
    num_filters_3 = 200
    kernel_widths_3 = 7
    pool_widths_3 = 4
    pool_stride_widths_3 = 4

    convlayer_conv_3, regularization_conv_3, weights_conv_3 = layers.conv1d(convlayer_maxpool_2, kernel_widths_3, 1, num_filters_3, 'convlayer_conv_3', model_state, relu=False)

    convlayer_batchnorm_3, conv_bn_3_op = layers.batchnorm(convlayer_conv_3, 'convlayer_batchnorm_3', model_state, convolutional=True)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             conv_bn_3_op)

    convlayer_relu_3 = tf.nn.relu(convlayer_batchnorm_3, 'convlayer_relu_3')
    convlayer_maxpool_3 = layers.maxpool(convlayer_relu_3, [1, pool_widths_3], [1, pool_stride_widths_3], 'convlayer_maxpool_3')

    # Flatten to put into fully connected layers
    flat_1 = layers.flatten(convlayer_maxpool_3, 'flat_1')
    
    # --------------------------------------------------------------
    # Match basset - Layer 4: fully connected layer to relu to dropout
    # --------------------------------------------------------------
    fc_size_1 = 1000
    fc_dropout_1 = 0.7 # NOTE this is the opposite definition from torch (keep_prob not drop_prob)

    fc_dense_1, regularization_1, weights_fc_1 = layers.inner_prod(flat_1, fc_size_1, 'fc_dense_1', model_state) 

    fc_dense_batchnorm_1, fc_bn_1_op = layers.batchnorm(fc_dense_1, 'fc_dense_batchnorm_1', model_state)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             fc_bn_1_op)

    fc_relu_1 = tf.nn.relu(fc_dense_batchnorm_1, 'fc_relu_1')
    fc_dropout_1 = layers.dropout(fc_relu_1, fc_dropout_1, 'fc_dropout_1', model_state) # Skip if not training (looking at validation)

    # --------------------------------------------------------------
    # Match basset - Layer 5: fully connected layer to relu to dropout
    # --------------------------------------------------------------
    fc_size_2 = 1000
    fc_dropout_2 = 0.7 # NOTE this is the opposite definition from torch
    
    fc_dense_2, regularization_2, weights_fc_2 = layers.inner_prod(fc_dropout_1, fc_size_2, 'fc_dense_2', model_state) 

    fc_dense_batchnorm_2, fc_bn_2_op = layers.batchnorm(fc_dense_2, 'fc_dense_batchnorm_2', model_state)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             fc_bn_2_op)

    fc_relu_2 = tf.nn.relu(fc_dense_batchnorm_2, 'fc_relu_2')
    fc_dropout_2 = layers.dropout(fc_relu_2, fc_dropout_2, 'fc_dropout_2', model_state) # Skip if not training (looking at validation)

    # Task output nodes
    out, regularization_out, weights_out = layers.inner_prod(fc_dropout_2, label_batch.get_shape()[-1], 'out', model_state)

    
    return out



