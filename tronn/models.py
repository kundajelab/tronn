""" Contains gene regulation nets

Currently implemented:

- Basset (Kelley et al Genome Research 2016)

"""


import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.ops import nn
from tronn import initializers
from tronn import layers
from tronn import nn_ops
from tronn import nn_utils


def basset(features, labels, is_training=True):
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
                conv1_filter_size, nn_utils.get_fan_in(features)),
            biases_initializer=initializers.torch_conv_initializer(
                conv1_filter_size, nn_utils.get_fan_in(features)),
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
                conv2_filter_size, nn_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_conv_initializer(
                conv2_filter_size, nn_utils.get_fan_in(net)),
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
                conv3_filter_size, nn_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_conv_initializer(
                conv3_filter_size, nn_utils.get_fan_in(net)),
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
                nn_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_fullyconnected_initializer(
                nn_utils.get_fan_in(net)),
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
                nn_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_fullyconnected_initializer(
                nn_utils.get_fan_in(net)),
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
                nn_utils.get_fan_in(net)),
            biases_initializer=initializers.torch_fullyconnected_initializer(
                nn_utils.get_fan_in(net)),
            scope='out')

    # Torch7 style maxnorm
    nn_ops.maxnorm(norm_val=7)

    return logits



