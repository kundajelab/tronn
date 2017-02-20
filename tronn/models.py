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

#TODO
#max pooling by selecting same arg over all channels

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

#TRIED:
#kernel=7,dim=64,stride=2,blocks=5
#kernel=7,dim=64,stride=3,blocks=5
#kernel=3,dim=16,stride=2,blocks=6

def _residual_block(net, dim, inrease_dim=False, down_sampling=None):
    if down_sampling:
        down_sampling_method, down_sampling_factor = down_sampling
        if down_sampling_method=='max_pool':
            net = slim.max_pool2d(net, stride=[1, down_sampling_factor])
            if inrease_dim:
                shortcut = slim.conv2d(net, dim)
            first_stride = [1, 1]
        elif down_sampling_method=='conv_stride':
            shortcut = slim.conv2d(net, dim, stride=[1, down_sampling_factor])
            first_stride = [1, down_sampling_factor]
        else:
            raise ValueError('unrecognized down_sampling: %s'%down_sampling)
    else:
        shortcut = net
        first_stride = [1, 1]
    net = slim.batch_norm(net)
    net = slim.conv2d(net, dim, stride=first_stride)
    net = slim.batch_norm(net)
    net = slim.conv2d(net, dim)
    net = shortcut + net
    return net

def custom(features, labels, is_training=True):
    net = features
    dim = 16
    with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
        #conv
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], kernel_size=[1, 3], padding='SAME'):
            with slim.arg_scope([slim.conv2d], activation_fn=None):
                net = slim.conv2d(net, dim, scope='embed')
                for block in xrange(7):
                    with tf.variable_scope('residual_block%d'%block):
                        if block==0:
                            net = _residual_block(net, dim)
                        else:
                            dim = int(dim*(2**0.5))
                            net = _residual_block(net, dim, inrease_dim=True, down_sampling=('conv_stride', 2))
        net = slim.batch_norm(net)
        #fc
        net = slim.flatten(net, scope='flatten')
        #net = tf.reduce_mean(net, axis=[1,2], name='global_average_pooling')
        with slim.arg_scope([slim.fully_connected], activation_fn=None):
            with slim.arg_scope([slim.dropout], keep_prob=1.0, is_training=is_training):
                with tf.variable_scope('fc1'):
                    net = slim.fully_connected(net, dim)
                    net = slim.batch_norm(net)
                    net = slim.dropout(net)
            logits = slim.fully_connected(net, int(labels.get_shape()[-1]), scope='logits')

    # Torch7 style maxnorm
    # nn_ops.maxnorm(norm_val=7)

    return logits

models = {}
models['basset'] = basset
models['custom'] = custom