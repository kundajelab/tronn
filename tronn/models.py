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
    Editing
    '''

    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            #normalizer_fn=slim.batch_norm,
            #normalizer_params={'is_training': model_state},
            weights_initializer=initializers.conv_weight_initializer(),
            #biases_initializer=layers.conv_bias_initializer()
            ):

        # Layer 1: conv layer to batch norm to relu to max pool. 
        conv1_stdev = nn_utils.calc_stdev(19, 1, 4)
        net = slim.conv2d(features, 300, [1, 19], 
            activation_fn=None,
            biases_initializer=initializers.slim_conv_bias_initializer(stdv=conv1_stdev),
            scope='conv1')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.max_pool2d(net, [1, 3], stride=[1, 3], scope='conv1')

        # Layer 2: conv layer to batch norm to relu to max pool. 
        conv2_stdev = nn_utils.calc_stdev(11, 1, 300)
        net = slim.conv2d(net, 200, [1, 11], 
            activation_fn=None,
            biases_initializer=initializers.slim_conv_bias_initializer(stdv=conv2_stdev),
            scope='conv2')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.max_pool2d(net, [1, 4], stride=[1, 4], scope='conv2')

        # Layer 3: conv layer to batch norm to relu to max pool. 
        conv3_stdev = nn_utils.calc_stdev(7, 1, 200)
        net = slim.conv2d(net, 200, [1, 7], 
            activation_fn=None,
            biases_initializer=initializers.slim_conv_bias_initializer(stdv=conv3_stdev),
            scope='conv3')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.max_pool2d(net, [1, 4], stride=[1, 4], scope='conv3')

    net = slim.flatten(net, scope='flatten')

    with slim.arg_scope(
        [slim.fully_connected],
        #normalizer_fn=slim.batch_norm,
        #normalizer_params={'is_training': model_state},
        weights_initializer=initializers.fc_weight_initializer(),
        #biases_initializer=layers.fc_bias_initializer()
        ):

        # Layer 4: fully connected layer to relu to dropout
        fc1_stdev = nn_utils.calc_stdev(1, 1, 3600, style='fc')
        net = slim.fully_connected(net, 1000, 
            activation_fn=None,
            biases_initializer=initializers.slim_conv_bias_initializer(stdv=fc1_stdev),
            scope='fc1')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training)

        # Layer 5: fully connected layer to relu to dropout
        fc2_stdev = nn_utils.calc_stdev(1, 1, 1000, style='fc')
        net = slim.fully_connected(net, 1000, 
            activation_fn=None,
            biases_initializer=initializers.slim_conv_bias_initializer(stdv=fc2_stdev),
            scope='fc2')
        net = slim.batch_norm(net, center=True, scale=True, activation_fn=nn.relu)
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training)

    # OUT
    out_stdev = nn_utils.calc_stdev(1, 1, 1000, style='fc')
    net = slim.fully_connected(net, int(labels.get_shape()[-1]), 
        activation_fn=None, # TODO watch this
        weights_initializer=initializers.fc_weight_initializer(),
        biases_initializer=initializers.slim_conv_bias_initializer(stdv=out_stdev),
        scope='out')

    # Make a maxnorm op and add to update ops
    # TODO check
    nn_ops.maxnorm(norm_val=7)

    return net



