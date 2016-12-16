""" Contains gene regulation nets

Currently implemented:

- Basset (Kelley et al Genome Research 2016)

"""

import layers 
import tensorflow as tf
import tensorflow.contrib.slim as slim


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
    maxnorm(model_state, 7)

    return net

