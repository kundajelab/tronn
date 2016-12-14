# run Basset

import os
import math
import layers 
import tensorflow as tf
import numpy as np
import threading
import h5py

from sklearn import metrics
from random import shuffle

# Simplified imports?
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


def basset_tronn(features, labels, model_state):
    '''
    Basset - Kelley et al Genome Research 2016
    Reconfigured for tf-slim
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
        net = slim.conv2d(features, 200, [1, 11], scope='conv2')
        net = slim.max_pool2d(net, [1, 4], stride=4, scope='conv2')

        # Layer 3: conv layer to batch norm to relu to max pool. 
        net = slim.conv2d(features, 200, [1, 7], scope='conv3')
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


def basset(tasks, feature_batch, label_batch, model_state):
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
    training_ops.append(conv_bn_1_op)

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
    training_ops.append(conv_bn_2_op)

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
    training_ops.append(conv_bn_3_op)

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
    training_ops.append(fc_bn_1_op)

    fc_relu_1 = tf.nn.relu(fc_dense_batchnorm_1, 'fc_relu_1')
    fc_dropout_1 = layers.dropout(fc_relu_1, fc_dropout_1, 'fc_dropout_1', model_state) # Skip if not training (looking at validation)

    # --------------------------------------------------------------
    # Match basset - Layer 5: fully connected layer to relu to dropout
    # --------------------------------------------------------------
    fc_size_2 = 1000
    fc_dropout_2 = 0.7 # NOTE this is the opposite definition from torch
    
    fc_dense_2, regularization_2, weights_fc_2 = layers.inner_prod(fc_dropout_1, fc_size_2, 'fc_dense_2', model_state) 

    fc_dense_batchnorm_2, fc_bn_2_op = layers.batchnorm(fc_dense_2, 'fc_dense_batchnorm_2', model_state)
    training_ops.append(fc_bn_2_op)

    fc_relu_2 = tf.nn.relu(fc_dense_batchnorm_2, 'fc_relu_2')
    fc_dropout_2 = layers.dropout(fc_relu_2, fc_dropout_2, 'fc_dropout_2', model_state) # Skip if not training (looking at validation)

    # Task output nodes
    out, regularization_out, weights_out = layers.inner_prod(fc_dropout_2, tasks, 'out', model_state)

    # --------------------------------------------------------------
    # Loss, predictions, optimizer
    # --------------------------------------------------------------

    # Predictions (to get metrics)
    train_prediction = tf.nn.sigmoid(out)

    # Loss: softmax cross entropy with logits
    loss = tf.nn.sigmoid_cross_entropy_with_logits(out, label_batch) 
    loss_sum = tf.reduce_sum(loss) 

    # Optimizer
    #optimizer = tf.train.AdamOptimizer(0.002, 0.9, 0.999)
    #optimizer = tf.train.AdamOptimizer(0.00002, 0.9, 0.999)
    #optimizer = tf.train.RMSPropOptimizer(0.0002, decay=0.98, momentum=0.0, epsilon=1e-8)
    optimizer = tf.train.RMSPropOptimizer(0.002, decay=0.98, momentum=0.0, epsilon=1e-8) 
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Training operation
    train_op = optimizer.minimize(loss_sum, global_step=global_step)

    # Max norm updates: get the weights and update those when training_placeholder is 2 (the max norm after optimizing at each iteration)
    maxnorm_ops = [weights_conv_1, weights_conv_2, weights_conv_3, weights_fc_1, weights_fc_2, weights_out]

    # --------------------------------------------------------------
    # Evaluation tools: (input)*(activation gradient)
    # --------------------------------------------------------------
    #[(feature_grad, feature_batch_var)] = optimizer.compute_gradients(loss_sum, var_list=[feature_batch])
    [feature_grad] = tf.gradients(loss_sum, [feature_batch])
    importance = tf.mul(feature_batch, feature_grad, 'importance_by_input_grad')
    
    return train_op, training_ops, maxnorm_ops, train_prediction, loss_sum, convlayer_relu_1, importance


def deepsea(tasks, feature_batch, label_batch, model_state):
    '''
    deepSEA - Zhou and Troyanskaya 2015
    '''

    # --------------------------------------------------------------
    # Layer 1 - convolution, thresholded, max pool, dropout
    # --------------------------------------------------------------
    num_filters_1 = 320
    kernel_widths_1 = 8 # NOTE: I think they flipped the axis, so that height is width??
    pool_widths_1 = 4
    pool_stride_widths_1 = 4
    convlayer_dropout_prob_1 = 0.8 # Keep prob
    
    convlayer_conv_1, regularization_conv_1, weights_conv_1 = layers.conv1d(feature_batch, kernel_widths_1, 1, num_filters_1, 'convlayer_conv_1', model_state, relu=False)
    convlayer_threshold_1 = layers.threshold(convlayer_conv_1, 0, 1e-6, 'convlayer_threshold_1')
    convlayer_maxpool_1 = layers.maxpool(convlayer_threshold_1, [1, pool_widths_1], [1, pool_stride_widths_1], 'convlayer_maxpool_1')
    convlayer_dropout_1 = layers.dropout(convlayer_maxpool_1, convlayer_dropout_prob_1, 'convlayer_dropout_1', model_state) # Skip if not training (looking at validation)
        
    # --------------------------------------------------------------
    # Layer 2 - convolution, thresholded, max pool, dropout
    # --------------------------------------------------------------
    num_filters_2 = 480
    kernel_widths_2 = 8
    pool_widths_2 = 4
    pool_stride_widths_2 = 4
    convlayer_dropout_prob_2 = 0.8 # Keep prob

    convlayer_conv_2, regularization_conv_2, weights_conv_2 = layers.conv1d(convlayer_dropout_1, kernel_widths_2, 1, num_filters_2, 'convlayer_conv_2', model_state, relu=False)
    convlayer_threshold_2 = layers.threshold(convlayer_conv_2, 0, 1e-6, 'convlayer_threshold_2')
    convlayer_maxpool_2 = layers.maxpool(convlayer_threshold_2, [1, pool_widths_2], [1, pool_stride_widths_2], 'convlayer_maxpool_2')
    convlayer_dropout_2 = layers.dropout(convlayer_maxpool_2, convlayer_dropout_prob_2, 'convlayer_dropout_2', model_state) # Skip if not training (looking at validation)
    
    # --------------------------------------------------------------
    # Layer 3 - convolution, thresholded, dropout
    # --------------------------------------------------------------
    num_filters_3 = 960
    kernel_widths_3 = 8
    convlayer_dropout_prob_3 = 0.5 # Keep prob

    convlayer_conv_3, regularization_conv_3, weights_conv_3 = layers.conv1d(convlayer_dropout_2, kernel_widths_2, 1, num_filters_2, 'convlayer_conv_2', model_state, relu=False)
    convlayer_threshold_3 = layers.threshold(convlayer_conv_3, 0, 1e-6, 'convlayer_threshold_3')
    convlayer_dropout_3 = layers.dropout(convlayer_threshold_3, convlayer_dropout_prob_3, 'convlayer_dropout_3', model_state) # Skip if not training (looking at validation)

    # Flatten to put into fully connected layers
    flat_1 = layers.flatten(convlayer_dropout_3, 'flat_1')
    
    # --------------------------------------------------------------
    # Layer 4 - fully connected, threshold
    # --------------------------------------------------------------
    fc_dense_1, regularization_1, weights_fc_1 = layers.inner_prod(flat_1, tasks, 'fc_dense_1', model_state) 
    fc_threshold_1 = layers.threshold(fc_dense_1, 0, 1e-6, 'fc_threshold_1')
    
    # out nodes
    out, regularization_out, weights_out = layers.inner_prod(fc_threshold_1, tasks, 'out', model_state)

    # --------------------------------------------------------------
    # Loss, predictions, optimizer
    # --------------------------------------------------------------

    # Predictions (to get metrics)
    train_prediction = tf.nn.sigmoid(out)

    # Loss: softmax cross entropy (BCECriterion) with logits, and L1 penalty to last layer
    loss = tf.nn.sigmoid_cross_entropy_with_logits(out, label_batch) 
    loss_sum = tf.reduce_sum(loss) + 1e-8 * regularization_out # TODO: check if this is right

    # Optimizer: SGD with -LearningRate 1 -LearningRateDecay 8e-7 -weightDecay 1e-6  -momentum 0.9
    global_step = tf.Variable(0, name='global_step', trainable=False)
    decaying_learning_rate = tf.train.exponential_decay(1.0, global_step, 1, 8e-7)
    optimizer = tf.train.MomentumOptimizer(decaying_learning_rate, 0.9)

    # Training operation: do the weight decay here. These lines correspond to optimizer.minimize
    gradients_and_variables = optimizer.compute_gradients(loss_sum)
    decayed_gradients = [ tf.add(grad, tf.mul(1e-6, var)) for grad, var in gradients_and_variables ]
    optimizer.apply_gradients(decayed_gradients, global_step=global_step)
    
    # Max norm updates: get the weights and update those when training_placeholder is 2 (the max norm after optimizing at each iteration)
    # TODO: factor out max norm value so that it's easy to modify
    maxnorm_ops = [weights_conv_1, weights_conv_2, weights_conv_3, weights_fc_1, weights_fc_2, weights_out]
        
    return train_op, training_ops, maxnorm_ops, train_prediction, loss_sum, convlayer_relu_1
    
