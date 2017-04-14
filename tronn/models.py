""" Contains gene regulation nets

Currently implemented:

- Basset (Kelley et al Genome Research 2016)

"""
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import nn_ops

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

def mlp_module(features, num_labels, fc_dim, fc_layers, dropout=0.0, l2=0.0, is_training=True):
    net = features
    with slim.arg_scope([slim.fully_connected], activation_fn=None, weights_regularizer=slim.l2_regularizer(l2)):
        for i in xrange(fc_layers):
            with tf.variable_scope('fc%d'%i):
                net = slim.fully_connected(net, fc_dim, biases_initializer=None)
                net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training)
                net = slim.dropout(net, keep_prob=1.0-dropout, is_training=is_training)
        logits = slim.fully_connected(net, num_labels, scope='logits')
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
        with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=None, weights_initializer=layers.variance_scaling_initializer(), biases_initializer=None):
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
    num_days = int(labels.get_shape()[-1])

    net = basset_conv_module(features, is_training)
    net = final_pool(net, config['final_pool'])
    if config['temporal']:
        logits = temporal_pred_module(net, num_days, share_logistic_weights=True, is_training=is_training)
    else:
        logits = mlp_module(net, num_days, config['fc_dim'], config['fc_layers'], is_training=is_training)
    # Torch7 style maxnorm
    nn_ops.maxnorm(norm_val=7)

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

def _residual_block(net_in, depth, pooling_info=(None, None), first=False, use_shortcut=True):
    first_stride = 1
    depth_in = net_in.get_shape().as_list()[-1]
    if depth_in!=depth and not first:
        pooling, pooling_stride = pooling_info
        if pooling=='conv':
            net_preact = slim.batch_norm(net_in)
            first_stride = pooling_stride
        elif pooling=='max':
            net = slim.max_pool2d(net_in, stride=[1, pooling_stride])#downsample for both shortcut and conv branch
            net_preact = slim.batch_norm(net_in)
        else:
            raise Exception('unrecognized pooling: %s'%pooling_info)
    else:
        net_preact = slim.batch_norm(net_in)
    net = slim.conv2d(net_preact, depth, stride=[1, first_stride])
    net = slim.batch_norm(net)
    net = slim.conv2d(net, depth, stride=[1, 1])

    if use_shortcut:
        if depth_in != depth:
            paddings = [(0,0),(0,0),(0,0),((depth-depth_in)/2, int(math.ceil((depth-depth_in)/2)))]
            shortcut = tf.pad(net_preact, paddings)
        elif first:
            shortcut = net_preact
        else:
            shortcut = net_in
        net = net + shortcut
    return net

def _resnet(features, initial_conv, kernel, stages, pooling_info, l2, is_training=True):
    with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], kernel_size=[1, kernel], padding='SAME'):
            with slim.arg_scope([slim.conv2d], activation_fn=None, weights_regularizer=slim.l2_regularizer(l2), weights_initializer=layers.variance_scaling_initializer(), biases_initializer=None):
                # We do not include batch normalization or activation functions in embed because the first ResNet unit will perform these.
                initial_filters, initial_kernel, initial_stride = initial_conv
                net = slim.conv2d(features, initial_filters, kernel_size=[1, initial_kernel], stride=[1,initial_stride], scope='embed')
                for i, stage in enumerate(stages):
                    with tf.variable_scope('stage%d'%i):
                        num_blocks, depth = stage
                        for j in xrange(num_blocks):
                            with tf.variable_scope('block%d'%j):
                                net = _residual_block(net, depth, pooling_info, first=(i==0 and j==0))
        net = slim.batch_norm(net)
    return net

def resnet(features, labels, config, is_training=True):
    initial_conv = config.get('initial_conv', (32, 3, 1))
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
