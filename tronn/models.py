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

def danq_untied_conv(features, labels, config, is_training=True):
    filters = config.get('filters', 320)
    kernel = config.get('kernel', 26)
    rnn_units = config.get('rnn_units', 320)
    fc_units = config.get('fc_units', 925)
    conv_drop = config.get('conv_drop', 0.2)
    rnn_drop = config.get('rnn_drop', 0.5)
    untied_rnn = 'untied_rnn' in config
    untied_fc = 'untied_fc' in config
    num_labels = int(labels.get_shape()[-1])

    logits = []
    for task in xrange(num_labels):
        net = slim.conv2d(features, filters, kernel_size=[1,kernel], stride=[1,1], activation_fn=tf.nn.relu, padding='VALID')
        net = slim.max_pool2d(net, kernel_size=[1,kernel/2], stride=[1,kernel/2], padding='VALID')
        net = slim.dropout(net, keep_prob=1.0-conv_drop, is_training=is_training)
        net = tf.squeeze(net, axis=1)#remove extra dim that was added so we could use conv2d. Results in batchXtimeXdepth
        rnn_inputs = tf.unstack(net, axis=1, name='unpack_time_dim')
        with tf.variable_scope('shared', reuse=task>0):
            cell_fw = tf.contrib.rnn.LSTMBlockCell(rnn_units)
            cell_bw = tf.contrib.rnn.LSTMBlockCell(rnn_units)
            outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
            net = tf.concat([state_fw[1], state_bw[1]], axis=1)
            net = slim.dropout(net, keep_prob=1.0-rnn_drop, is_training=is_training)
            net = slim.fully_connected(net, fc_units, activation_fn=tf.nn.relu)
            logit = slim.fully_connected(net, 1, activation_fn=None)
        logits.append(logit)
    logits = tf.concat(logits, axis=1)
    return logits

def danq(features, labels, config, is_training=True):
    filters = config.get('filters', 320)
    kernel = config.get('kernel', 26)
    rnn_units = config.get('rnn_units', 320)
    fc_units = config.get('fc_units', 925)
    conv_drop = config.get('conv_drop', 0.2)
    rnn_drop = config.get('rnn_drop', 0.5)
    untied_rnn = 'untied_rnn' in config
    untied_fc = 'untied_fc' in config
    num_labels = int(labels.get_shape()[-1])

    #conv
    net = slim.conv2d(features, filters, kernel_size=[1,kernel], stride=[1,1], activation_fn=tf.nn.relu, padding='VALID')
    net = slim.max_pool2d(net, kernel_size=[1,kernel/2], stride=[1,kernel/2], padding='VALID')
    net = slim.dropout(net, keep_prob=1.0-conv_drop, is_training=is_training)

    #rnn
    net = tf.squeeze(net, axis=1)#remove extra dim that was added so we could use conv2d. Results in batchXtimeXdepth
    rnn_inputs = tf.unstack(net, axis=1, name='unpack_time_dim')

    if untied_rnn:
        logits = []
        for task in xrange(num_labels):
            with tf.variable_scope('task%d'%task):
                cell_fw = tf.contrib.rnn.LSTMBlockCell(rnn_units)
                cell_bw = tf.contrib.rnn.LSTMBlockCell(rnn_units)
                outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
                task_net = tf.concat([state_fw[1], state_bw[1]], axis=1)
                task_net = slim.dropout(task_net, keep_prob=1-rnn_drop, is_training=is_training)
                task_net = slim.fully_connected(task_net, fc_units, activation_fn=tf.nn.relu)
                task_logit = slim.fully_connected(task_net, 1, activation_fn=None)
            logits.append(task_logit)
        logits = tf.concat(logits, axis=1)
    else:
        cell_fw = tf.contrib.rnn.LSTMBlockCell(rnn_units)
        cell_bw = tf.contrib.rnn.LSTMBlockCell(rnn_units)
        outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
        net = tf.concat([state_fw[1], state_bw[1]], axis=1)
        net = slim.dropout(net, keep_prob=1.0-rnn_drop, is_training=is_training)
        if untied_fc:
            logits = []
            for task in xrange(num_labels):
                with tf.variable_scope('task%d'%task):
                    task_net = slim.fully_connected(net, fc_units, activation_fn=tf.nn.relu)
                    task_logit = slim.fully_connected(net, 1, activation_fn=None)
                logits.append(task_logit)
            logits = tf.concat(logits, axis=1)
        else:
            net = slim.fully_connected(net, fc_units, activation_fn=tf.nn.relu)
            logits = slim.fully_connected(net, num_labels, activation_fn=None)
    return logits

def _residual_block(net, depth, pooling_info=(None, None)):
    first_stride = 1
    depth_in = net.get_shape()[-1]
    if depth_in!=depth:
        net = slim.batch_norm(net)
        pooling, pooling_stride = pooling_info
        if pooling=='conv':
            first_stride = pooling_stride
        elif pooling=='max':
            net = slim.max_pool2d(net, stride=[1, pooling_stride])#downsample for both shortcut and conv branch
            #no need to stride in conv branch since we have already downsampled
        elif pooling is not None:
            raise Exception('unrecognized pooling: %s'%pooling_info)
        shortcut = slim.conv2d(net, depth, kernel_size=[1, 1], stride=[1, first_stride])
    else:
        shortcut = net
        net = slim.batch_norm(net)
    net = slim.conv2d(net, depth, stride=[1, first_stride])
    net = slim.batch_norm(net)
    net = slim.conv2d(net, depth, stride=[1, 1])
    net = shortcut + net
    return net

def _resnet(features, kernel, initial_filters, stages, pooling_info, is_training=True):
    with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], kernel_size=[1, kernel], padding='SAME'):
            with slim.arg_scope([slim.conv2d], activation_fn=None):
                # We do not include batch normalization or activation functions in embed because the first ResNet unit will perform these.
                net = slim.conv2d(features, initial_filters, scope='embed')
                for i, stage in enumerate(stages):
                    with tf.variable_scope('stage%d'%i):
                        num_blocks, depth = stage
                        for j in xrange(num_blocks):
                            with tf.variable_scope('block%d'%j):
                                net = _residual_block(net, depth, pooling_info if (j==0 and i>0) else None)
        net = slim.batch_norm(net)
    return net

def conv_fc(features, labels, config, is_training=True):
    kernel = config.get('kernel', 3)
    initial_filters = config.get('initial_filters', 32)
    stages = config.get('stages', [(1, 32),(1, 64),(1, 128),(1, 256)])
    pooling = config.get('pooling', 'max')
    pooling_stride = config.get('pooling_stride', 2)
    final_pooling = config.get('final_pooling', 'global_mean')
    fc_units = config.get('fc_units', 1024)
    fc_layers = config.get('fc_layers', 2)
    drop = config.get('drop', 0.0)
    num_labels = int(labels.get_shape()[-1])

    pooling_info = (pooling, pooling_stride)
    net = _resnet(features, kernel, initial_filters, stages, pooling_info, is_training)

    if final_pooling == 'global_mean':
        net = tf.reduce_mean(net, axis=[1,2], name='global_average_pooling')
    elif final_pooling == 'global_max':
        net = tf.reduce_max(net, axis=[1,2], name='global_max_pooling')
    elif final_pooling == 'global_k_max':
        net = tf.squeeze(net, axis=1)#remove width that was used for conv2d; result is batch x time x dim
        net_time_last = tf.transpose(net, perm=[0,2,1])
        net_time_last = nn_ops.order_preserving_k_max(net_time_last, k=8)
    elif final_pooling is not None:
        raise Exception('Unrecognized final_pooling: %s'% final_pooling)

    if len(net.get_shape().as_list())>2:
        net = slim.flatten(net, scope='flatten')

    with slim.arg_scope([slim.fully_connected], activation_fn=None):
        with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
            with slim.arg_scope([slim.dropout], keep_prob=1.0-drop, is_training=is_training):
                for i in xrange(fc_layers):
                    with tf.variable_scope('fc%d'%i):
                        net = slim.fully_connected(net, fc_units)
                        net = slim.batch_norm(net)
                        net = slim.dropout(net)
        logits = slim.fully_connected(net, num_labels, scope='logits')
    return logits

# def conv_rnn(features, labels, config, is_training=True):
#     kernel = config.get('kernel', 3)
#     initial_filters = config.get('initial_filters', 32)
#     stages = config.get('stages', [(1, 32),(1, 64),(1, 128),(1, 256)])
#     pooling = config.get('pooling', 'max')
#     final_pooling = config.get('final_pooling', 'global_mean')
#     fc_units = config.get('fc_units', 1024)
#     fc_layers = config.get('fc_layers', 2)
#     drop = config.get('drop', 0.0)
#     num_labels = int(labels.get_shape()[-1])

#     net = _resnet(features, num_blocks=6, initial_filters=16, is_training=is_training)
#     depth = net.get_shape().as_list()[-1]
#     net = tf.squeeze(net, axis=1)#remove extra dim that was added so we could use conv2d. Results in batchXtimeXdepth
#     rnn_inputs = tf.unstack(net, axis=1, name='unpack_time_dim')
#     cell_fw = tf.contrib.rnn.LSTMBlockCell(depth)
#     cell_bw = tf.contrib.rnn.LSTMBlockCell(depth)
#     outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
#     if use_only_final_state:
#         state_avg = tf.div(tf.add(state_fw[1], state_bw[1]), 2, name='average_fwbw_states')#use final output(state) from fw and bw pass
#         net = state_avg
#     else:
#         outputs_fwbw_sum = tf.add_n(outputs_fwbw_list)
#         outputs_fw_sum, outputs_bw_sum = tf.split(outputs_fwbw_sum, 2, axis=1)
#         outputs_avg = tf.div(outputs_fw_sum + outputs_bw_sum, 2, name='average_fwbw_outputs')
#         net = outputs_avg
#     net = slim.dropout(net, keep_prob=1.0, is_training=is_training)
#     logits = slim.fully_connected(net, int(labels.get_shape()[-1]), activation_fn=None, scope='logits')
#     return logits

models = {}
models['basset'] = basset
models['danq'] = danq
models['danq_untied_conv'] = danq_untied_conv
#models['conv_rnn'] = conv_rnn
models['conv_fc'] = conv_fc