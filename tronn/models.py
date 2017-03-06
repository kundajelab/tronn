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

def danq(features, labels, config, is_training=True):
    #conv
    net = slim.conv2d(features, 320, kernel_size=[1,26], stride=[1,1], activation_fn=tf.nn.relu, padding='VALID')
    net = slim.max_pool2d(net, kernel_size=[1,13], stride=[1,13], padding='VALID')
    net = slim.dropout(net, keep_prob=0.8, is_training=is_training)

    print 'pre rnn', net.get_shape().as_list()
    #rnn
    net = tf.squeeze(net, axis=1)#remove extra dim that was added so we could use conv2d. Results in batchXtimeXdepth
    rnn_inputs = tf.unstack(net, axis=1, name='unpack_time_dim')

    if config['untied_rnn']:
        logits = []
        for task in xrange(len(labels.get_shape().as_list())):
            cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config['rnn_units'])
            cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config['rnn_units'])
            outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
            task_net = tf.concat([state_fw[1], state_bw[1]], axis=1)
            task_net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
            task_net = slim.fully_connected(net, config['fc_units'], activation_fn=tf.nn.relu)
            task_logit = slim.fully_connected(net, 1, activation_fn=None)
            logits.append(task_logit)
        logits = tf.stack(logits)
    else:
        cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config['rnn_units'])
        cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config['rnn_units'])
        outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
        net = tf.concat([state_fw[1], state_bw[1]], axis=1)
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
        if config['untied_fc']:
            logits = []
            for task in xrange(len(labels.get_shape().as_list())):
                task_net = slim.fully_connected(net, config['fc_units'], activation_fn=tf.nn.relu)
                task_logit = slim.fully_connected(net, 1, activation_fn=None)
                logits.append(task_logit)
            logits = tf.stack(logits)
        else:
            net = slim.fully_connected(net, 925, activation_fn=tf.nn.relu)
            logits = slim.fully_connected(net, int(labels.get_shape()[-1]), activation_fn=None)
    return logits

def _residual_block(net, depth, down_sampling=None):
    first_stride = 1
    depth_in = net.get_shape()[-1]
    if depth_in!=depth:
        net = slim.batch_norm(net)
        if down_sampling is None:
            pass
        elif down_sampling=='conv_stride':
            first_stride = 2
        elif down_sampling=='max_pool':
            net = slim.max_pool2d(net, stride=[1, 2])#downsample for both shortcut and conv branch
            #no need to stride in conv branch since we have already downsampled
        else:
            raise Exception('unrecognized down_sampling: %s'%down_sampling)
        shortcut = slim.conv2d(net, depth, kernel_size=[1, 1], stride=[1, first_stride])
    else:
        shortcut = net
        net = slim.batch_norm(net)
    net = slim.conv2d(net, depth, stride=[1, first_stride])
    net = slim.batch_norm(net)
    net = slim.conv2d(net, depth, stride=[1, 1])
    net = shortcut + net
    return net

def _resnet(features, initial_depth, stages, down_sampling='max_pool', is_training=True):
    net = features
    with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
        #conv
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], kernel_size=[1, 3], padding='SAME'):
            with slim.arg_scope([slim.conv2d], activation_fn=None):
                # We do not include batch normalization or activation functions in embed because the first ResNet unit will perform these.
                net = slim.conv2d(net, initial_depth, scope='embed')
                for i, stage in enumerate(stages):
                    with tf.variable_scope('stage%d'%i):
                        num_blocks, depth = stage
                        for j in xrange(num_blocks):
                            with tf.variable_scope('block%d'%j):
                                net = _residual_block(net, depth, down_sampling if (j==0 and i>0) else None)
        net = slim.batch_norm(net)
    return net

def conv_rnn(features, labels, use_only_final_state=False, is_training=True):
    net = _resnet(features, num_blocks=6, initial_depth=16, is_training=is_training)
    depth = net.get_shape().as_list()[-1]
    net = tf.squeeze(net, axis=1)#remove extra dim that was added so we could use conv2d. Results in batchXtimeXdepth
    rnn_inputs = tf.unstack(net, axis=1, name='unpack_time_dim')
    cell_fw = tf.contrib.rnn.LSTMBlockCell(depth)
    cell_bw = tf.contrib.rnn.LSTMBlockCell(depth)
    outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
    if use_only_final_state:
        state_avg = tf.div(tf.add(state_fw[1], state_bw[1]), 2, name='average_fwbw_states')#use final output(state) from fw and bw pass
        net = state_avg
    else:
        outputs_fwbw_sum = tf.add_n(outputs_fwbw_list)
        outputs_fw_sum, outputs_bw_sum = tf.split(outputs_fwbw_sum, 2, axis=1)
        outputs_avg = tf.div(outputs_fw_sum + outputs_bw_sum, 2, name='average_fwbw_outputs')
        net = outputs_avg
    net = slim.dropout(net, keep_prob=1.0, is_training=is_training)
    logits = slim.fully_connected(net, int(labels.get_shape()[-1]), activation_fn=None, scope='logits')
    return logits

#configs
#resnet
#   initial depth
#   stages
#   filter sizes
#   pooling
#fc
#   pooling
#   num layers

def conv_fc(features, labels, is_training=True, pre_fc_pooling='global_mean'):
    net = _resnet(features, initial_depth=32, stages=[(1, 32),(1, 64),(1, 128),(1, 256)], is_training=is_training)

    print 'post_resnet shape: %s'%net.get_shape().as_list()
    
    if pre_fc_pooling is None:
        pass #net = slim.avg_pool2d(net, kernel_size=[1,3], stride=[1,2], padding='SAME')
    elif pre_fc_pooling == 'global_mean':
        net = tf.reduce_mean(net, axis=[1,2], name='global_average_pooling')
    elif pre_fc_pooling == 'global_max':
        net = tf.reduce_max(net, axis=[1,2], name='global_max_pooling')
    elif pre_fc_pooling == 'global_k_max':
        net = tf.squeeze(net, axis=1)#remove width that was used for conv2d; result is batch x time x dim
        net_time_last = tf.transpose(net, perm=[0,2,1])
        print 'pre_pooling shape: %s'%net_time_last.get_shape().as_list()
        net_time_last = nn_ops.order_preserving_k_max(net_time_last, k=8)
        print 'post_pooling shape: %s'%net_time_last.get_shape().as_list()
        net = tf.transpose(net_time_last, perm=[0,2,1])
    else:
        raise Exception('Unrecognized pre_fc_pooling: %s'% pre_fc_pooling)

    print 'pre_flatten shape: %s'%net.get_shape().as_list()
    if len(net.get_shape().as_list())>2:
        net = slim.flatten(net, scope='flatten')

    dim = net.get_shape().as_list()[-1]
    fc_dim = 1024#dim if pre_fc_pooling else 1024
    print 'fc: in_dim, out_dim: %d, %d'%(dim, fc_dim)
    num_fc_layers = 2
    with slim.arg_scope([slim.fully_connected], activation_fn=None):
        with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
            with slim.arg_scope([slim.dropout], keep_prob=1.0, is_training=is_training):
                for i in xrange(num_fc_layers):
                    with tf.variable_scope('fc%d'%i):
                        net = slim.fully_connected(net, fc_dim)
                        net = slim.batch_norm(net)
                        net = slim.dropout(net)
        logits = slim.fully_connected(net, int(labels.get_shape()[-1]), scope='logits')
    return logits


models = {}
models['basset'] = basset
models['danq'] = danq
models['conv_rnn'] = conv_rnn
models['conv_fc'] = conv_fc