""" Contains gene regulation nets

Currently implemented:

- Basset (Kelley et al Genome Research 2016)

"""


import tensorflow as tf
import tensorflow.contrib.slim as slim
import nn_ops

def _final_pool(net, pool):
    if pool is None:
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

def basset(features, labels, config, is_training=True):
    '''
    Basset - Kelley et al Genome Research 2016
    '''
    with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
        with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=None):
            net = slim.conv2d(features, 300, [1, 19])
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [1, 3], stride=[1, 3])

            net = slim.conv2d(net, 200, [1, 11])
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [1, 4], stride=[1, 4])

            net = slim.conv2d(net, 200, [1, 7])
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [1, 4], stride=[1, 4])

        net = _final_pool(net, config['final_pool'])
        for i in xrange(config['fc_layers']):
            with tf.variable_scope('fc%d'%i):
                net = slim.fully_connected(net, 1000, activation_fn=None)
                net = slim.batch_norm(net)
                net = slim.dropout(net, keep_prob=0.7, is_training=is_training)
        logits = slim.fully_connected(net, int(labels.get_shape()[-1]), activation_fn=None)

    # Torch7 style maxnorm
    nn_ops.maxnorm(norm_val=7)

    return logits




# def _residual_block(net, depth, pooling_info=(None, None)):
#     first_stride = 1
#     depth_in = net.get_shape()[-1]
#     if depth_in!=depth:
#         net = slim.batch_norm(net)
#         pooling, pooling_stride = pooling_info
#         if pooling=='conv':
#             first_stride = pooling_stride
#         elif pooling=='max':
#             net = slim.max_pool2d(net, stride=[1, pooling_stride])#downsample for both shortcut and conv branch
#             #no need to stride in conv branch since we have already downsampled
#         elif pooling is not None:
#             raise Exception('unrecognized pooling: %s'%pooling_info)
#         shortcut = slim.conv2d(net, depth, kernel_size=[1, 1], stride=[1, first_stride])
#     else:
#         shortcut = net
#         net = slim.batch_norm(net)
#     net = slim.conv2d(net, depth, stride=[1, first_stride])
#     net = slim.batch_norm(net)
#     net = slim.conv2d(net, depth, stride=[1, 1])
#     net = shortcut + net
#     return net

# def _resnet(features, kernel, initial_filters, stages, pooling_info, is_training=True):
#     with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
#         with slim.arg_scope([slim.conv2d, slim.max_pool2d], kernel_size=[1, kernel], padding='SAME'):
#             with slim.arg_scope([slim.conv2d], activation_fn=None):
#                 # We do not include batch normalization or activation functions in embed because the first ResNet unit will perform these.
#                 net = slim.conv2d(features, initial_filters, scope='embed')
#                 for i, stage in enumerate(stages):
#                     with tf.variable_scope('stage%d'%i):
#                         num_blocks, depth = stage
#                         for j in xrange(num_blocks):
#                             with tf.variable_scope('block%d'%j):
#                                 net = _residual_block(net, depth, pooling_info if (j==0 and i>0) else None)
#         net = slim.batch_norm(net)
#     return net

# def conv_fc(features, labels, config, is_training=True):
#     kernel = config.get('kernel', 3)
#     initial_filters = config.get('initial_filters', 32)
#     stages = config.get('stages', [(1, 32),(1, 64),(1, 128),(1, 256)])
#     pooling = config.get('pooling', 'max')
#     pooling_stride = config.get('pooling_stride', 2)
#     final_pooling = config.get('final_pooling', 'global_mean')
#     fc_units = config.get('fc_units', 1024)
#     fc_layers = config.get('fc_layers', 1)
#     drop = config.get('drop', 0.0)
#     num_labels = int(labels.get_shape()[-1])

#     pooling_info = (pooling, pooling_stride)
#     net = _resnet(features, kernel, initial_filters, stages, pooling_info, is_training)

#     if final_pooling == 'global_mean':
#         net = tf.reduce_mean(net, axis=[1,2], name='global_average_pooling')
#     elif final_pooling == 'global_max':
#         net = tf.reduce_max(net, axis=[1,2], name='global_max_pooling')
#     elif final_pooling == 'global_k_max':
#         net = tf.squeeze(net, axis=1)#remove width that was used for conv2d; result is batch x time x dim
#         net_time_last = tf.transpose(net, perm=[0,2,1])
#         net_time_last = nn_ops.order_preserving_k_max(net_time_last, k=8)
#     elif final_pooling is not None:
#         raise Exception('Unrecognized final_pooling: %s'% final_pooling)

#     if len(net.get_shape().as_list())>2:
#         net = slim.flatten(net, scope='flatten')

#     with slim.arg_scope([slim.fully_connected], activation_fn=None):
#         with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
#             with slim.arg_scope([slim.dropout], keep_prob=1.0-drop, is_training=is_training):
#                 for i in xrange(fc_layers):
#                     with tf.variable_scope('fc%d'%i):
#                         net = slim.fully_connected(net, fc_units)
#                         net = slim.batch_norm(net)
#                         net = slim.dropout(net)
#         logits = slim.fully_connected(net, num_labels, scope='logits')
#     return logits

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
# models['danq'] = danq
# models['danq_untied_fc'] = danq_untied_fc
# models['conv_rnn'] = conv_rnn
# models['conv_fc'] = conv_fc