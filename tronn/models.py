""" Contains gene regulation nets

Currently implemented:

- Basset (Kelley et al Genome Research 2016)

"""


import tensorflow as tf
import tensorflow.contrib.slim as slim
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

def mlp_module(features, num_labels, fc_layers, fc_dim, dropout=0.0, is_training=True):
    for i in xrange(fc_layers):
        with tf.variable_scope('fc%d'%i):
            net = slim.fully_connected(features, fc_dim, activation_fn=None)
            net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training)
            net = slim.dropout(net, keep_prob=1.0-dropout, is_training=is_training)
    logits = slim.fully_connected(net, num_labels, activation_fn=None)
    return logits

def temporal_pred_module(features, num_days, share_logistic_weights):
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
    return net

def basset(features, labels, config, is_training=True):
    '''
    Basset - Kelley et al Genome Research 2016
    '''
    config['temporal'] = 'temporal' in config
    config['final_pool'] = config.get('final_pool', 'flatten')
    config['fc_layers'] = config.get('fc_layers', 2)
    config['fc_dim'] = config.get('fc_dim', 1000)
    config['drop'] = config.get('drop', 0.3)
    num_days = int(labels.get_shape()[-1])

    net = basset_conv_module(features, is_training)
    net = final_pool(net, config['final_pool'])
    if config['temporal']:
        logits = temporal_pred_module(net, num_days, share_logistic_weights=True)
    else:
        logits = mlp_module(net, num_days, config['fc_layers'], config['fc_dim'], is_training=True)
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

models = {}
models['basset'] = basset
models['danq'] = danq
