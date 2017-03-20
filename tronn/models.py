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


models = {}
models['basset'] = basset
