"""description: contains nets for training
"""

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

from tronn.nets.tfslim import inception_v3
from tronn.nets.tfslim import resnet_v2

from tronn.util.tf_ops import maxnorm


def empty_net(inputs, params):
    """Placeholder model to pass through features without modifying them
    """
    # for all outputs, return 0 as logit - average prediction
    inputs["logits"] = tf.zeros(inputs["labels"].get_shape())
    
    return inputs, params


def final_pool(net, pool):
    """Pooling function after convolutional layers
    """
    if pool == 'flatten':
        net = slim.flatten(
            net, scope='flatten')
    elif pool == 'mean':
        net = tf.reduce_mean(
            net, axis=[1,2], name='avg_pooling')
    elif pool == 'max':
        net = tf.reduce_max(
            net, axis=[1,2], name='max_pooling')
    elif pool == 'kmax':
        # remove width that was used for conv2d; result is batch x time x dim
        net = tf.squeeze(net, axis=1)
        net_time_last = tf.transpose(net, perm=[0,2,1])
        net_time_last = nn_ops.order_preserving_k_max(net_time_last, k=8)
        net = slim.flatten(net_time_last, scope='flatten')
    elif pool is not None:
        raise Exception('Unrecognized final_pooling: %s'% pool)
    return net


def mlp_module(
        features,
        num_tasks,
        fc_dim,
        fc_layers,
        dropout=0.0,
        fc_l2=0.0,
        logit_l1=0.0,
        is_training=True,
        prefix=""):
    """MLP
    """
    net = features
    with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(fc_l2)):
        for i in xrange(fc_layers):
            with tf.variable_scope('{}fc{}'.format(prefix, i)):
                net = slim.fully_connected(
                    net,
                    fc_dim,
                    biases_initializer=None)
                net = slim.batch_norm(
                    net,
                    center=True,
                    scale=True,
                    activation_fn=tf.nn.relu,
                    is_training=is_training)
                net = slim.dropout(
                    net,
                    keep_prob=1.0-dropout,
                    is_training=is_training)
        logits = slim.fully_connected(
            net, num_tasks, 
            weights_regularizer=slim.l1_regularizer(logit_l1),
            scope='{}logits'.format(prefix))
    return logits


def mlp_module_v2(
        features,
        num_tasks,
        fc_dim,
        fc_layers,
        dropout=0.0,
        fc_l2=0.0,
        logit_l1=0.0,
        logit_dropout=0.0,
        split_before=None,
        is_training=True,
        prefix=""):
    """MLP with options for task specific layers
    
    Args:
      split_before: if None, no splitting. If index, before that index level, split layers.

    """
    # set up splits
    if split_before is not None:
        assert split_before >= 0
        assert (fc_layers - split_before) >= 0
        split_layers = fc_layers - split_before
        fc_layers = split_before
    else:
        split_layers = 0

    # set up a single FC layer
    def mlp_fc(input_tensor, fc_dim, dropout, is_training, scope_name):
        """Internal fc layer: fc, batch norm, dropout
        """
        with tf.variable_scope(scope_name):
            net = slim.fully_connected(
                input_tensor,
                fc_dim,
                biases_initializer=None)
            tf.add_to_collection("DEEPLIFT_ACTIVATIONS", net)
            net = slim.batch_norm(
                net,
                center=True,
                scale=True,
                activation_fn=tf.nn.relu,
                is_training=is_training)
            tf.add_to_collection("DEEPLIFT_ACTIVATIONS", net)
            net = slim.dropout(
                net,
                keep_prob=1.0-dropout,
                is_training=is_training)

        return net

    # stack layers
    net = features
    with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(fc_l2)):

        # first generate shared FC layers
        for i in xrange(fc_layers):
            net = mlp_fc(
                net,
                fc_dim,
                dropout,
                is_training,
                "{}fc{}".format(prefix, i))

        # then generate split FC layers (if any). If none, continue to logits
        task_specific_outputs = []
        for i in xrange(num_tasks):
            task_specific_net = net
            for j in xrange(split_layers):
                # same as above
                task_specific_net = mlp_fc(
                    task_specific_net,
                    fc_dim,
                    dropout,
                    is_training,
                    "{}task{}-fc{}".format(prefix, i, j+fc_layers))
            task_specific_outputs.append(task_specific_net)

        # adjust logits based on if any split layers
        if split_layers > 0:
            task_logits = [
                slim.fully_connected(
                    task_specific_outputs[i], 1, 
                    weights_regularizer=slim.l1_regularizer(logit_l1),
                    scope="{}task{}-logits".format(prefix, i))
                for i in xrange(len(task_specific_outputs))]
            logits = tf.concat(task_logits, 1)
        else:
            logits = slim.fully_connected(
                net, num_tasks, 
                normalizer_fn=None,
                weights_regularizer=slim.l1_regularizer(logit_l1),
                scope='{}logits'.format(prefix))

    # logit dropout
    logits = slim.dropout(
        logits,
        keep_prob=1.0-logit_dropout,
        is_training=is_training)

    return logits


def temporal_pred_module(
        features,
        num_days,
        share_logistic_weights,
        is_training=True):
    """Temporal module, use RNN. NOTE: specifically designed for timecourse, don't use otherwise for now
    """
    dim = features.shape.as_list()[1]
    day_nets = [slim.fully_connected(features, dim, activation_fn=tf.nn.relu)
                for day in xrange(num_days)]#remove relu?
    cell_fw = tf.contrib.rnn.LSTMBlockCell(dim)
    cell_bw = tf.contrib.rnn.LSTMBlockCell(dim)
    day_nets, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        cell_fw, cell_bw, day_nets, dtype=tf.float32)
    if share_logistic_weights:
        net = tf.concat([tf.expand_dims(day_nets, 1)
                         for day_net in day_nets], 1)#batch X day X 2*dim
        net = tf.reshape(net, [-1, 2*dim])#batch*day X 2*dim
        logits_flat = slim.fully_connected(net, 1, activation_fn=None)#batch*day X 1
        logits = tf.reshape(logits_flat, [-1, num_days])#batch X num_days
    else:
        day_logits = [slim.fully_connected(day_net, 1, activation_fn=None)
                      for day_net in day_nets]
        logits = tf.concat(day_logits, 1)
    return logits


def basset_conv_factorized_module(features, is_training=True, width_factor=1):
    with slim.arg_scope(
            [slim.batch_norm],
            center=True,
            scale=True,
            activation_fn=tf.nn.relu,
            is_training=is_training):
        with slim.arg_scope(
                [slim.conv2d],
                activation_fn=None,
                weights_initializer=layers.variance_scaling_initializer(),
                biases_initializer=None):
            net = slim.conv2d(features, int(width_factor*48), [1, 3])
            net = slim.batch_norm(net)
	    net = slim.conv2d(net, int(width_factor*64), [1, 3])
            net = slim.batch_norm(net)
	    net = slim.conv2d(net, int(width_factor*100), [1, 3])
            net = slim.batch_norm(net)
	    net = slim.conv2d(net, int(width_factor*150), [1, 7])
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [1, 3], stride=[1, 3])

            net = slim.conv2d(net, int(width_factor*200), [1, 7])
            net = slim.batch_norm(net)
	    net = slim.conv2d(net, int(width_factor*200), [1, 3])
            net = slim.batch_norm(net)
	    net = slim.conv2d(net, int(width_factor*200), [1, 3])
            net = slim.batch_norm(net) 
            net = slim.max_pool2d(net, [1, 4], stride=[1, 4])

            net = slim.conv2d(net, int(width_factor*200), [1, 7])
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [1, 4], stride=[1, 4])
	 
    return net


def fbasset(features, labels, config, is_training=True):
    '''
    Basset - Kelley et al Genome Research 2016
    '''
    config['width_factor'] = config.get('width_factor', 1) # extra config to widen model (NOT deepen)
    config['temporal'] = config.get('temporal', False)
    config['final_pool'] = config.get('final_pool', 'flatten')
    config['fc_layers'] = config.get('fc_layers', 2)
    config['fc_dim'] = config.get('fc_dim', int(config['width_factor']*1000))
    config['drop'] = config.get('drop', 0.3)

    net = basset_conv_factorized_module(features, is_training, width_factor=config['width_factor'])
    net = final_pool(net, config['final_pool'])
    if config['temporal']:
        logits = temporal_pred_module(
            net,
            int(labels.get_shape()[-1]),
            share_logistic_weights=True,
            is_training=is_training)
    else:
        logits = mlp_module(
            net, 
            num_tasks = int(labels.get_shape()[-1]), 
            fc_dim = config['fc_dim'], 
            fc_layers = config['fc_layers'],
            dropout=config['drop'],
            is_training=is_training)
    
    # Torch7 style maxnorm
    maxnorm(norm_val=7)

    return logits


def lstm_module(features, is_training):
    """LSTM module, can pop on top of CNN
    """
    # lstm
    net = tf.squeeze(features, axis=1) #remove extra dim that was added so we could use conv2d. Results in batchXtimeXdepth
    rnn_inputs = tf.unstack(net, axis=1, name='unpack_time_dim')
    
    cell_fw = tf.contrib.rnn.LSTMBlockCell(320)
    cell_bw = tf.contrib.rnn.LSTMBlockCell(320)
    outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
        cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
    net = tf.concat([state_fw[1], state_bw[1]], axis=1)
    net = slim.dropout(net, keep_prob=0.5, is_training=is_training)

    return net


def basset_conv_module(features, is_training=True, width_factor=1):
    """basset convolutional layers
    """
    with slim.arg_scope(
            [slim.batch_norm],
            center=True,
            scale=True,
            activation_fn=tf.nn.relu,
            is_training=is_training):
        with slim.arg_scope(
                [slim.conv2d],
                activation_fn=None,
                weights_initializer=layers.variance_scaling_initializer(),
                biases_initializer=None):
            net = slim.conv2d(features, int(width_factor*300), [1, 19], scope="Conv")
            # need to do this for tf_deeplift
            tf.add_to_collection("DEEPLIFT_ACTIVATIONS", net)
            net = slim.batch_norm(net, scope="BatchNorm")
            tf.add_to_collection("DEEPLIFT_ACTIVATIONS", net)
            net = slim.max_pool2d(net, [1, 3], stride=[1, 3])

            net = slim.conv2d(net, int(width_factor*200), [1, 11], scope="Conv_1")
            tf.add_to_collection("DEEPLIFT_ACTIVATIONS", net)
            net = slim.batch_norm(net, scope="BatchNorm_1")
            tf.add_to_collection("DEEPLIFT_ACTIVATIONS", net)
            net = slim.max_pool2d(net, [1, 4], stride=[1, 4])

            net = slim.conv2d(net, int(width_factor*200), [1, 7], scope="Conv_2")
            tf.add_to_collection("DEEPLIFT_ACTIVATIONS", net)
            net = slim.batch_norm(net, scope="BatchNorm_2")
            tf.add_to_collection("DEEPLIFT_ACTIVATIONS", net)
            net = slim.max_pool2d(net, [1, 4], stride=[1, 4])
            
    return net


def basset_old(features, labels, config, is_training=True, variable_scope="basset"):
    """Basset - Kelley et al Genome Research 2016
    """
    config['width_factor'] = config.get('width_factor', 1) # extra config to widen model (NOT deepen)
    config["recurrent"] = config.get("recurrent", False)
    config['temporal'] = config.get('temporal', False)
    config['final_pool'] = config.get('final_pool', 'flatten')
    config['fc_layers'] = config.get('fc_layers', 2)
    config['fc_dim'] = config.get('fc_dim', int(config['width_factor']*1000))
    config['drop'] = config.get('drop', 0.3)
    config["logit_drop"] = config.get("logit_drop", 0.0)
    config["split_before"] = config.get("split_before", None)

    with tf.variable_scope(variable_scope):
        # convolutional layers
        net = basset_conv_module(
            features,
            is_training,
            width_factor=config['width_factor'])

        # recurrent layers (if any)
        if config["recurrent"]:
            net = lstm_module(net, is_training)
        else:
            net = final_pool(net, config['final_pool'])

        # mlp to logits
        if config['temporal']:
            logits = temporal_pred_module(
                net,
                int(labels.get_shape()[-1]),
                share_logistic_weights=True,
                is_training=is_training)
        else:
            logits = mlp_module_v2(
                net, 
                num_tasks = int(labels.get_shape()[-1]), 
                fc_dim = config['fc_dim'], 
                fc_layers = config['fc_layers'],
                dropout=config['drop'],
                logit_dropout=config["logit_drop"],
                split_before=config["split_before"],
                is_training=is_training)

        # Torch7 style maxnorm
        maxnorm(norm_val=7)

    return logits


def basset(inputs, params):
    """Basset - Kelley et al Genome Research 2016
    """
    # set up the needed inputs
    assert inputs.get("features") is not None
    outputs = dict(inputs)

    # features
    features_key = params["features_key"]
    logits_key = params["logits_key"]
    labels_key = params["labels_key"]
    features = inputs[features_key]
    labels = inputs[labels_key]
    is_training = params.get("is_training", False)
    
    # get params
    params['width_factor'] = params.get('width_factor', 1) # extra config to widen model (NOT deepen)
    params["recurrent"] = params.get("recurrent", False)
    params['temporal'] = params.get('temporal', False)
    params['final_pool'] = params.get('final_pool', 'flatten')
    params['fc_layers'] = params.get('fc_layers', 2)
    params['fc_dim'] = params.get('fc_dim', int(params['width_factor']*1000))
    params['drop'] = params.get('drop', 0.3)
    params["logit_drop"] = params.get("logit_drop", 0.0)
    params["split_before"] = params.get("split_before", None)
    
    # set up model
    with tf.variable_scope("basset"):
        # convolutional layers
        net = basset_conv_module(
            features,
            is_training,
            width_factor=params['width_factor'])

        # recurrent layers (if any)
        if params["recurrent"]:
            net = lstm_module(net, is_training)
        else:
            net = final_pool(net, params['final_pool'])

        # mlp to logits
        if params['temporal']:
            logits = temporal_pred_module(
                net,
                int(labels.get_shape()[-1]),
                share_logistic_weights=True,
                is_training=is_training)
        else:
            # TODO - here, expose the hidden layer so we can save it to cluster on it
            logits = mlp_module_v2(
                net, 
                num_tasks = int(labels.get_shape()[-1]), 
                fc_dim = params['fc_dim'], 
                fc_layers = params['fc_layers'],
                dropout=params['drop'],
                logit_dropout=params["logit_drop"],
                split_before=params["split_before"],
                is_training=is_training)

        # Torch7 style maxnorm
        maxnorm(norm_val=7)

    # store outputs
    outputs[logits_key] = logits # store logits
    
    return outputs, params



def deepsea_conv_module(features, is_training, l2_weight=0.0000005):
    """deepsea convolutional layers
    """
    with slim.arg_scope(
            [slim.conv2d],
            activation_fn=tf.nn.relu,
            weights_initializer=layers.variance_scaling_initializer(), # note that this is slim specific and needs to be updated for tf.layers
            weights_regularizer=slim.l2_regularizer(l2_weight),
            biases_regularizer=slim.l2_regularizer(l2_weight)):
        net = slim.conv2d(features, 320, [1, 8])
        net = slim.max_pool2d(net, [1, 4], stride=[1, 4])
        net = slim.dropout(net, keep_prob=0.8, is_training=is_training)

        net = slim.conv2d(features, 480, [1, 8])
        net = slim.max_pool2d(net, [1, 4], stride=[1, 4])
        net = slim.dropout(net, keep_prob=0.8, is_training=is_training)

        net = slim.conv2d(features, 960, [1, 8])
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training)

    return net


def deepsea(features, labels, config, is_training=True):
    """ DeepSEA - Zhou and Troyanskaya 2015
    """
    net = deepsea_conv_module(features, is_training)
    net = final_pool(net, "flatten")

    logits = mlp_module(
        net,
        num_tasks = int(labels.get_shape()[-1]),
        fc_dim = 925,
        fc_layers = 1,
        dropout = 0.0,
        logit_l1=0.00000001,
        is_training=is_training)
    
    maxnorm(norm_val=0.9)

    return logits


def danq(features, labels, config, is_training=True):
    """ DANQ - Quang et al 2016
    """
    net = slim.conv2d(
        features,
        320,
        kernel_size=[1,26],
        stride=[1,1],
        activation_fn=tf.nn.relu,
        padding='VALID')
    net = slim.max_pool2d(
        net,
        kernel_size=[1,13],
        stride=[1,13],
        padding='VALID')
    net = slim.dropout(
        net,
        keep_prob=0.8,
        is_training=is_training)
    
    net = tf.squeeze(net, axis=1)#remove extra dim that was added so we could use conv2d. Results in batchXtimeXdepth
    rnn_inputs = tf.unstack(net, axis=1, name='unpack_time_dim')

    cell_fw = tf.contrib.rnn.LSTMBlockCell(320)
    cell_bw = tf.contrib.rnn.LSTMBlockCell(320)
    outputs_fwbw_list, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
        cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
    net = tf.concat([state_fw[1], state_bw[1]], axis=1)
    net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
    net = slim.fully_connected(net, 925, activation_fn=tf.nn.relu)
    logits = slim.fully_connected(
        net, int(labels.get_shape()[-1]), activation_fn=None)

    return logits


def _residual_block(net_in, depth, pooling_info=(None, None), first=False):
    first_stride = 1
    depth_in = net_in.get_shape().as_list()[-1]

    if depth_in!=depth and not first:
        pooling, pooling_stride = pooling_info
        if pooling=='conv':
            shortcut = slim.avg_pool2d(
                net_in, stride=[1, pooling_stride])#downsample for both shortcut and conv branch
            net_preact = slim.batch_norm(net_in)
            first_stride = pooling_stride
        elif pooling=='max':
            net = slim.max_pool2d(
                net_in, stride=[1, pooling_stride])#downsample for both shortcut and conv branch
            shortcut = net
            net_preact = slim.batch_norm(net)
        else:
            raise Exception('unrecognized pooling: %s'%pooling_info)
    else:
        net_preact = slim.batch_norm(net_in)
        if first:
            shortcut = net_preact
        else:
            shortcut = net_in
    net = slim.conv2d(net_preact, depth, stride=[1, first_stride])
    net = slim.batch_norm(net)
    net = slim.conv2d(net, depth, stride=[1, 1])

    if depth_in != depth:
        paddings = [(0,0),
                    (0,0),
                    (0,0),
                    ((depth-depth_in)/2, int(math.ceil((depth-depth_in)/2)))]
        shortcut = tf.pad(net_preact, paddings)
    net = net + shortcut
    return net


def _resnet(
        features,
        initial_conv,
        kernel,
        stages,
        pooling_info,
        l2,
        is_training=True):
    """Resnet
    """
    print features.get_shape().as_list()
    with slim.arg_scope(
            [slim.batch_norm],
            center=True,
            scale=True,
            activation_fn=tf.nn.relu,
            is_training=is_training):
        with slim.arg_scope(
                [slim.conv2d, slim.max_pool2d],
                kernel_size=[1, kernel],
                padding='SAME'):
            with slim.arg_scope(
                    [slim.conv2d],
                    activation_fn=None,
                    weights_regularizer=slim.l2_regularizer(l2),
                    weights_initializer=layers.variance_scaling_initializer(),
                    biases_initializer=None):
                # We do not include batch normalization or activation functions in embed because the first ResNet unit will perform these.
                with tf.variable_scope('embed'):
                    initial_filters, initial_kernel, initial_stride = initial_conv
                    net = slim.conv2d(
                        features,
                        initial_filters,
                        kernel_size=[1, initial_kernel])
                    net = slim.max_pool2d(
                        net,
                        kernel_size=[1, initial_stride],
                        stride=[1, initial_stride])
                print net.get_shape().as_list()
                for i, stage in enumerate(stages):
                    with tf.variable_scope('stage%d'%i):
                        num_blocks, depth = stage
                        for j in xrange(num_blocks):
                            with tf.variable_scope('block%d'%j):
                                net = _residual_block(
                                    net,
                                    depth,
                                    pooling_info,
                                    first=(i==0 and j==0))
                                print net.get_shape().as_list()
        net = slim.batch_norm(net)
    return net


def resnet(features, labels, config, is_training=True):
    """Resnet
    """
    initial_conv = config.get('initial_conv', (16, 3, 1))
    kernel = config.get('kernel', 3)
    stages = config.get('stages', [(1, 32),(1, 64),(1, 128),(1, 256)])
    pooling_info = config.get('pooling', ('max', 2))
    final_pooling = config.get('final_pooling', 'mean')
    fc_layers, fc_dim = config.get('fc', (1, 1024))
    drop = config.get('drop', 0.0)
    l2 = config.get('l2', 0.0001)
    num_labels = int(labels.get_shape()[-1])

    net = _resnet(
        features,
        initial_conv,
        kernel,
        stages,
        pooling_info,
        l2,
        is_training)
    net = final_pool(
        net,
        final_pooling)
    logits = mlp_module(
        net,
        num_labels,
        fc_dim,
        fc_layers,
        drop,
        l2,
        is_training)
    return logits


def tfslim_inception(features, labels, config, is_training=True):
    """Wrapper around inception v3 from tf slim
    """
    num_classes = labels.get_shape()[1]
    logits, end_points = inception_v3.inception_v3(
        features,
        num_classes=num_classes,
        is_training=is_training) # note that prediction fn is technically softmax but unused
    
    return logits


def tfslim_resnet(features, labels, config, is_training=True):
    """Wrapper around resnet
    """
    num_classes = labels.get_shape()[1]
    logits, end_points = resnet_v2.resnet_v2_50(
        features,
        num_classes=num_classes,
        is_training=is_training)

    logits = tf.squeeze(logits, [1, 2])
    
    return logits


def ensemble(features, labels, config, is_training=True):
    """Given a list of models, instantiate each under an index
    need to instantiate each under an index to be able to 
    load the correct checkpoint to each model
    """
    from tronn.nets.nets import net_fns

    # get the models
    models = config.get("models")
    models = [net_fns[model_name]
              for model_name in models.split(",")]

    # set up each model
    all_logits = []
    for i in xrange(len(models)):
        new_scope = "model_{}".format(i)
        print new_scope
        with tf.variable_scope(new_scope):
            model_logits = models[i](features, labels, config, is_training=is_training)
            all_logits.append(model_logits)

    # TODO may want to put in a function to load an MLP on top of the ensemble?
    logits = tf.reduce_mean(
        tf.stack(all_logits, axis=1), axis=1)

    return logits
