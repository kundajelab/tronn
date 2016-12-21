import tensorflow as tf
import math

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.training import moving_averages

"""
Neural net layer wrapper functions.

All layers follow the API: layer_name(input_tensor, *params); returns output tensor

The dimension ordering used throughout is [batch, height, width, channel] (NHWC in caffe-speak).

Valid padding mode is assumed in all conv layers, which does not exceed input dimensions.
"""


def slim_conv_bias_initializer(stdv, dtype=dtypes.float32):
    """hacky initializer"""
    def _initializer(shape, dtype=dtype, partition_info=None, seed=1234):
        return random_ops.random_uniform(shape, -stdv, stdv, dtype, seed=seed)
    return _initializer

# ========================

def conv_weight_initializer(dtype=dtypes.float32):
    """hacky initializer for torch style, conv layers!!"""
    def _initializer(shape, dtype=dtype, partition_info=None, seed=1234):
        stdv = 1. / math.sqrt(shape[1] * shape[0] * shape[2]) # width of filter, height of filter, input num channels
        return random_ops.random_uniform(shape, -stdv, stdv, dtype, seed=seed)
    return _initializer

def conv_bias_initializer(dtype=dtypes.float32): # Give it the full param shape (so that it can calculate stdv) but then output the right shape
    """Hacky initializer to implement torch style weights for the biases"""
    def _initializer(shape, dtype=dtype, partition_info=None, seed=1234):
        stdv = 1. / math.sqrt(shape[1] * shape[0] * shape[2]) # width of filter, height of filter, input num channels
        return random_ops.random_uniform([shape[3]], -stdv, stdv, dtype, seed=seed)
    return _initializer

def fc_weight_initializer(dtype=dtypes.float32):
    """hacky initializer for torch style, conv layers!!"""
    def _initializer(shape, dtype=dtype, partition_info=None, seed=1234):
        stdv = 1. / math.sqrt(shape[0]) # input size
        return random_ops.random_uniform(shape, -stdv, stdv, dtype, seed=seed)
    return _initializer

def fc_bias_initializer(dtype=dtypes.float32):
    """hacky initializer for torch style, conv layers!!"""
    def _initializer(shape, dtype=dtype, partition_info=None, seed=1234):
        stdv = 1. / math.sqrt(shape[0]) # input size
        return random_ops.random_uniform([shape[1]], -stdv, stdv, dtype, seed=seed)
    return _initializer

def uniform_initializer(dtype=dtypes.float32):
    """Quick initializer for uniform distribution"""
    def _initializer(shape, dtype=dtype, partition_info=None, seed=1234):
        return random_ops.random_uniform(shape, 0, 1, dtype)
    
    return _initializer


def make_var(shape, initializer, name='param', trainable=True):
    """Make a variable with the given name and initializer."""
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def batchnorm(input_tensor, name, model_state, convolutional=False, decay=0.9):
    """Batch normalization extension to include learnable variables and packed all nicely."""
    with tf.variable_scope(name) as scope:

        # Initialize batch norm
        channels = input_tensor.get_shape()[-1].value
        if convolutional:
            batch_mean, batch_var = tf.nn.moments(input_tensor, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
        scale_gamma = make_var([channels], uniform_initializer(), 'gamma')
        offset_beta = make_var([channels], tf.constant_initializer(0.0), 'beta')

        # Set up moving mean and variance for proper batch norm implementation
        moving_mean = tf.get_variable('moving_mean', batch_mean.get_shape(), initializer=tf.constant_initializer(0.0))
        moving_variance = tf.get_variable('moving_variance', batch_var.get_shape(), initializer=tf.constant_initializer(1.0))
        moving_mean_update_op = moving_averages.assign_moving_average(moving_mean, batch_mean, decay)
        moving_variance_update_op = moving_averages.assign_moving_average(moving_variance, batch_var, decay),

        # Conditional choice depending on whether training or not
        batchnormed_tensor = tf.cond(tf.equal(model_state, 'train'),
                                     lambda: tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, offset_beta, scale_gamma, 1e-5),
                                     lambda: tf.nn.batch_normalization(input_tensor, moving_mean, moving_variance, offset_beta, scale_gamma, 1e-5))
    return batchnormed_tensor, [moving_mean_update_op, moving_variance_update_op]


def conv2d(input_tensor, filter_dims, stride_dims, out_channels, name, model_state, biased=True, relu=True):
    """A 2-D conv layer, with optional bias and relu."""
    filter_height, filter_width = filter_dims
    stride_height, stride_width = stride_dims
    with tf.variable_scope(name) as scope:
        in_channels = input_tensor.get_shape()[-1].value
        conv_filter_shape = [filter_height,
                             filter_width, in_channels, out_channels]
        filter_param = make_var(
            conv_filter_shape, conv_weight_initializer(), 'filter') # TO DO: this is where to initialize with PWM
        # maxnorm here, and then give this as an output so that i can use as a handle to ONLY update this after the train step
        filter_param_norm = tf.cond(tf.equal(model_state, 'maxnorm'),
                                    lambda: tf.clip_by_norm(filter_param, 7, axes=[0,1,2],  name='filter_maxnorm'),
                                    lambda: tf.identity(filter_param))
        strides = [1, stride_height, stride_width, 1]
        conv_output = tf.nn.conv2d(
            input_tensor, filter_param_norm, strides, 'VALID', name=name)
            #input_tensor, filter_param, strides, 'VALID', name=name)
        regularizers = tf.nn.l2_loss(filter_param)
        if biased:
            biases = make_var(
                conv_filter_shape, conv_bias_initializer(), 'biases') 
            conv_output = tf.nn.bias_add(conv_output, biases)
            regularizers += tf.nn.l2_loss(biases) # remove regularizers
        if relu:
            conv_output = tf.nn.relu(conv_output, name=scope.name)

        # Collections
        #tf.add_to_collection('weight_collection', filter_param_norm)
        #tf.add_to_collection('bias_collection', biases)
    return conv_output, regularizers, filter_param_norm


def conv1d(input_tensor, filter_width, stride_width, out_channels, name, model_state, biased=True, relu=True):
    """A 1-D conv layer with filter height and stride height = input height. Optional bias and relu."""
    filter_height = input_tensor.get_shape()[1].value
    stride_height = filter_height
    filter_dims = [filter_height, filter_width]
    stride_dims = [stride_height, stride_width]
    return conv2d(input_tensor, filter_dims, stride_dims, out_channels, name, model_state, biased, relu)


def maxpool(input_tensor, filter_dims, stride_dims, name):
    """Maxpool layer."""
    k_height, k_width = filter_dims
    s_height, s_width = stride_dims
    with tf.variable_scope(name) as scope:
        return tf.nn.max_pool(input_tensor, ksize=[1, k_height, k_width, 1],
                              strides=[1, s_height, s_width, 1], padding='VALID', name=scope.name)


def maxout(input_tensor, name):
    """Maxout layer; performs a maxout over the channel (last) dimension."""
    with tf.variable_scope(name) as scope:
        dim_to_max = len(input_tensor.get_shape()) - 1
        return tf.reduce_max(input_tensor, [dim_to_max], keep_dims=True, name=scope.name)


def sliced_maxout(input_tensor, dim_to_slice, num_slices, name):
    """Maxout using slices; equivalent to using a reshape and then maxout."""
    with tf.variable_scope(name) as scope:
        input_shape = [d.value for d in input_tensor.get_shape()]
        new_shape = input_shape[:dim_to_slice] + [num_slices, -1]
        reshaped = tf.reshape(input_tensor, new_shape, name='reshaped')
        return tf.reduce_max(reshaped, [dim_to_slice], keep_dims=False, name=scope.name)


def dropout(input_tensor, drop_p, name, model_state):
    """Dropout layer."""
    with tf.variable_scope(name) as scope:
        return tf.cond(tf.equal(model_state, 'train'),
                       lambda: tf.nn.dropout(input_tensor, drop_p),
                       lambda: input_tensor)


def avgpool(input_tensor, filter_dims, stride_dims, name):
    """Average pooling layer; pools over height/width dimension."""
    filter_dims = [1] + filter_dims + [1]
    stride_dims = [1] + stride_dims + [1]
    with tf.variable_scope(name) as scope:
        return tf.nn.avg_pool(input_tensor, filter_dims, stride_dims,
                              padding='VALID', name=scope.name)


def flatten(input_tensor, name):
    """Flatten layer."""
    with tf.variable_scope(name) as scope:
        output_shape = [input_tensor.get_shape()[0].value, -1]
        flat = tf.reshape(input_tensor, output_shape, name=scope.name)
    return flat


def concat(input_tensors, concat_dim, name):
    """Concat layer."""
    with tf.variable_scope(name) as scope:
        concat = tf.concat(concat_dim, input_tensors, name=scope.name)
    return concat


def inner_prod(input_tensor, output_dim, name, model_state):
    """Inner product layer; currently only supports 2-D tensors."""
    with tf.variable_scope(name) as scope:
        input_shape = input_tensor.get_shape()
        assert(len(input_shape) == 2)
        param_shape = input_shape[-1].value
        param = make_var([param_shape, output_dim],
                         fc_weight_initializer())
        # max norm here
        param_norm = tf.cond(tf.equal(model_state, 'maxnorm'),
                             lambda: tf.clip_by_norm(param, 7, axes=[0],  name='filter_maxnorm'),
                             lambda: tf.identity(param))
        prod = tf.matmul(input_tensor, param_norm, name=scope.name) 
        #prod = tf.matmul(input_tensor, param, name=scope.name) 
        # Add biases
        biases = make_var(
            #[output_dim], tf.constant_initializer(0.0), 'biases') # glorot uniform
            [param_shape, output_dim], fc_bias_initializer(), 'biases') # glorot uniform
        prod_output = tf.nn.bias_add(prod, biases)
        regularization = tf.nn.l2_loss(param) + tf.nn.l2_loss(biases)
    return prod_output, regularization, param_norm

# -----------------------------------------
# Builds for deepSEA
# -----------------------------------------

def threshold(input_tensor, th, val, name):
    '''
    Sets up a threshold
    '''
    with tf.variable_scope(name) as scope:

        # First figure out where values are less than
        lessthan_tensor = tf.cast(tf.lesser(input_tensor, th), tf.int32)
        greaterthan_tensor = tf.cast(tf.greater(input_tensor, th), tf.int32)

        # Then make thresholded values equal to val and the rest are the same
        out_tensor = input_tensor * greaterthan_tensor + val * lessthan_tensor

    return out_tensor
