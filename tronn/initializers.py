"""Contains custom initializers.

Most relevant is that torch7 initialization is not
implemented in tensorflow

"""

import math
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops

def torch_initializer(stdv, dtype=dtypes.float32):
    '''
    Torch7 style initializer. Can be used for weights and biases.
    '''
    def _initializer(shape, dtype=dtype, partition_info=None, seed=1337):
        return random_ops.random_uniform(shape, -stdv, stdv, dtype, seed=seed)
    return _initializer


def torch_conv_initializer(filter_shape, fan_in, dtype=dtypes.float32):
    '''
    Wraps the calculation of the standard dev for convolutional layers:
    stdv = 1 / sqrt( filter_width * filter_height * fan_in )
    '''
    stdv = 1. / math.sqrt(filter_shape[0] * filter_shape[1] * fan_in)
    return torch_initializer(stdv)


def torch_fullyconnected_initializer(fan_in, dtype=dtypes.float32):
    '''
    Wraps the calculation of the standard dev for fully connected layers:
    stdv = 1 / sqrt( fan_in )
    '''
    stdv = 1. / math.sqrt(fan_in)
    return torch_initializer(stdv)