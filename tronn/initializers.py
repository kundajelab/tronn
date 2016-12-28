"""Contains custom initializers.

Most relevant is that torch7 initialization is not
implemented in tensorflow

"""

import math
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops


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