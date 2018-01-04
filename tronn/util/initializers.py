"""Description: Contains custom initializers.
"""

import math
import numpy as np
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


def pwm_simple_initializer(
        filter_shape,
        pwm_list,
        fan_in,
        length_norm=True,
        squared=False,
        binarize=False,
        dtype=dtypes.float32):
    """Load just PWMs into layer
    """
    filter_length = filter_shape[1]
    left_pad = math.ceil(filter_length / 2.)
    right_pad = math.floor(filter_length / 2.)
    num_pwms = len(pwm_list)

    # for each PWM
    weights_list = []
    for i in range(num_pwms):
        pwm = pwm_list[i]
        extend_length = int((filter_length - pwm.weights.shape[1]) / 2)
        
        if extend_length >= 0:
            # centered weight
            padded_weights = np.concatenate(
                (np.zeros(
                    (4, extend_length)),
                 pwm.weights,
                 np.zeros(
                     (4, filter_length-extend_length-pwm.weights.shape[1]))),
                axis=1)
        else:
            pwm_center = pwm.weights.shape[1] / 2
            padded_weights = pwm.weights[:,pwm_center-left_pad:pwm_center+right_pad]

        if length_norm:
            num_nonzero_basepairs = np.sum((np.sum(padded_weights, axis=0) != 0)) # is this right?
            #print num_nonzero_basepairs
            padded_weights = np.divide(padded_weights, num_nonzero_basepairs)

            # also need to normalize so that max score is 1 on flat sequence?

            # different normalization - normalize so that total info is approx 1, normalized to max base pair?
            
            
        # This bit is used for vector projection - mostly keep FALSE
        if squared:
            padded_weights = np.square(padded_weights)

        # use this if just want hits in specific directions
        if binarize:
            pass
            
        weights_list.append(padded_weights)

    # stack into weights tensor and assign to subset
    pwm_all_weights = np.stack(weights_list, axis=0).transpose(2, 1, 0)
    pwm_weights_final = np.expand_dims(pwm_all_weights, axis=0)

    def _initializer(shape, dtype=dtype, partition_info=None):
        return pwm_weights_final

    return _initializer
        

    
    


def pwm_initializer(filter_shape, pwm_list, fan_in, total_filters=900, flip=False, dtype=dtypes.float32):
    '''
    Load PWMs into layer
    '''

    filter_length = filter_shape[1]
    
    # for each PWM,
    weights_list = []
    for i in range(len(pwm_list)):
        pwm = pwm_list[i]
        
        extend_length = int((filter_length - pwm.weights.shape[1]) / 2)
        
        if extend_length >= 0:
            # centered weight
            padded_weights = np.concatenate((np.zeros((4, extend_length)), pwm.weights, np.zeros((4, 19-extend_length-pwm.weights.shape[1]))), axis=1)
        else:
            pwm_center = pwm.weights.shape[1] / 2
            padded_weights = pwm.weights[:,pwm_center-10:pwm_center+9]
        weights_list.append(padded_weights) # do it twice to double chance of it succeeding
        if flip:
            weights_list.append(np.flipud(np.fliplr(padded_weights)))
            
    # stack into weights tensor and assign to subset
    pwm_all_weights = np.stack(weights_list, axis=0).transpose(2, 1, 0)
    pwm_np_array_subset = np.expand_dims(pwm_all_weights, axis=0)

    complementary_shape = list(pwm_np_array_subset.shape)
    complementary_shape[3] = total_filters - pwm_np_array_subset.shape[3]
            
    # conv initializer stdv
    stdv = 1. / math.sqrt(filter_shape[0] * filter_shape[1] * fan_in)
            
    def _initializer(shape, dtype=dtype, partition_info=None, seed=1337):
        return tf.concat([pwm_np_array_subset, random_ops.random_uniform(complementary_shape, -stdv, stdv, dtype, seed=seed)], axis=3)

    return _initializer
