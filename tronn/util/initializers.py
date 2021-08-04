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
        reverse_complement=False,
        max_centered=False,
        unit_vector=True,
        length_norm=True,
        dtype=dtypes.float32):
    """Load just PWMs into layer
    """
    # calc params
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
            if max_centered:
                # center on max position
                filter_length = 2*filter_shape[1]
                left_pad = math.ceil(filter_length / 2.)
                right_pad = math.floor(filter_length / 2.)
                max_position = np.argmax(np.max(pwm.weights, axis=0))

                left_extend = int(left_pad) - max_position
                right_extend = int(right_pad) - (pwm.weights.shape[1] - max_position)
                padded_weights = np.concatenate(
                    (np.zeros(
                        (4, left_extend)),
                     pwm.weights,
                     np.zeros(
                         (4, right_extend))),
                    axis=1)
            else:
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

        if unit_vector:
            # adjust to convert to unit vector
            padded_weights = np.divide(padded_weights, np.sqrt(np.sum(np.square(padded_weights))))
            
        if length_norm:
            # adjust for length of pwm
            nonzero_fraction = pwm.weights.shape[1] / float(padded_weights.shape[1])
            padded_weights = np.multiply(padded_weights, nonzero_fraction)
            
        # append
        weights_list.append(padded_weights)

    # if reverse complement, go through list and attach on the flipped version
    if reverse_complement:
        reversed_weights_list = []
        for weights in weights_list:
            rc = np.fliplr(np.flipud(weights))
            reversed_weights_list.append(rc)
        weights_list += reversed_weights_list
            
    # stack into weights tensor and assign to subset
    pwm_all_weights = np.stack(weights_list, axis=0).transpose(2, 1, 0)
    pwm_weights_final = np.expand_dims(pwm_all_weights, axis=0)

    def _initializer(shape, dtype=dtype, partition_info=None):
        return pwm_weights_final

    return _initializer
