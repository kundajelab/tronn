"""Contains custom layers.

New layers as needed should be defined here.

"""

import tensorflow as tf


def threshold(input_tensor, th, val, name):
    '''
    Sets up a threshold. Needed for deepSEA and torch7 models that use
    the threshold function instead of the ReLU function
    '''
    with tf.variable_scope(name) as scope:

        # First figure out where values are less than
        lessthan_tensor = tf.cast(tf.lesser(input_tensor, th), tf.int32)
        greaterthan_tensor = tf.cast(tf.greater(input_tensor, th), tf.int32)

        # Then make thresholded values equal to val and the rest are the same
        out_tensor = input_tensor * greaterthan_tensor + val * lessthan_tensor

    return out_tensor
