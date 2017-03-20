"""Contains various utility functions
"""
import tensorflow as tf

def get_fan_in(tensor, type='NHWC'):
    '''
    Get the fan in (number of in channels)
    '''

    return int(tensor.get_shape()[-1])

def add_var_summaries(var):
    with tf.name_scope('summaries'):
        with tf.name_scope(filter(str.isalnum, str(var.name))):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)