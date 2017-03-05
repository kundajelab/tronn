"""Contains various utility functions
"""

import math
import h5py
import tensorflow as tf


def get_total_num_examples(hdf5_file_list):
    '''
    Quickly extracts total examples represented in an hdf5 file list. Can 
    be used to calculate total steps to take (when 1 step represents going 
    through a batch of examples)
    '''
    
    num_examples = 0
    for filename in hdf5_file_list:
        with h5py.File(filename,'r') as hf:
            num_examples += hf['features'].shape[0]

    return num_examples


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