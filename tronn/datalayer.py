""" Contains functions for I/O 

Currently hdf5 is supported as an important standardized filetype
used frequently in genomics datasets.

"""

import tensorflow as tf
import numpy as np
import h5py
import collections


def check_dataset_params(hdf5_file_list):
    '''
    Gathers basic information
    '''

    num_examples = 0
    for filename in hdf5_file_list:
        with h5py.File(filename,'r') as hf:
            num_examples += hf['features'].shape[0]
            seq_length = hf['features'].shape[2]
            num_tasks = hf['labels'].shape[1]

    return num_examples, seq_length, num_tasks


# def setup_queue(features, labels, seq_length, tasks, 
#     capacity=10000):
#     '''
#     Set up data queue as well as queue runner.
#     '''

#     with tf.variable_scope('datalayer'):
#         queue = tf.FIFOQueue(capacity,
#                              [tf.float32, tf.float32],
#                              shapes=[[1, seq_length, 4],
#                                      [tasks]])
#         enqueue_op = queue.enqueue_many([features, labels])
#         queue_runner = tf.train.QueueRunner(
#             queue=queue,
#             enqueue_ops=[enqueue_op],
#             close_op=queue.close(),
#             cancel_op=queue.close(cancel_pending_enqueues=True))
#         tf.train.add_queue_runner(queue_runner, tf.GraphKeys.QUEUE_RUNNERS)

#     return queue

def setup_queue(features, labels, capacity=10000):
    '''
    Set up data queue as well as queue runner.
    '''

    with tf.variable_scope('datalayer'):
        queue = tf.FIFOQueue(capacity,
                             [tf.float32, tf.float32],
                             shapes=[features.get_shape()[1:],
                                     labels.get_shape()[1:]])
        enqueue_op = queue.enqueue_many([features, labels])
        queue_runner = tf.train.QueueRunner(
            queue=queue,
            enqueue_ops=[enqueue_op],
            close_op=queue.close(),
            cancel_op=queue.close(cancel_pending_enqueues=True))
        tf.train.add_queue_runner(queue_runner, tf.GraphKeys.QUEUE_RUNNERS)

    return queue


def get_hdf5_list_reader_pyfunc(hdf5_files, batch_size):
    '''
    Takes in a list of hdf5 files and generates a function that returns a group
    of examples and labels
    '''

    h5py_handlers = [ h5py.File(filename) for filename in hdf5_files ]

    # TODO: at this point probably want to check shapes and then after 
    # function call set shape
    feature_shape = h5py_handlers[0]['features'].shape[1:]
    label_shape = h5py_handlers[0]['labels'].shape[1:]

    def hdf5_reader_fn():
        '''
        Given batch start and stop, pulls those examples from hdf5 file
        '''
        global batch_start
        global batch_end
        global filename_index

        # check if at end of file, and move on to the next file
        # todo: allow adding nulls or something (the queue size is causing the problem)
        if batch_end > h5py_handlers[filename_index]['features'].shape[0]:
            print hdf5_files[filename_index]
            filename_index += 1
            batch_start = 0
            batch_end = batch_size

        if filename_index >= len(h5py_handlers):
            filename_index = 0
            batch_start = 0
            batch_end = batch_size

        # TODO(dk) need to add some asserts to prevent running over the end

        features = h5py_handlers[filename_index]['features'][batch_start:batch_end,:,:,:]
        labels = h5py_handlers[filename_index]['labels'][batch_start:batch_end,:]

        batch_start += batch_size
        batch_end += batch_size

        return [features, labels]

    [py_func_features, py_func_labels] = tf.py_func(hdf5_reader_fn, [], [tf.float32, tf.float32],
                      stateful=True)

    # Set the shape so that we can infer sizes etc in later layers.
    py_func_features.set_shape([batch_size, feature_shape[0], feature_shape[1], feature_shape[2]])
    py_func_labels.set_shape([batch_size, label_shape[0]])

    return py_func_features, py_func_labels


def load_data_from_filename_list(hdf5_files, batch_size):
    '''
    Put it all together
    '''

    global batch_start 
    batch_start = 0
    global batch_end 
    batch_end = batch_size
    global filename_index
    filename_index = 0

    [hdf5_features, hdf5_labels] = get_hdf5_list_reader_pyfunc(hdf5_files,
                                                          batch_size)

    # queue = setup_queue(hdf5_features, hdf5_labels,
    #                     seq_length, tasks)

    queue = setup_queue(hdf5_features, hdf5_labels)

    [features, labels] = queue.dequeue_many(batch_size)

    return features, labels
