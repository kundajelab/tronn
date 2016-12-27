""" Contains functions for I/O 

Currently hdf5 is supported as an important standardized filetype
used frequently in genomics datasets.

"""


import h5py
import tensorflow as tf
import numpy as np


def setup_queue(features, labels, capacity=10000):
    '''
    Set up data queue as well as queue runner. The shapes of the
    tensors are inferred from the inputs, so input shapes must be
    set before this function is called.
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
    Takes in a list of hdf5 files and generates a tensorflow op that returns a 
    group of examples and labels when called in the graph. Be aware that this
    setup uses global variables that must be initialized first to make this
    work.
    '''

    # Get all file handles before starting learning.
    h5py_handles = [ h5py.File(filename) for filename in hdf5_files ]

    # Check shapes from the hdf5 file so that we can set the tensor shapes
    feature_shape = h5py_handles[0]['features'].shape[1:]
    label_shape = h5py_handles[0]['labels'].shape[1:]

    def hdf5_reader_fn():
        '''
        Given batch start and stop, pulls those examples from hdf5 file
        '''
        global batch_start
        global batch_end
        global filename_index

        # check if at end of file, and move on to the next file
        if batch_end > h5py_handles[filename_index]['features'].shape[0]:
            print hdf5_files[filename_index]
            filename_index += 1
            batch_start = 0
            batch_end = batch_size

        if filename_index >= len(h5py_handles):
            filename_index = 0
            batch_start = 0
            batch_end = batch_size

        current_handle = h5py_handles[filename_index]

        features = current_handle['features'][batch_start:batch_end,:,:,:]
        labels = current_handle['labels'][batch_start:batch_end,:]

        batch_start += batch_size
        batch_end += batch_size

        return [features, labels]

    [py_func_features, py_func_labels] = tf.py_func(hdf5_reader_fn,
                                                    [],
                                                    [tf.float32, tf.float32],
                                                    stateful=True)

    # Set the shape so that we can infer sizes etc in later layers.
    py_func_features.set_shape([batch_size,
                                feature_shape[0],
                                feature_shape[1],
                                feature_shape[2]])
    py_func_labels.set_shape([batch_size, label_shape[0]])

    return py_func_features, py_func_labels


def load_data_from_filename_list(hdf5_files, batch_size):
    '''
    Put it all together
    '''

    global batch_start
    global batch_end
    global filename_index

    batch_start = 0
    batch_end = batch_size
    filename_index = 0

    [hdf5_features, hdf5_labels] = get_hdf5_list_reader_pyfunc(hdf5_files,
                                                          batch_size)

    queue = setup_queue(hdf5_features, hdf5_labels)

    [features, labels] = queue.dequeue_many(batch_size)

    return features, labels
