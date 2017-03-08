""" Contains functions for I/O 

Currently hdf5 is supported as an important standardized filetype
used frequently in genomics datasets.

"""
import h5py
import tensorflow as tf
from tensorflow.python.framework import errors
import numpy as np
import random

# def batch_generator(h5py_handles, batch_locations, batch_size):
#     while True:
#         random.shuffle(batch_locations)
#         print 'batcher restarted'
#         for i in xrange(len(batch_locations)):
#             file_num, batch_num_in_file = batch_locations[i]
#             h5py_handle = h5py_handles[file_num]
#             batch_start = batch_num_in_file*batch_size
#             batch_end = batch_start + batch_size

#             features = h5py_handle['features'][batch_start:batch_end,:,:,:]
#             labels = h5py_handle['labels'][batch_start:batch_end,:]
#             metadata = h5py_handle['regions'][batch_start:batch_end].reshape((batch_size, 1))
#             yield [features, labels, metadata]

# def load_data_from_filename_list(hdf5_files, batch_size):
#     batch_locations = []
#     h5py_handles = []
#     for file_num, hdf5_file in enumerate(hdf5_files):
#         h5py_handle = h5py.File(hdf5_file)
#         num_examples = h5py_handle['features'].shape[0]
#         num_batches = num_examples/batch_size
#         batch_locations.extend(zip([file_num]*num_batches, range(num_batches)))
#         h5py_handles.append(h5py_handle)

#     next_batch_fn = batch_generator(h5py_handles, batch_locations, batch_size).next

#     [features_tensor, labels_tensor, metadata_tensor] = tf.py_func(func=next_batch_fn,
#         inp=[],
#         Tout=[tf.float32, tf.float32, tf.string],
#         stateful=True, name='py_func_batch_generator_next')

#     # Check shapes from the hdf5 file and set the tensor shapes
#     feature_shape = h5py_handles[0]['features'].shape[1:]
#     label_shape = h5py_handles[0]['labels'].shape[1:]
#     features_tensor.set_shape([batch_size, feature_shape[0], feature_shape[1], feature_shape[2]])
#     labels_tensor.set_shape([batch_size, label_shape[0]])
#     metadata_tensor.set_shape([batch_size, 1])

#     min_after_dequeue=10000
#     features_batch, labels_batch, metadata_batch = tf.train.shuffle_batch([features_tensor, labels_tensor, metadata_tensor], batch_size=batch_size, capacity=5*min_after_dequeue, min_after_dequeue=min_after_dequeue, enqueue_many=True, name='batcher')

#     return features_batch, labels_batch, metadata_batch



def hdf5_to_slices(hdf5_file, batch_size, days):
    h5py_handle = h5py.File(hdf5_file)
    num_examples = h5py_handle['features'].shape[0]
    max_batches = num_examples/batch_size
    batch_id_queue = tf.train.range_input_producer(max_batches, shuffle=True)

    # Check shapes from the hdf5 file so that we can set the tensor shapes
    feature_shape = h5py_handle['features'].shape[1:]
    label_shape = h5py_handle['labels'].shape[1:]

    # Extract examples based on batch_id
    def batchid_to_examples(batch_id):
        batch_start = batch_id*batch_size
        batch_end = batch_start + batch_size
        features = h5py_handle['features'][batch_start:batch_end]
        labels_all_days = h5py_handle['labels'][batch_start:batch_end]
        labels_days = labels_all_days[:, days]
        metadata = h5py_handle['regions'][batch_start:batch_end].reshape((batch_size, 1))
        return [features, labels_days, metadata]


    batch_id_tensor = batch_id_queue.dequeue()
    [features_tensor, labels_tensor, metadata_tensor] = tf.py_func(func=batchid_to_examples,
        inp=[batch_id_tensor],
        Tout=[tf.float32, tf.float32, tf.string],
        stateful=False, name='py_func_batchid_to_examples')

    features_tensor.set_shape([batch_size, feature_shape[0], feature_shape[1], feature_shape[2]])
    labels_tensor.set_shape([batch_size, len(days)])
    metadata_tensor.set_shape([batch_size, 1])

    return features_tensor, labels_tensor, metadata_tensor

def load_data_from_filename_list(hdf5_files, batch_size, days=[0,1,2,3,4,5,6,7,8,9,10,11,12], shuffle_seed=0):
    example_slices_list = [hdf5_to_slices(hdf5_file, batch_size, days) for hdf5_file in hdf5_files]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + (len(example_slices_list)+10) * batch_size
    features, labels, metadata = tf.train.shuffle_batch_join(example_slices_list, batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue, seed=shuffle_seed, enqueue_many=True, name='batcher')
    return features, labels, metadata
