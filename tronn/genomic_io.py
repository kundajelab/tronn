# run Basset

import os
import math
import layers
import models
import tensorflow as tf
import numpy as np
import threading
import h5py

from sklearn import metrics as skmetrics
from random import shuffle


def get_data_params(hdf5_file):
    '''
    Get basic parameters
    '''

    with h5py.File(hdf5_file, 'r') as hf:
        num_train_examples = hf['train_in'].shape[0]
        seq_length = hf['train_in'].shape[2]
        num_tasks = hf['train_out'].shape[1]

    return num_train_examples, seq_length, num_tasks


def data_loader_simple(batch_size, seq_length, tasks, data_splits=['train', 'valid', 'test']):
    '''
    Set up data queues and conditionals to feed examples and labels
    hacky set up to test tronn
    '''
    
    queues, inputs, enqueue_ops, dequeue, case_dict = {}, {}, {}, {}, {}
    conditional_fns = []
    
    with tf.name_scope('inputs') as scope:

        # Set up data loading for each split. Note that the first split becomes default in the conditional.
        for split_idx in range(len(data_splits)):
            split = data_splits[split_idx]
            queues[split] = tf.FIFOQueue(10000, [tf.float32, tf.float32, tf.string], shapes=[[1, seq_length, 4], [tasks], [1]])
            inputs[split] = { 'features': tf.placeholder(tf.float32, shape=[batch_size, 1, seq_length, 4]),
                              'labels' : tf.placeholder(tf.float32, shape=[batch_size, tasks]),
                              'metadata': tf.placeholder(tf.string, shape=[batch_size, 1]) } 
            enqueue_ops[split] = queues[split].enqueue_many([inputs[split]['features'],
                                                             inputs[split]['labels'],
                                                             inputs[split]['metadata']])
            dequeue[split] = queues[split].dequeue_many(batch_size)

        [feature_batch, label_batch, metadata_batch] = dequeue['train']

    return queues, inputs, enqueue_ops, feature_batch, label_batch, metadata_batch


def data_loader(batch_size, seq_length, tasks, model_state, data_splits=['train', 'valid', 'test']):
    '''
    Set up data queues and conditionals to feed examples and labels
    '''
    
    queues, inputs, enqueue_ops, dequeue, case_dict = {}, {}, {}, {}, {}
    conditional_fns = []
    
    with tf.name_scope('inputs') as scope:

        # Set up data loading for each split. Note that the first split becomes default in the conditional.
        for split_idx in range(len(data_splits)):
            split = data_splits[split_idx]
            queues[split] = tf.FIFOQueue(10000, [tf.float32, tf.float32, tf.string], shapes=[[1, seq_length, 4], [tasks], [1]])
            inputs[split] = { 'features': tf.placeholder(tf.float32, shape=[batch_size, 1, seq_length, 4]),
                              'labels' : tf.placeholder(tf.float32, shape=[batch_size, tasks]),
                              'metadata': tf.placeholder(tf.string, shape=[batch_size, 1]) } 
            enqueue_ops[split] = queues[split].enqueue_many([inputs[split]['features'],
                                                             inputs[split]['labels'],
                                                             inputs[split]['metadata']])
            dequeue[split] = queues[split].dequeue_many(batch_size)
            #case_dict[tf.equal(model_state, split)] = lambda: dequeue[split]

            # Setting up this weird stacked conditionals because tf.case is buggy and does not work properly yet.
            if split_idx == 0:
                conditional_fns.append(tf.cond(tf.equal(model_state, split),
                                               lambda: dequeue[split],
                                               lambda: dequeue[split]))
            else:
                conditional_fns.append(tf.cond(tf.equal(model_state, split),
                                               lambda: dequeue[split],
                                               lambda: conditional_fns[split_idx - 1]))

        [feature_batch, label_batch, metadata_batch] = conditional_fns[-1] 

    return queues, inputs, enqueue_ops, feature_batch, label_batch, metadata_batch


def load_and_enqueue(sess, coord, hdf5_file, batch_size, inputs, enqueue_ops, order='NHWC', data_splits=['train', 'valid', 'test']): 
    '''
    This function loads data from the hdf5 files into the feed dicts, which then go into the queues.
    '''
    with h5py.File(hdf5_file, 'r') as hf:

        h5_data = {}
        for split in data_splits:
            h5_data[split] = {
                'in': hf.get('{}_in'.format(split)),
                'out': hf.get('{}_out'.format(split)),
                'metadata': hf.get('{}_regions'.format(split))
            }
            h5_data[split]['num_examples'] = h5_data[split]['in'].shape[0]
            h5_data[split]['batch_start'] = 0
            h5_data[split]['batch_end'] = batch_size

        with coord.stop_on_exception():
            while not coord.should_stop():
                # Go through the splits and load data
                for split in data_splits:
                    batch_start = h5_data[split]['batch_start']
                    batch_end = h5_data[split]['batch_end']
                    if order == 'NHWC':
                        feature_array = h5_data[split]['in'][batch_start:batch_end,:,:,:]                        
                    elif order == 'NCHW':
                        feature_array = np.rollaxis(h5_data[split]['in'][batch_start:batch_end,:,:,:], 1, 4)
                    else:
                        print "Axis order not specified!!"
                    labels_array = h5_data[split]['out'][batch_start:batch_end,:]
                    metadata_array = h5_data[split]['metadata'][batch_start:batch_end].reshape((batch_size, 1))

                    sess.run(enqueue_ops[split], feed_dict={inputs[split]['features']: feature_array,
                                                            inputs[split]['labels']: labels_array,
                                                            inputs[split]['metadata']: metadata_array}) 

                    if batch_end + batch_size > h5_data[split]['num_examples']:
                        h5_data[split]['batch_start'] = 0
                        h5_data[split]['batch_end'] = batch_size
                    else:
                        h5_data[split]['batch_start'] = batch_end
                        h5_data[split]['batch_end'] = batch_end + batch_size

    return None

