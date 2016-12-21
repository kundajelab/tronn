""" Contains functions for I/O 

Supported formats:

- hdf5 

"""

import tensorflow as tf
import numpy as np
import h5py
import collections


def get_hdf5_list_reader_pyfunc(hdf5_files, batch_size):
    '''
    Takes in a list of hdf5 files and generates a function that returns a group
    of examples and labels
    '''

    h5py_handlers = [ h5py.File(filename) for filename in hdf5_files ]


    def hdf5_reader_fn():
        '''
        Given batch start and stop, pulls those examples from hdf5 file
        '''
        global batch_start
        global batch_end
        global filename_index

        # check if at end of file, and move on to the next file
        if batch_end > h5py_handlers[filename_index]['features'].shape[0]:
            filename_index += 1

        if filename_index > len(h5py_handlers):
            filename_index = 0

        # TODO(dk) need to add some asserts to prevent running over the end

        features = h5py_handlers[filename_index]['features'][batch_start:batch_end,:,:,:]
        labels = h5py_handlers[filename_index]['labels'][batch_start:batch_end,:]

        batch_start += batch_size
        batch_end += batch_size

        return [features, labels]

    return tf.py_func(hdf5_reader_fn, [], [tf.float32, tf.float32],
                      stateful=True)





# =================================================

def get_hdf5_reader_pyfunc(hdf5_file, batch_size):
    '''
    Takes in an hdf5 file and generates a function that returns a group
    of examples and labels
    '''

    # Load in hdf5 file
    hf = h5py.File(hdf5_file, 'r')

    def hdf5_reader_fn():
        '''
        Given batch start and stop, pulls those examples from hdf5 file
        '''
        global batch_start
        global batch_end

        features = hf['train_in'][batch_start:batch_end,:,:,:]
        labels = hf['train_out'][batch_start:batch_end,:]

        batch_start += batch_size
        batch_end += batch_size

        return [features, labels]

    return tf.py_func(hdf5_reader_fn, [], [tf.float32, tf.float32],
                      stateful=True)


def setup_queue(batch_size, features, labels, seq_length, tasks):
    '''
    Function that wraps everything
    '''

    with tf.variable_scope('train_queue'):
        queue = tf.FIFOQueue(10000,
                             [tf.float32, tf.float32],
                             shapes=[[1, seq_length, 4],
                                     [tasks]])
        enqueue_op = queue.enqueue_many([features, labels])
        queue_runner = tf.train.QueueRunner(
            queue=queue,
            enqueue_ops=[enqueue_op],
            close_op=queue.close(),
            cancel_op=queue.close(cancel_pending_enqueues=True))
        tf.train.add_queue_runner(queue_runner, tf.GraphKeys.QUEUE_RUNNERS)

    return queue


def load_data(hdf5_file, batch_size, seq_length, tasks, split='train'):
    '''
    Put it all together
    '''

    global batch_start 
    batch_start = 0
    global batch_end 
    batch_end = batch_size

    [hdf5_features, hdf5_labels] = get_hdf5_reader_pyfunc(hdf5_file,
                                                          batch_size)

    queue = setup_queue(batch_size, hdf5_features, hdf5_labels,
                        seq_length, tasks)

    [features, labels] = queue.dequeue_many(batch_size)

    return features, labels


def get_data_params(hdf5_file):
    '''
    Get basic parameters
    '''

    with h5py.File(hdf5_file, 'r') as hf:
        num_train_examples = hf['train_in'].shape[0]
        seq_length = hf['train_in'].shape[2]
        num_tasks = hf['train_out'].shape[1]

    return num_train_examples, seq_length, num_tasks

# =================================================

# def data_loader_simple(batch_size, seq_length, tasks, data_splits=['train', 'valid', 'test']):
#     '''
#     Set up data queues and conditionals to feed examples and labels
#     hacky set up to test tronn
#     '''
    
#     queues, inputs, enqueue_ops, dequeue, case_dict = {}, {}, {}, {}, {}
#     conditional_fns = []
    
#     with tf.name_scope('inputs') as scope:

#         # Set up data loading for each split. Note that the first split becomes default in the conditional.
#         for split_idx in range(len(data_splits)):
#             split = data_splits[split_idx]
#             queues[split] = tf.FIFOQueue(10000, [tf.float32, tf.float32, tf.string], shapes=[[1, seq_length, 4], [tasks], [1]])
#             inputs[split] = { 'features': tf.placeholder(tf.float32, shape=[batch_size, 1, seq_length, 4]),
#                               'labels' : tf.placeholder(tf.float32, shape=[batch_size, tasks]),
#                               'metadata': tf.placeholder(tf.string, shape=[batch_size, 1]) } 
#             enqueue_ops[split] = queues[split].enqueue_many([inputs[split]['features'],
#                                                              inputs[split]['labels'],
#                                                              inputs[split]['metadata']])
#             dequeue[split] = queues[split].dequeue_many(batch_size)

#         [feature_batch, label_batch, metadata_batch] = dequeue['train']

#     return queues, inputs, enqueue_ops, feature_batch, label_batch, metadata_batch


# def data_loader(batch_size, seq_length, tasks, model_state, data_splits=['train', 'valid', 'test']):
#     '''
#     Set up data queues and conditionals to feed examples and labels
#     '''
    
#     queues, inputs, enqueue_ops, dequeue, case_dict = {}, {}, {}, {}, {}
#     conditional_fns = []
    
#     with tf.name_scope('inputs') as scope:

#         # Set up data loading for each split. Note that the first split becomes default in the conditional.
#         for split_idx in range(len(data_splits)):
#             split = data_splits[split_idx]
#             queues[split] = tf.FIFOQueue(10000, [tf.float32, tf.float32, tf.string], shapes=[[1, seq_length, 4], [tasks], [1]])
#             inputs[split] = { 'features': tf.placeholder(tf.float32, shape=[batch_size, 1, seq_length, 4]),
#                               'labels' : tf.placeholder(tf.float32, shape=[batch_size, tasks]),
#                               'metadata': tf.placeholder(tf.string, shape=[batch_size, 1]) } 
#             enqueue_ops[split] = queues[split].enqueue_many([inputs[split]['features'],
#                                                              inputs[split]['labels'],
#                                                              inputs[split]['metadata']])
#             dequeue[split] = queues[split].dequeue_many(batch_size)
#             #case_dict[tf.equal(model_state, split)] = lambda: dequeue[split]

#             # Setting up this weird stacked conditionals because tf.case is buggy and does not work properly yet.
#             if split_idx == 0:
#                 conditional_fns.append(tf.cond(tf.equal(model_state, split),
#                                                lambda: dequeue[split],
#                                                lambda: dequeue[split]))
#             else:
#                 conditional_fns.append(tf.cond(tf.equal(model_state, split),
#                                                lambda: dequeue[split],
#                                                lambda: conditional_fns[split_idx - 1]))

#         [feature_batch, label_batch, metadata_batch] = conditional_fns[-1] 

#     return queues, inputs, enqueue_ops, feature_batch, label_batch, metadata_batch


# def load_and_enqueue(sess, coord, hdf5_file, batch_size, inputs, enqueue_ops, order='NHWC', data_splits=['train', 'valid', 'test']): 
#     '''
#     This function loads data from the hdf5 files into the feed dicts, which then go into the queues.
#     '''
#     with h5py.File(hdf5_file, 'r') as hf:

#         h5_data = {}
#         for split in data_splits:
#             h5_data[split] = {
#                 'in': hf.get('{}_in'.format(split)),
#                 'out': hf.get('{}_out'.format(split)),
#                 'metadata': hf.get('{}_regions'.format(split))
#             }
#             h5_data[split]['num_examples'] = h5_data[split]['in'].shape[0]
#             h5_data[split]['batch_start'] = 0
#             h5_data[split]['batch_end'] = batch_size

#         with coord.stop_on_exception():
#             while not coord.should_stop():
#                 # Go through the splits and load data
#                 for split in data_splits:
#                     batch_start = h5_data[split]['batch_start']
#                     batch_end = h5_data[split]['batch_end']
#                     if order == 'NHWC':
#                         feature_array = h5_data[split]['in'][batch_start:batch_end,:,:,:]                        
#                     elif order == 'NCHW':
#                         feature_array = np.rollaxis(h5_data[split]['in'][batch_start:batch_end,:,:,:], 1, 4)
#                     else:
#                         print "Axis order not specified!!"
#                     labels_array = h5_data[split]['out'][batch_start:batch_end,:]
#                     metadata_array = h5_data[split]['metadata'][batch_start:batch_end].reshape((batch_size, 1))

#                     sess.run(enqueue_ops[split], feed_dict={inputs[split]['features']: feature_array,
#                                                             inputs[split]['labels']: labels_array,
#                                                             inputs[split]['metadata']: metadata_array}) 

#                     if batch_end + batch_size > h5_data[split]['num_examples']:
#                         h5_data[split]['batch_start'] = 0
#                         h5_data[split]['batch_end'] = batch_size
#                     else:
#                         h5_data[split]['batch_start'] = batch_end
#                         h5_data[split]['batch_end'] = batch_end + batch_size

#     return None

