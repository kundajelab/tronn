"""Contains functions for I/O to tensorflow graphs
"""

import h5py
import math
import logging

import numpy as np
import tensorflow as tf


def get_total_num_examples(hdf5_filename_list, feature_key="features"):
    """Get total number of examples in a list of hdf5 files.

    Args:
      hdf5_filename_list: a list of hdf5 filenames.

    Returns:
      Number of examples.
    """
    num_examples = 0
    for filename in hdf5_filename_list:
        with h5py.File(filename,'r') as hf:
            num_examples += hf[feature_key].shape[0]

    return num_examples


def get_positive_weights_per_task(hdf5_filename_list):
    """Calculates positive weights to be used in weighted cross entropy
    
    Args:
      hdf_filename_list: a list of hdf5 filenames.

    Returns:
      List of positive weights calculated as (total_negs / total_pos)
    """
    for filename_idx in range(len(hdf5_filename_list)):
        with h5py.File(hdf5_filename_list[filename_idx], 'r') as hf:
            file_pos = np.sum(hf['labels'], axis=0)
            file_tot = np.repeat(hf['labels'].shape[0], hf['labels'].shape[1])

            if filename_idx == 0:
                total_pos = file_pos
                total_examples = file_tot
            else:
                total_pos += file_pos
                total_examples += file_tot

    return np.divide(total_examples - total_pos, total_pos)


def get_task_and_class_weights(hdf5_filename_list):
    """Calculates task and class weights to be used in positives focused loss
    
    Args:
      hdf_filename_list: a list of hdf5 filenames.

    Returns:
      List of task weights (pos/negs) and class weights (negs/pos)
    """
    for filename_idx in range(len(hdf5_filename_list)):
        with h5py.File(hdf5_filename_list[filename_idx], 'r') as hf:
            file_pos = np.sum(hf['labels'], axis=0)
            file_tot = np.repeat(hf['labels'].shape[0], hf['labels'].shape[1])

            if filename_idx == 0:
                total_pos = file_pos
                total_examples = file_tot
            else:
                total_pos += file_pos
                total_examples += file_tot

    return np.divide(total_pos, total_examples - total_pos), np.divide(total_examples - total_pos, total_pos)


def make_bed_from_h5(h5_file, bed_file):
    """Extract the regions from an hdf5 to make a bed file
    
    Args:
      h5_file: input file with 'regions'
      bed_file: output BED format file

    Returns:
      None
    """
    with h5py.File(h5_file, 'r') as hf:
        regions = hf['regions']
        with open(bed_file, 'w') as out:
            for i in range(regions.shape[0]):
                region = regions[i,0]
                chrom = region.split(':')[0]
                start = region.split(':')[1].split('-')[0]
                stop = region.split(':')[1].split('-')[1]
                out.write('{0}\t{1}\t{2}\n'.format(chrom, start, stop))


def hdf5_to_slices(
        h5_handle,
        task_indices,
        start_idx,
        batch_size,
        features_key="features",
        labels_key="labels",
        metadata_key="example_metadata"):
    """Get slices from the datasets back, and pad with null sequences as needed to
    fill the batch.

    Args:
      h5_handle: open h5py file
      task_indices: list of task indices that you want
      start_idx: start position in file
      batch_size: size of batch
      features_key: features key in h5 file. Can be features or importances
      labels_key: labels key in h5 file
      metadata_key: metadata key in h5 file

    Returns:
      slices of features, labels, metadata 
    """
    end_idx = start_idx + batch_size

    # if end is within the file, extract out slice. otherwise, pad and pass out full batch
    if end_idx < h5_handle[features_key].shape[0]:
        features = h5_handle[features_key][start_idx:end_idx]
        labels = h5_handle[labels_key][start_idx:end_idx, task_indices]
        metadata = h5_handle[metadata_key][start_idx:end_idx].reshape(
            (batch_size, 1))

    else:
        end_idx_file = h5_handle[features_key].shape[0]
        batch_padding_num = batch_size - (end_idx_file - start_idx)

        # features - {batch_size, 1, seq_len, 4}
        features_tmp = h5_handle[features_key][start_idx:end_idx_file]
        features_padding_shape = [batch_padding_num] + list(features_tmp.shape[1:])
        features_padding_array = np.zeros(features_padding_shape, dtype=np.float32)
        features = np.concatenate([features_tmp, features_padding_array], axis=0)
            
        # labels - {batch_size, task_num}
        labels_tmp = h5_handle[labels_key][start_idx:end_idx_file, task_indices]
        labels_padding_shape = [batch_padding_num, len(task_indices)]
        labels_padding_array = np.zeros(labels_padding_shape, dtype=np.float32)
        labels = np.concatenate([labels_tmp, labels_padding_array], axis=0)

        # metadata - {batch_size, 1}
        metadata_tmp = h5_handle[metadata_key][start_idx:end_idx_file].reshape(
            (end_idx_file - start_idx, 1))
        metadata_padding_array = np.array(
            ["false=chrY:0-0" for i in xrange(batch_padding_num)]).reshape(
                (batch_padding_num, 1))
        metadata = np.concatenate([metadata_tmp, metadata_padding_array], axis=0)
        
    return [features, labels, metadata]

                
def hdf5_to_tensors(
        hdf5_file,
        batch_size,
        task_indices=[],
        features_key="features",
        labels_key="labels",
        shuffle=True,
        num_epochs=1,
        fake_task_num=0):
    """Extract batches from a single hdf5 file.
    
    Args:
      hdf5_file: hdf5 file with input data 
      batch_size: desired batch size to return to tensor
      tasks: use if a subset of tasks desired
      features_key: the key for the input dataset examples ('features' or 'importances')
      shuffle: shuffle the batches
      fake_task_num: how many EXTRA outputs to generate to fill in for multi-output

    Returns:
      features_tensor: a tensor (batch_size, feature_dims) for examples
      labels_tensor: a tensor (batch_size, num_tasks) for labels
      metadata_tensor: a tensor (batch_size, 1) for a metadata string (often region name)
    """
    # get hdf5 file params
    logging.info("loading {}".format(hdf5_file))
    h5_handle = h5py.File(hdf5_file)
    if len(task_indices) == 0:
        task_indices = range(h5_handle[labels_key].shape[1])
    
    # set up batch id producer
    max_batches = int(math.ceil(h5_handle[features_key].shape[0]/float(batch_size)))
    logging.info("max batches: {}".format(max_batches))
    batch_id_queue = tf.train.range_input_producer(
        max_batches, shuffle=shuffle, num_epochs=num_epochs)
    batch_id_tensor = batch_id_queue.dequeue()

    # get examples based on batch_id
    def batchid_to_examples(batch_id):
        """Given a batch ID, get the features, labels, and metadata
        This is an important wrapper to be able to use TensorFlow's pyfunc.

        Args:
          batch_id: an int that is the batch ID

        Returns:
          tensors of features, labels, metadata
        """
        batch_start = batch_id*batch_size
        return hdf5_to_slices(
            h5_handle,
            task_indices,
            batch_start,
            batch_size,
            features_key=features_key)

    # pyfunc
    [features_tensor, labels_tensor, metadata_tensor] = tf.py_func(
        func=batchid_to_examples,
        inp=[batch_id_tensor],
        Tout=[tf.float32, tf.float32, tf.string],
        stateful=False, name='py_func_batchid_to_examples')

    # set shapes
    features_tensor.set_shape([batch_size] + list(h5_handle[features_key].shape[1:]))
    labels_tensor.set_shape([batch_size, len(task_indices)])
    metadata_tensor.set_shape([batch_size, 1])

    # fake tasks num: this is to be able to check a multitask model on a single output dataset
    if fake_task_num > 0:
        labels_list = tf.unstack(labels_tensor, axis=1)
        labels_final = labels_list + [labels_list[-1] for i in xrange(fake_task_num)]
        labels_tensor = tf.stack(labels_final, axis=1)
    
    return features_tensor, labels_tensor, metadata_tensor


def hdf5_list_to_ordered_tensors(
        hdf5_files, 
        batch_size, 
        task_indices=[], 
        features_key="features",
        labels_key="labels",
        shuffle=False,
        num_epochs=1,
        fake_task_num=0):
    """Extract batches from hdf5 file list. This is used to get ordered examples out 
    to re-merge back into regions.

    Args:
      hdf5_files: list of hdf5 files with input data 
      batch_size: desired batch size to return to tensor
      tasks: use if a subset of tasks desired
      features_key: the key for the input dataset examples ('features' or 'importances')

    Returns:
      features_tensor: a tensor (batch_size, feature_dims) for examples
      labels_tensor: a tensor (batch_size, num_tasks) for labels
      metadata_tensor: a tensor (batch_size, 1) for a metadata string (often region name)
    """
    # get hdf5 file params
    logging.info("loading {}".format(" ".join(hdf5_files)))
    h5_handles = [h5py.File(hdf5_file) for hdf5_file in hdf5_files]
    num_examples_per_file = [get_total_num_examples([hdf5_file]) for hdf5_file in hdf5_files]
    total_batches_per_file = [int(math.ceil(num_examples) / float(batch_size))
                              for num_examples in num_examples_per_file ]
    if len(task_indices) == 0:
        task_indices = range(h5_handles[0][labels_key].shape[1])
    
    # set up batch id producer
    max_batches = sum(total_batches_per_file)
    logging.info("max batches: {}".format(max_batches))
    batch_id_queue = tf.train.range_input_producer(
        max_batches, shuffle=shuffle, num_epochs=num_epochs)
    batch_id_tensor = batch_id_queue.dequeue()
    
    # generate a batch_to_file dictionary so it's easy to get the file
    global_batch_to_file_idx = {}
    local_batch = 0
    current_file_idx = 0
    for global_batch in xrange(max_batches):
        if local_batch > total_batches_per_file[current_file_idx]:
            # go to next file idx
            local_batch = 0
            if (current_file_idx + 1) == len(h5_handles): # loop around as needed
                current_file_idx = 0
            else:
                current_file_idx += 1
        global_batch_to_file_idx[global_batch] = (current_file_idx, local_batch)
        local_batch += 1

    # Extract examples based on batch_id
    def batchid_to_examples(batch_id):
        """Given a batch ID, get the features, labels, and metadata

        Args:
          batch_id: an int that is the batch ID

        Returns:
          tensors of features, labels, metadata
        """
        # set up h5_handle and batch_start
        file_idx, local_batch = global_batch_to_file_idx[batch_id]
	h5_handle = h5py_handles[file_idx]
        batch_start = local_batch*batch_size
        return hdf5_to_slices(
            h5_handle,
            task_indices,
            batch_start,
            batch_size,
            features_key=features_key)

    # pyfunc
    [features_tensor, labels_tensor, metadata_tensor] = tf.py_func(
        func=batchid_to_examples,
        inp=[batch_id_tensor],
        Tout=[tf.float32, tf.float32, tf.string],
        stateful=False, name='py_func_batchid_to_examples')

    # set shapes
    features_tensor.set_shape([batch_size] + list(h5_handles[0][features_key][1:]))
    labels_tensor.set_shape([batch_size, len(task_indices)])
    metadata_tensor.set_shape([batch_size, 1])

    # fake tasks num: this is to be able to check a multitask model on a single output dataset
    if fake_task_num > 0:
        labels_list = tf.unstack(labels_tensor, axis=1)
        labels_final = labels_list + [labels_list[-1] for i in xrange(fake_task_num)]
        labels_tensor = tf.stack(labels_final, axis=1)

    return features_tensor, labels_tensor, metadata_tensor


def filter_through_labels(features, labels, metadata, filter_tasks):
    """Given specific filter tasks, only push through examples 
    if they are positive in these tasks
    """
    # set up labels mask for filter tasks
    labels_mask_np = np.zeros((labels.get_shape()[1]))
    for task_idx in filter_tasks:
        labels_mask_np[task_idx] = 1
    labels_mask = tf.cast(
        tf.stack([tf.constant(labels_mask_np) for i in xrange(batch_size)], axis=0),
        tf.float32)
        
    # run a conditional on the labels and get indices
    pos_labels = tf.multiply(labels, labels_mask)
    matches = tf.greater(tf.reduce_sum(pos_labels, axis=1), [0])
    selected_items = tf.reshape(tf.where(matches), [-1])

    # gather
    features_filtered = tf.gather(features, selected_items)
    labels_filtered = tf.gather(labels, selected_items)
    metadata_filtered = tf.gather(metadata, selected_items)
        
    # set up a second queue
    features, labels, metadata = tf.train.batch(
        [features_filtered, labels_filtered, metadata_filtered],
        batch_size,
        capacity=100000,
        enqueue_many=True,
        name="filter_batcher")
    
    return features, labels, metadata


def load_data_from_filename_list(
        hdf5_files,
        batch_size,
        task_indices=[],
        features_key='features',
        shuffle=True,
        shuffle_seed=0,
        ordered_num_epochs=1,
        fake_task_num=0,
        filter_tasks=[]):
    """Load data into queues from a filename list of hdf5 files

    Args:
      hdf_files: list of hdf5 filenames
      batch_size: batch size
      tasks: list of tasks if using subset of tasks
      features_key: features key ('features' or 'importances')
      shuffle_seed: seed to make randomness deterministic

    Returns:
      features: feature tensor
      labels: label tensor
      metadata: metadata tensor
    """
    logging.info("loading data for tasks:%s from hdf5_files:%s" % (task_indices, hdf5_files))

    # TODO(dk) check file sizes and change num epochs as needed to fill the queue - or change queue capacity?
    
    if shuffle:
        # use a thread for each hdf5 file to put together in parallel
        example_slices_list = [hdf5_to_tensors(hdf5_file, batch_size, task_indices, features_key, shuffle=True, fake_task_num=fake_task_num)
                               for hdf5_file in hdf5_files]
        min_after_dequeue = 10000
        capacity = min_after_dequeue + (len(example_slices_list)+10) * batch_size
        features, labels, metadata = tf.train.shuffle_batch_join(
            example_slices_list,
            batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            seed=shuffle_seed,
            enqueue_many=True,
            name='batcher')

    elif shuffle == False:
        # since ordered, keep hdf5 files in a list
    	example_slices_list = hdf5_list_to_ordered_tensors(
            hdf5_files, batch_size, task_indices, features_key, fake_task_num=fake_task_num, num_epochs=ordered_num_epochs)
    	features, labels, metadata = tf.train.batch(
            example_slices_list,
            batch_size,
            capacity=100000,
            enqueue_many=True,
            name='batcher')

    # filtering as desired
    if len(filter_tasks) > 0:
        features, labels, metadata = filter_through_labels(
            features, labels, metadata, filter_tasks)
        
    return features, labels, metadata


# =======================
# other input functions
# maybe defunct
# =======================


def hdf5_kmers_to_slices(hdf5_file, batch_size, tasks=[], features_key='features', shuffle=True):
    """Reads in kmer feature file from hdf5

    Args:
      hdf5_file: hdf5 file with input data 
      batch_size: desired batch size to return to tensor
      tasks: use if a subset of tasks desired
      features_key: the key for the input dataset examples ('features' or 'importances')

    Returns:
      features_tensor: a tensor (batch_size, feature_dims) for examples
      labels_tensor: a tensor (batch_size, num_tasks) for labels
      metadata_tensor: a tensor (batch_size, 1) for a metadata string (often region name)
    """
    # Extract hdf5 file params (number of examples, max batches, batch IDs)
    print "Data layer: loading {}".format(hdf5_file)
    h5py_handle = h5py.File(hdf5_file)
    num_examples = h5py_handle[features_key].shape[0]
    max_batches = num_examples/batch_size
    batch_id_queue = tf.train.range_input_producer(max_batches, shuffle=shuffle)

    # Check shapes from the hdf5 file so that we can set the tensor shapes
    num_features = h5py_handle[features_key].shape[1]
    num_labels= h5py_handle['labels'].shape[1]
    if len(tasks) == 0:
        tasks = range(num_labels)

    # Extract examples based on batch_id
    def batchid_to_examples(batch_id):
        """Given a batch ID, get the features, labels, and metadata

        Args:
          batch_id: an int that is the batch ID

        Returns:
          features: feature array
          labels: label array
          metadata: metadata array
        """
        batch_start = batch_id*batch_size
        batch_end = batch_start + batch_size
        features = h5py_handle[features_key][batch_start:batch_end,:]
        labels = h5py_handle['labels'][batch_start:batch_end, tasks]
        metadata = h5py_handle['example_metadata'][batch_start:batch_end].reshape((batch_size, 1))
        return [features, labels, metadata]

    batch_id_tensor = batch_id_queue.dequeue()
    [features_tensor, labels_tensor, metadata_tensor] = tf.py_func(
        func=batchid_to_examples,
        inp=[batch_id_tensor],
        Tout=[tf.float32, tf.float32, tf.string], 
        stateful=False, name='py_func_batchid_to_examples')

    # set shapes
    features_tensor.set_shape([batch_size, num_features])
    labels_tensor.set_shape([batch_size, len(tasks)])
    metadata_tensor.set_shape([batch_size, 1])

    return features_tensor, labels_tensor, metadata_tensor


def tflearn_kmer_input_fn(
        hdf5_files,
        batch_size,
        tasks=[],
        features_key='features',
        shuffle=True,
        shuffle_seed=0,
        featurize_fn=None,
        featurize_params={}): 
    """Wrapper to make input function work in TFLearn
    """

    def load_kmer_data_from_filename_list():
        """Load kmer features into TFLearn style output (features, labels)
        """
        
        example_slices_list = [hdf5_kmers_to_slices(hdf5_file, batch_size, tasks, features_key, shuffle=shuffle) for hdf5_file in hdf5_files]
        min_after_dequeue = 10000
        capacity = min_after_dequeue + (len(example_slices_list)+10) * batch_size
        features, labels, metadata = tf.train.shuffle_batch_join(example_slices_list,
                                                                 batch_size,
                                                                 capacity=capacity,
                                                                 min_after_dequeue=min_after_dequeue,
                                                                 seed=shuffle_seed,
                                                                 enqueue_many=True,
                                                                 name='batcher')

        # TODO(dk) put in a featurization layer here?
        if featurize_fn is not None:
            features = featurize_fn(features, **featurize_params)

            print features.get_shape()
            
        return features, labels

    return load_kmer_data_from_filename_list


def tflearn_input_fn(
        hdf5_files,
        batch_size,
        tasks=[],
        features_key='features',
        shuffle=True,
        shuffle_seed=0,
        featurize_fn=None,
        featurize_params={}): 
    """Wrapper to make input function work in TFLearn
    """
    
    def load_onehot_sequences_from_filename_list():
        """Function to put into tflearn
        """
        example_slices_list = [hdf5_to_slices(hdf5_file, batch_size, tasks, features_key, shuffle=True)
                               for hdf5_file in hdf5_files]
        min_after_dequeue = 10000
        capacity = min_after_dequeue + (len(example_slices_list)+10) * batch_size
        features, labels, metadata = tf.train.shuffle_batch_join(
            example_slices_list,
            batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            seed=shuffle_seed,
            enqueue_many=True,
            name='batcher')
        
        if featurize_fn is not None:
            #with tf.device("/cpu:0"):
            features = featurize_fn(features, **featurize_params)

        return features, labels

    return load_onehot_sequences_from_filename_list






