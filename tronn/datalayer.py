"""Description: Contains functions for I/O to tensorflow graphs

Currently hdf5 is supported as an important standardized filetype
used frequently in genomics datasets.
"""

import h5py
import numpy as np
import tensorflow as tf


def get_total_num_examples(hdf5_filename_list):
    """Get total number of examples in a list of hdf5 files.

    Args:
      hdf5_filename_list: a list of hdf5 filenames.

    Returns:
      Number of examples.
    """
    num_examples = 0
    for filename in hdf5_filename_list:
        with h5py.File(filename,'r') as hf:
            num_examples += hf['features'].shape[0]

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


def get_positives(h5_in_file, task_num, h5_out_file, region_set=None):
    """Extract positives for a specific task
    
    Args:
      h5_in_file: hdf5 input file
      task_num: task from which to extract positives
      h5_out_file: hdf5 file to store positives
      region_set: text file of indices which correspond to positives

    Returns:
      None
    """
    h5_batch_size = 2000
    with h5py.File(h5_in_file, 'r') as in_hf:
        with h5py.File(h5_out_file, 'w') as out_hf:

            # check paramters of in file
            importances_key = 'importances_task{}'.format(task_num)
            total_examples  = in_hf[importances_key].shape[0]

            if region_set != None:
                pos_indices = np.loadtxt(region_set, dtype=int)
                is_positive = np.sort(pos_indices)
                total_pos = is_positive.shape[0]
                print 'total positives:', total_pos
            else:
                is_positive = np.where(in_hf['labels'][:,task_num] > 0)
                total_pos = np.sum(in_hf['labels'][:,task_num] > 0)
                print 'total positives:', total_pos

            width = in_hf[importances_key].shape[2]
            total_labels = in_hf['labels'].shape[1]
            
            # create new datasets: importances, labels, regions
            importances_hf = out_hf.create_dataset(importances_key, [total_pos, 4, width])
            labels_hf = out_hf.create_dataset('labels', [total_pos, total_labels])
            regions_hf = out_hf.create_dataset('regions', [total_pos, 1], dtype='S100')
            
            # copy over positives by chunks
            in_start_idx = 0
            out_start_idx = 0
            
            while in_start_idx < total_examples:
                print in_start_idx
                in_end_idx = in_start_idx + h5_batch_size

                importance_in = np.array(in_hf[importances_key][in_start_idx:in_end_idx,:,:])
                labels_in = np.array(in_hf['labels'][in_start_idx:in_end_idx,:])
                regions_in = np.array(in_hf['regions'][in_start_idx:in_end_idx,:])
                if region_set != None:
                    current_indices = is_positive[np.logical_and(in_start_idx <= is_positive, is_positive < in_end_idx)] - in_start_idx
                else:
                    current_indices = is_positive[0][np.logical_and(in_start_idx <= is_positive[0], is_positive[0] < in_end_idx)] - in_start_idx

                out_end_idx = out_start_idx + current_indices.shape[0]

                importances_hf[out_start_idx:out_end_idx,:,:] = importance_in[current_indices,:,:]
                labels_hf[out_start_idx:out_end_idx,:] = labels_in[current_indices,:]
                regions_hf[out_start_idx:out_end_idx,:] = regions_in[current_indices,:]

                in_start_idx += h5_batch_size
                out_start_idx = out_end_idx
                print out_start_idx


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


def hdf5_to_slices(hdf5_file, batch_size, tasks=[], features_key='features', shuffle=False):
    """Extract batches from hdf5 file

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
    feature_shape = h5py_handle[features_key].shape[1:]
    label_shape = h5py_handle['labels'].shape[1:]
    if len(tasks) == 0:
        tasks = range(label_shape[0])

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
        features = h5py_handle[features_key][batch_start:batch_end]
        if features_key != 'features':# features are importance scores
            features = np.expand_dims(features.transpose(0, 2, 1), axis=1)
        labels = h5py_handle['labels'][batch_start:batch_end, tasks]
        metadata = h5py_handle['regions'][batch_start:batch_end].reshape((batch_size, 1))
        return [features, labels, metadata]

    batch_id_tensor = batch_id_queue.dequeue()
    [features_tensor, labels_tensor, metadata_tensor] = tf.py_func(
        func=batchid_to_examples,
        inp=[batch_id_tensor],
        Tout=[tf.float32, tf.float32, tf.string],
        stateful=False, name='py_func_batchid_to_examples')

    # set shapes
    if features_key == 'features':
        features_tensor.set_shape([batch_size, feature_shape[0], feature_shape[1], feature_shape[2]])
    else: # features are importance scores
        features_tensor.set_shape([batch_size, 1, feature_shape[1], 4])
    labels_tensor.set_shape([batch_size, len(tasks)])
    metadata_tensor.set_shape([batch_size, 1])

    return features_tensor, labels_tensor, metadata_tensor


def hdf5_list_to_slices(hdf5_files, batch_size, tasks=[], features_key='features', shuffle=False):
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
    # Extract hdf5 file params (number of examples, max batches, batch IDs)
    #print "Data layer: loading {}".format(hdf5_files)
    h5py_handles = [h5py.File(hdf5_file) for hdf5_file in hdf5_files]
    num_examples_per_file = [get_total_num_examples([hdf5_file]) for hdf5_file in hdf5_files]
    total_num_examples = get_total_num_examples(hdf5_files)
    max_batches = total_num_examples / batch_size - 10

    batch_id_queue = tf.train.range_input_producer(max_batches, shuffle=shuffle)

    # generate a batch_to_file dictionary so it's easy to get the file
    global_batch_to_file_idx = {}
    local_batch = 0
    current_file_idx = 0
    current_file_example_total = num_examples_per_file[current_file_idx] / batch_size
    for global_batch in xrange(max_batches):
        if local_batch < current_file_example_total:
            global_batch_to_file_idx[global_batch] = (current_file_idx, local_batch)
            local_batch += 1
        else:
            local_batch = 0

            while True:
                # go to next file index
                if (current_file_idx + 1) == len(h5py_handles):
                    current_file_idx = 0
                else:
                    current_file_idx += 1
                
                current_file_example_total = num_examples_per_file[current_file_idx] / batch_size
                if local_batch < current_file_example_total:
                    break
            
            #current_file_example_total = num_examples_per_file[current_file_idx] / batch_size
            global_batch_to_file_idx[global_batch] = (current_file_idx, local_batch)
            local_batch += 1
    
    # Check shapes from the hdf5 file so that we can set the tensor shapes
    feature_shape = h5py_handles[0][features_key].shape[1:]
    label_shape = h5py_handles[0]['labels'].shape[1:]
    if len(tasks) == 0:
        tasks = range(label_shape[0])

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
        file_idx, local_batch = global_batch_to_file_idx[batch_id]
            
	# check end point of batch
	h5py_handle = h5py_handles[file_idx]
        batch_start = local_batch*batch_size
        batch_end = batch_start + batch_size

        features = h5py_handle[features_key][batch_start:batch_end]
        if features_key != 'features':# features are importance scores
            features = np.expand_dims(features.transpose(0, 2, 1), axis=1)
        labels = h5py_handle['labels'][batch_start:batch_end, tasks]
        metadata = h5py_handle['regions'][batch_start:batch_end].reshape((batch_size, 1))
        return [features, labels, metadata]

    batch_id_tensor = batch_id_queue.dequeue()
    [features_tensor, labels_tensor, metadata_tensor] = tf.py_func(
        func=batchid_to_examples,
        inp=[batch_id_tensor],
        Tout=[tf.float32, tf.float32, tf.string],
        stateful=False, name='py_func_batchid_to_examples')

    # set shapes
    if features_key == 'features':
        features_tensor.set_shape([batch_size, feature_shape[0], feature_shape[1], feature_shape[2]])
    else: # features are importance scores
        features_tensor.set_shape([batch_size, 1, feature_shape[1], 4])
    labels_tensor.set_shape([batch_size, len(tasks)])
    metadata_tensor.set_shape([batch_size, 1])

    return features_tensor, labels_tensor, metadata_tensor


def load_data_from_filename_list(hdf5_files, batch_size, tasks=[], features_key='features', shuffle=True, shuffle_seed=0):
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
    print 'loading data for tasks:%s from hdf5_files:%s' % (tasks, hdf5_files)
    # TODO add in ordered loading (for importance scores)
    # filename_id? dequeue in order?
    # with the ordered filename ids, make a py_func that takes in the hdf5_to_slices function, and input is the filename,
    # to output tensors
    if shuffle:
        example_slices_list = [hdf5_to_slices(hdf5_file, batch_size, tasks, features_key, shuffle=True) for hdf5_file in hdf5_files]
        min_after_dequeue = 10000
        capacity = min_after_dequeue + (len(example_slices_list)+10) * batch_size
        features, labels, metadata = tf.train.shuffle_batch_join(example_slices_list,
                                                                 batch_size,
                                                                 capacity=capacity,
                                                                 min_after_dequeue=min_after_dequeue,
                                                                 seed=shuffle_seed,
                                                                 enqueue_many=True,
                                                                 name='batcher')

    elif shuffle == False:
    	example_slices_list = hdf5_list_to_slices(hdf5_files, batch_size, tasks, features_key)
    	features, labels, metadata = tf.train.batch(example_slices_list,
                                                                 batch_size,
                                                                 capacity=100000,
                                                                 enqueue_many=True,
                                                                 name='batcher')



        
    return features, labels, metadata


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
        metadata = h5py_handle['regions'][batch_start:batch_end].reshape((batch_size, 1))
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


def tflearn_input_fn(hdf5_files, batch_size, tasks=[], features_key='features', shuffle=True, shuffle_seed=0): # CHANGE THIS LATER
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
        
        return features, labels

    return load_kmer_data_from_filename_list
