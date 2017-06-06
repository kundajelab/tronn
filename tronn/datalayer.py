""" Contains functions for I/O 

Currently hdf5 is supported as an important standardized filetype
used frequently in genomics datasets.

"""

import h5py
import numpy as np
import tensorflow as tf


def get_total_num_examples(hdf5_filename_list):
    '''
    Quickly extracts total examples represented in an hdf5 file list. Can 
    be used to calculate total steps to take (when 1 step represents going 
    through a batch of examples)
    '''
    
    num_examples = 0
    for filename in hdf5_filename_list:
        with h5py.File(filename,'r') as hf:
            num_examples += hf['features'].shape[0]

    return num_examples


def get_positive_weights_per_task(hdf5_file_list):
    '''
    Returns a list of positive weights to be used
    in weighted cross entropy
    '''

    for filename_idx in range(len(hdf5_file_list)):
        with h5py.File(hdf5_file_list[filename_idx], 'r') as hf:
            file_pos = np.sum(hf['labels'], axis=0)
            file_tot = np.repeat(hf['labels'].shape[0], hf['labels'].shape[1])

            if filename_idx == 0:
                total_pos = file_pos
                total_examples = file_tot
            else:
                total_pos += file_pos
                total_examples += file_tot

    # return (total_negs / total_pos) to use as positive weights
    return np.divide(total_examples - total_pos, total_pos)


def get_positives(h5_in_file, task_num, h5_out_file, region_set=None):
    '''
    Quick helper function to just get the positives for one task
    '''

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
                print total_pos
            else:
                is_positive = np.where(in_hf['labels'][:,task_num] > 0)
                total_pos = np.sum(in_hf['labels'][:,task_num] > 0)
                print total_pos


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

                #print is_positive[0][0:10,]
                importance_in = np.array(in_hf[importances_key][in_start_idx:in_end_idx,:,:])
                labels_in = np.array(in_hf['labels'][in_start_idx:in_end_idx,:])
                regions_in = np.array(in_hf['regions'][in_start_idx:in_end_idx,:])
                if region_set != None:
                    current_indices = is_positive[np.logical_and(in_start_idx <= is_positive, is_positive < in_end_idx)] - in_start_idx
                else:
                    current_indices = is_positive[0][np.logical_and(in_start_idx <= is_positive[0], is_positive[0] < in_end_idx)] - in_start_idx

                #current_indices = is_positive[np.logical_and(in_start_idx <= is_positive, is_positive < in_end_idx)]
                #current_indices = is_positive[(in_start_idx <= is_positive) & (is_positive < in_end_idx)]
                #print current_indices[-10:-1,]
                #print current_indices.shape

                out_end_idx = out_start_idx + current_indices.shape[0]

                importances_hf[out_start_idx:out_end_idx,:,:] = importance_in[current_indices,:,:]
                labels_hf[out_start_idx:out_end_idx,:] = labels_in[current_indices,:]
                regions_hf[out_start_idx:out_end_idx,:] = regions_in[current_indices,:]

                in_start_idx += h5_batch_size
                out_start_idx = out_end_idx
                print out_start_idx


def make_bed_from_h5(h5_file, out_file):
    '''
    Extract the regions from an hdf5 to make a bed file
    '''

    with h5py.File(h5_file, 'r') as hf:
        regions = hf['regions']

        with open(out_file, 'w') as out:
            for i in range(regions.shape[0]):
                region = regions[i,0]

                chrom = region.split(':')[0]
                start = region.split(':')[1].split('-')[0]
                stop = region.split(':')[1].split('-')[1]

                out.write('{0}\t{1}\t{2}\n'.format(chrom, start, stop))


def hdf5_to_slices(hdf5_file, batch_size, tasks=[], features_key='features'):
    print "Data layer: loading {}".format(hdf5_file)
    h5py_handle = h5py.File(hdf5_file)
    num_examples = h5py_handle[features_key].shape[0]
    max_batches = num_examples/batch_size
    batch_id_queue = tf.train.range_input_producer(max_batches, shuffle=True)

    # Check shapes from the hdf5 file so that we can set the tensor shapes
    feature_shape = h5py_handle[features_key].shape[1:]
    label_shape = h5py_handle['labels'].shape[1:]
    if len(tasks)==0:
        tasks = range(label_shape[0])

    # Extract examples based on batch_id
    def batchid_to_examples(batch_id):
        batch_start = batch_id*batch_size
        batch_end = batch_start + batch_size
        features = h5py_handle[features_key][batch_start:batch_end]
        if features_key != 'features':# features are importance scores
            features = np.expand_dims(features.transpose(0, 2, 1), axis=1)
        labels = h5py_handle['labels'][batch_start:batch_end, tasks]
        metadata = h5py_handle['regions'][batch_start:batch_end].reshape((batch_size, 1))
        return [features, labels, metadata]


    batch_id_tensor = batch_id_queue.dequeue()
    [features_tensor, labels_tensor, metadata_tensor] = tf.py_func(func=batchid_to_examples,
        inp=[batch_id_tensor],
        Tout=[tf.float32, tf.float32, tf.string],
        stateful=False, name='py_func_batchid_to_examples')
    if features_key == 'features':
        features_tensor.set_shape([batch_size, feature_shape[0], feature_shape[1], feature_shape[2]])
    else:# features are importance scores
        features_tensor.set_shape([batch_size, 1, feature_shape[1], 4])
    labels_tensor.set_shape([batch_size, len(tasks)])
    metadata_tensor.set_shape([batch_size, 1])

    return features_tensor, labels_tensor, metadata_tensor


def load_data_from_filename_list(hdf5_files, batch_size, tasks=[], features_key='features', shuffle_seed=0):
    print 'loading data for tasks:%s from hdf5_files:%s' % (tasks, hdf5_files)
    example_slices_list = [hdf5_to_slices(hdf5_file, batch_size, tasks, features_key) for hdf5_file in hdf5_files]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + (len(example_slices_list)+10) * batch_size
    features, labels, metadata = tf.train.shuffle_batch_join(example_slices_list, batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue, seed=shuffle_seed, enqueue_many=True, name='batcher')
    return features, labels, metadata
