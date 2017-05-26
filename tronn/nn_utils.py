"""Contains various utility functions
"""

import math
import h5py
import numpy as np


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


def get_fan_in(tensor, type='NHWC'):
    '''
    Get the fan in (number of in channels)
    '''

    return int(tensor.get_shape()[-1])


def get_positives(h5_in_file, task_num, h5_out_file):
    '''
    Quick helper function to just get the positives for one task
    '''

    h5_batch_size = 2000
    
    with h5py.File(h5_in_file, 'r') as in_hf:
        with h5py.File(h5_out_file, 'w') as out_hf:

            # check paramters of in file
            importances_key = 'importances_task{}'.format(task_num)
            total_examples  = in_hf[importances_key].shape[0]
            is_positive = np.where(in_hf['labels'][:,task_num] > 0)
            total_pos = np.sum(in_hf['labels'][:,task_num] > 0)
            width = in_hf[importances_key].shape[2]
            total_labels = in_hf['labels'].shape[1]
            print np.sum(in_hf['labels'][:,task_num] > 0)
            
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

    return None



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

    return None


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


    return None
