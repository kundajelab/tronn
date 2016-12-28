"""Contains various utility functions
"""

import math
import h5py


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



