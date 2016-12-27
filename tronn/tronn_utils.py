"""Contains various utility functions
"""

import math


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


def calc_stdev(filter_height, filter_width, num_in_channels, style='conv'):
    '''
    For torch7 initialization, you need the standard dev as calculated on
    filter size and fan in. This function calculates that standard dev.
    '''

    if style == 'conv':
        return 1. / math.sqrt(filter_width * filter_height * num_in_channels)
    else:
        return 1. / math.sqrt(num_in_channels)


