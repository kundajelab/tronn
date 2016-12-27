"""Contains various utility functions
"""

import math


def calc_stdev(filter_height, filter_width, num_in_channels, style='conv'):
    '''
    For torch7 initialization, you need the standard dev as calculated on
    filter size and fan in. This function calculates that standard dev.
    '''

    if style == 'conv':
        return 1. / math.sqrt(filter_width * filter_height * num_in_channels)
    else:
        return 1. / math.sqrt(num_in_channels)
