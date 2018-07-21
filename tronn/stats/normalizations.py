# description: various normalizations

import numpy


def rownorm2d(array):
    """generic row normalization
    """
    max_vals = np.max(array, axis=1, keepdims=True)
    array_norm = np.divide(
        array,
        max_vals,
        out=np.zeros_like(array),
        where=max_vals!=0)
    
    return array_norm



