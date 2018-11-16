"""description: code for working with combinatorial functions
"""

import itertools

import numpy as np


def setup_combinations(num_items):
    """helper function to set up a combinatorial matrix
    """
    combinations = np.zeros((num_items, 2**num_items)) # {num_items, 2**num_items}
    bool_presence = [[0,1]]*num_items
    i = 0
    for combo in itertools.product(*bool_presence):
        combinations[:,i] = combo
        i += 1 
    
    return combinations
    
