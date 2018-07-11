# description: some helpful utils for working with h5 files

import h5py

import numpy as np
import pandas as pd


# TODO try to deprecate this?
def get_absolute_label_indices(label_keys, key_dict, test_h5_file, list_of_lists=False):
    """given a key dict and model, get the absolute indices

    label_keys: the full ordered set used in training
    key_dict: the ones you actually want to grab back
    h5_file: a representative h5 file with all the label keys

    """
    absolute_task_indices = []
    absolute_task_indices_subsetted = []
    
    start_idx = 0
    for key in label_keys:
        # track how many indices in this key
        with h5py.File(test_h5_file, "r") as hf:
            num_tasks = hf[key].shape[1] # basically just need this number
        if key in key_dict.keys():
            # and create absolute indices and append
            if len(key_dict[key][0]) > 0:
                key_indices = [start_idx + idx for idx in key_dict[key][0]]
            else:
                key_indices = [start_idx + idx for idx in xrange(num_tasks)]
            absolute_task_indices += key_indices
            absolute_task_indices_subsetted.append(key_indices)

        # and then adjust start idx
        start_idx += num_tasks

    if list_of_lists:
        return absolute_task_indices_subsetted
    else:
        return absolute_task_indices


