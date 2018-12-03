# description: some helpful utils for working with h5 files

import h5py

import numpy as np
import pandas as pd

from tronn.util.utils import DataKeys


class AttrKeys(object):
    """standard names for attributes in h5 files
    """

    PWM_NAMES = "pwm_names"
    CLUSTER_IDS = "cluster_ids"
    FILE_NAMES = "file_names"
    TASK_INDICES = "task_indices"
    PLOT_LABELS = "labels"


def add_pwm_names_to_h5(
        h5_file,
        pwm_names,
        substring=DataKeys.PWM_SCORES_ROOT,
        pwm_names_key=AttrKeys.PWM_NAMES,
        other_keys=[]):
    """if substring is in the dataset key, then add the 
    pwm names as an attribute t o h5
    """
    with h5py.File(h5_file, "a") as hf:
        for key in hf.keys():
            if substring in key:
                hf[key].attrs[pwm_names_key] = pwm_names
            if key in other_keys:
                hf[key].attrs[pwm_names_key] = pwm_names
                
    return None


def add_task_metadata_to_h5(
        h5_file):
    """this should transfer over the filenames and indices as attributes
    """


    return


def copy_h5_datasets(in_h5_file, out_h5_file, keys=[]):
    """copy keys into new file
    """
    for key in keys:
        with h5py.File(in_h5_file, "r") as hf:
            with h5py.File(out_h5_file, "a") as out:
                # copy over data
                out.create_dataset(key, data=hf[key][:])
                # and copy all attributes
                for attr_key, val in hf[key].attrs.iteritems():
                    out[key].attrs[attr_key] = val
    
    return None


def copy_h5_dataset_slices(in_h5_file, out_h5_file, keys=[], indices=[], test_key=DataKeys.LABELS):
    """copy slices by indices into output h5
    """
    assert len(keys) != 0
    assert len(indices) != 0
    # first check shape
    with h5py.File(in_h5_file, "r") as hf:
        num_examples = hf[test_key].shape[0]
    
    for key in keys:
        with h5py.File(in_h5_file, "r") as hf:
            with h5py.File(out_h5_file, "a") as out:
                # copy over data
                if hf[key].shape[0] == num_examples:
                    out.create_dataset(key, data=hf[key][indices])
                else:
                    out.create_dataset(key, data=hf[key][:])
                # and copy all attributes
                for attr_key, val in hf[key].attrs.iteritems():
                    out[key].attrs[attr_key] = val

    return None
