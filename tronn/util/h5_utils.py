# description: some helpful utils for working with h5 files

import os
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


def load_data_from_multiple_h5_files(h5_files, key, example_indices=None, concat=True):
    """convenience wrapper
    example indices is a list of index arrays, ordered same as h5 files
    """
    key_data = []
    for h5_idx in range(len(h5_files)):
        h5_file = h5_files[h5_idx]
        with h5py.File(h5_file, "r") as hf:
            if example_indices is not None:
                key_data.append(hf[key][:][example_indices[h5_idx]])
            else:
                key_data.append(hf[key][:])

    if concat:
        key_data = np.concatenate(key_data, axis=0)
    
    return key_data



def compress_h5_file(input_file, output_file=None):
    """take an h5 file and repack to compress
    """
    # if output file is none, then we want to compress
    # and save to original file. make a tmp
    if output_file is None:
        tmp_file = "{}/{}.tmp.h5".format(
            os.path.dirname(input_file),
            os.path.basename(input_file).split(".h5")[0].split(".hdf5")[0])
        os.system("mv {} {}".format(input_file, tmp_file))
        output_file = input_file
        input_file = tmp_file
    else:
        tmp_file = None
        
    # pull the keys
    with h5py.File(input_file, "r") as hf:
        keys = sorted(hf.keys())
    
    # for each key
    for key in keys:
        print key
        with h5py.File(input_file, "r") as hf:
            with h5py.File(output_file, "a") as out:

                if len(hf[key].shape) != 0:
                    # array: compress and copy
                    out.create_dataset(
                        key, data=hf[key],
                        compression="gzip", compression_opts=9, shuffle=True)
                else:
                    # scalar: just copy
                    out.create_dataset(key, data=hf[key])
                    
                # and copy all attributes
                for attr_key, val in hf[key].attrs.iteritems():
                    out[key].attrs[attr_key] = val

    # clean up
    if tmp_file is not None:
        os.system("rm {}".format(tmp_file))
    
    return
