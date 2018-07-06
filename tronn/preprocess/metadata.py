# description: code to deal with metadata

import h5py

import numpy as np


def save_metadata(bed_file, h5_file, key="example_metadata"):
    """save out metadata into h5 file
    """
    # read in metadata as numpy
    bed_array = np.loadtxt(bed_file, delimiter="\t", dtype="S1000")
    print bed_array.shape
    
    # save into h5 file
    with h5py.File(h5_file, "a") as hf:
        hf.create_dataset(key, data=bed_array[:,3])

    return None


