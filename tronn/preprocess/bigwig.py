# description: process bigwig files by regions

import os
import logging
import h5py

import numpy as np
import pandas as pd


def generate_signal_vals(
        bed_file,
        bigwig_files,
        key,
        h5_file,
        tmp_dir="."):
    """given a region set (BED file), calculate
    average signal in each region from the bigwigs
    and save out the array to h5.
    """
    os.system("mkdir -p {}".format(tmp_dir))
    num_files = len(bigwig_files)
    logging.info("Found {} bigwig files as annotations".format(num_files))

    # get metadata
    file_metadata = [
        "index={0};file={1}".format(
            i, os.path.basename(bigwig_files[i]))
        for i in xrange(len(bigwig_files))]

    # for each bigwig file
    for i in xrange(len(bigwig_files)):
        bigwig_file = bigwig_files[i]

        # generate signals        
        #out_tmp_file = "{}/{}_x_{}.signal.tmp".format(
        #    tmp_dir,
        #    os.path.basename(bed_file),
        #    os.path.basename(bigwig_file))
        out_tmp_file = "{}/{}.signal.tmp".format(
            tmp_dir,
            os.path.basename(bigwig_file))
        get_average_signal = (
            "bigWigAverageOverBed {} {} {}").format(
                bigwig_file, bed_file, out_tmp_file)
        print get_average_signal
        os.system(get_average_signal)

        # read in and save to h5 file
        bigwig_average_signals = pd.read_table(
            out_tmp_file,
            header=None,
            names=["name", "size", "covered", "sum", "mean0", "mean"])

        if i == 0:
            num_rows = bigwig_average_signals.shape[0]
            with h5py.File(h5_file, "a") as hf:
                all_signals = np.zeros((num_rows, num_files))
                hf.create_dataset(key, data=all_signals)
                hf[key][:,i] = bigwig_average_signals["mean0"]
        else:
            with h5py.File(h5_file, "a") as hf:
                hf[key][:,i] = bigwig_average_signals["mean0"]
                
        # delete tmp file
        os.system("rm {}".format(out_tmp_file))

    # add in bigwig files as attr on h5
    with h5py.File(h5_file, "a") as hf:
        hf[key].attrs["filenames"] = file_metadata

    return None
