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


def normalize_signal_vals(positive_h5_files, h5_files, key, new_key):
    """run some form of normalization to get values between 0 to 1
    """
    # take a random set of positive h5 files to get signal range
    signal_vals = []
    for h5_file in positive_h5_files:
        with h5py.File(h5_file, "r") as hf:
            signal_vals.append(hf[key][:].flatten())
    signal_vals = np.concatenate(signal_vals)

    # asinh
    signal_vals = np.arcsinh(signal_vals)
    
    # rescale according to 0.01 - 0.98 percentiles
    min_val = np.percentile(signal_vals, 1)
    max_val = np.percentile(signal_vals, 90)
    
    # then for each h5 file, adjust to have values between 0 to 1
    for h5_file in h5_files:
        with h5py.File(h5_file, "a") as hf:
            # debug
            del hf[new_key]
            
            signal_vals = hf[key][:]
            # asinh
            signal_vals = np.arcsinh(signal_vals)
            signal_vals = (signal_vals - min_val) / max_val
            
            signal_vals[signal_vals < 0.] = 0
            signal_vals[signal_vals > 1.] = 1.
            
            hf.create_dataset(new_key, data=signal_vals)

    return None
