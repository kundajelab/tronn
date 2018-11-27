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
        tmp_dir=".",
        params={}):
    """given a region set (BED file), calculate
    average signal in each region from the bigwigs
    and save out the array to h5.
    """
    os.system("mkdir -p {}".format(tmp_dir))
    num_files = len(bigwig_files)
    logging.info("Found {} bigwig files as annotations".format(num_files))

    # sample?
    sample_around_center = int(params.get("window", 0))
    
    # get metadata
    file_metadata = [
        "index={0};file={1}".format(
            i, os.path.basename(bigwig_files[i]))
        for i in xrange(len(bigwig_files))]

    # for each bigwig file
    for i in xrange(len(bigwig_files)):
        bigwig_file = bigwig_files[i]

        # generate signals        
        out_tmp_file = "{}/{}.signal.tmp".format(
            tmp_dir,
            os.path.basename(bigwig_file))
        if sample_around_center == 0:
            get_average_signal = (
                "bigWigAverageOverBed {} {} {}").format(
                    bigwig_file, bed_file, out_tmp_file)
            mean_col_name = "mean0"
        else:
            get_average_signal = (
                "bigWigAverageOverBed -sampleAroundCenter={} {} {} {}").format(
                    sample_around_center,
                    bigwig_file,
                    bed_file,
                    out_tmp_file)
            mean_col_name = "mean" # TODO adjust here if using mean of just covered bp
            
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
                hf[key][:,i] = bigwig_average_signals[mean_col_name]
        else:
            with h5py.File(h5_file, "a") as hf:
                hf[key][:,i] = bigwig_average_signals[mean_col_name]
                
        # delete tmp file
        os.system("rm {}".format(out_tmp_file))

    # add in bigwig files as attr on h5
    with h5py.File(h5_file, "a") as hf:
        hf[key].attrs["filenames"] = file_metadata

    return None


def normalize_signal_vals(positive_h5_files, h5_files, key, new_key):
    """run some form of normalization to get values between 0 to 1
    """
    # extract all positives and concat
    full_signal_vals = []
    for h5_file in positive_h5_files:
        with h5py.File(h5_file, "r") as hf:
            full_signal_vals.append(hf[key][:]) # {N, task}
    full_signal_vals = np.concatenate(full_signal_vals, axis=0) # {N, task}

    # set up sorted mean and originals to interpolate
    full_signal_vals = np.sort(full_signal_vals, axis=0)
    full_signal_val_means = np.mean(full_signal_vals, axis=1)
    
    # now use ranks to adjust values
    for h5_file in h5_files:
        with h5py.File(h5_file, "a") as hf:
            print h5_file
            signal_vals = hf[key][:]
            
            # interpolate for each task
            for i in xrange(signal_vals.shape[1]):
                signal_vals[:,i] = np.interp(
                    signal_vals[:,i],
                    full_signal_vals[:,i],
                    full_signal_val_means)

            if False:
                # DEPRECATE LATER
                
                # sort and get average
                new_signal_vals = np.mean(
                    np.sort(signal_vals, axis=0), axis=1) # {N}
            
                # get indices sorted
                sorted_indices = np.argsort(signal_vals, axis=0)
            
                # adjust values using the sorted indices to make life easy
                signal_vals[sorted_indices, np.arange(signal_vals.shape[1])] = new_signal_vals[
                    :, np.newaxis]

            # and then log2
            signal_vals = np.log2(signal_vals)
            signal_vals[signal_vals < 0] = 0 # anything that is not FC > 1 should be zero
            signal_vals[~np.isfinite(signal_vals)] = 0

            # and save out
            if hf.get(new_key) is not None:
                del hf[new_key]
            
            hf.create_dataset(new_key, data=signal_vals)

    return None
