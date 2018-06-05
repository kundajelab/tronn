# description: preprocess FASTA sequences

import os
import re
import sys
import gzip
import glob
import subprocess
import h5py
import time
import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tronn.util.parallelize import setup_multiprocessing_queue
from tronn.util.parallelize import run_in_parallel


def one_hot_encode(sequence):
    """One hot encode sequence from string to numpy array

    Args:
      sequence: string of sequence, using [ACGTN] convention

    Returns:
      one hot encoded numpy array of sequence
    """
    # set for python version
    integer_type = np.int8 if sys.version_info[0] == 2 else np.int32

    # set up sequence array
    sequence_length = len(sequence)
    sequence_npy = np.fromstring(sequence, dtype=integer_type)

    # one hot encode
    integer_array = LabelEncoder().fit(
        np.array(('ACGTN',)).view(integer_type)).transform(
            sequence_npy.view(integer_type)).reshape(1, sequence_length)
    one_hot_encoding = OneHotEncoder(
        sparse=False, n_values=5).fit_transform(integer_array)

    return one_hot_encoding.reshape(
        1, 1, sequence_length, 5)[:, :, :, [0, 1, 2, 4]]


def generate_one_hot_sequences(
        binned_file,
        binned_extended_file,
        fasta_sequences,
        examples_file,
        examples_key,
        bin_size,
        final_length,
        ref_fasta_file,
        reverse_complemented):
    """Generate one hot encoded sequence examples from binned regions

    Args:
      binned_file: BED file of binned (equal length) regions
      binned_extended_file: BED output file of extended size regions 
                            (created in this function)
      fasta_sequences: TAB delim output file of sequence strings for each bin
      examples_file: h5 output file where final examples are stored
      bin_size: bin size
      final_length: final length of sequence for example
      ref_fasta_file: reference FASTA file (to get sequences)
      reverse_complemented: Boolean for whether to generate RC sequences too

    Returns:
      None
    """
    logging.info("extending bins for {}...".format(binned_file))
    # First build extended binned file
    extend_length = (final_length - bin_size)/2
    with gzip.open(binned_extended_file, 'w') as out:
        with gzip.open(binned_file, 'rb') as fp:
            for line in fp:
                [chrom, start, stop, metadata] = line.strip().split('\t')
                if int(start) - extend_length < 0:
                    new_start = 0
                    new_stop = final_length
                else:
                    new_start = int(start) - extend_length
                    new_stop = int(stop) + extend_length

                if reverse_complemented:
                    out.write('{0}\t{1}\t{2}\tfeatures={0}:{1}-{2}(+);{3}\t1\t+\n'.format(
                        chrom, new_start, new_stop, metadata))
                    out.write('{0}\t{1}\t{2}\tfeatures={0}:{1}-{2}(-);{3}\t1\t-\n'.format(
                        chrom, new_start, new_stop, metadata))
                else:
                    out.write('{0}\t{1}\t{2}\tfeatures={0}:{1}-{2};{3}\n'.format(
                        chrom, new_start, new_stop, metadata))

    # Then run get fasta to get sequences
    logging.info("getting fasta sequences...")
    get_sequence = ("bedtools getfasta -s -fo {0} -tab -name "
                    "-fi {1} "
                    "-bed {2}").format(fasta_sequences,
                                       ref_fasta_file,
                                       binned_extended_file)
    print get_sequence
    os.system(get_sequence)

    # Then one hot encode and save into hdf5 file
    logging.info("one hot encoding...")
    num_bins = int(subprocess.check_output(
        ['wc','-l', fasta_sequences]).strip().split()[0])
    
    with h5py.File(examples_file, 'a') as hf:
        features_hf = hf.create_dataset(examples_key,
                                         (num_bins, 1, final_length, 4))
        metadata_hf = hf.create_dataset('example_metadata',
                                        (num_bins,), dtype='S1000')
        
        counter = 0
        with open(fasta_sequences, 'rb') as fp:
            for line in fp:
                
                if counter % 50000 == 0:
                    print counter

                sequence = line.strip().split()[1].upper()
                metadata = line.strip().split()[0]
                
                # convert each sequence into one hot encoding
                # and load into hdf5 file
                features_hf[counter,:,:,:] = one_hot_encode(sequence)

                # track the region name too.
                metadata_hf[counter] = metadata
                
                counter += 1

    # now gzip the file to keep but reduce space usage
    os.system('gzip {}'.format(fasta_sequences))

    return None


def generate_examples_in_data_dict(
        data_dict,
        bin_dir,
        bin_ext_dir,
        fasta_dir,
        out_dir,
        prefix,
        bin_size,
        final_length,
        ref_fasta_file,
        reverse_complemented,
        parallel=12):
    """Utilize func_workder to run on chromosomes

    Args:
      bin_dir: directory with binned BED files
      bin_ext_dir: directory to put extended length binned BED files
      fasta_dir: directory to store FASTA sequences for each bin
      out_dir: directory to store h5 files
      prefix: prefix to append to output files
      bin_size: bin size
      final_length: final length of example
      ref_fasta_file: reference FASTA file
      reverse_complemented: boolean on whether to produce RC examples
      parallel: number of parallel processes

    Returns:
      None
    """
    example_queue = setup_multiprocessing_queue()

    for chrom in data_dict.keys():

        # set up filenames?
        chrom_prefix = data_dict[chrom]["bin_bed"].split('/')[-1].split('.bin')[0]
        data_dict["bin_ext_bed"] = '{0}/{1}.extbin.bed.gz'.format(
            bin_ext_dir,
            chrom_prefix)
        data_dict["fasta_seq"] = '{0}/{1}.fa'.format(
            fasta_dir,
            chrom_prefix)
        data_dict["h5_file"] = "{0}/{1}.h5".format(
            out_dir,
            chrom_prefix)
        
        example_args = [
            data_dict[chrom]["bin_bed"],
            data_dict[chrom]["bin_ext_bed"],
            data_dict[chrom]["fasta_seq"],
            data_dict[chrom]["h5_file"],
            final_length,
            ref_fasta_file,
            reverse_complemented]

        example_queue.put(
            [generate_one_hot_sequences, example_args])

    run_in_parallel(example_queue, parallel=parallel, wait=True)

    return data_dict



