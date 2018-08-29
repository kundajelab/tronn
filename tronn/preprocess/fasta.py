# description: preprocess FASTA sequences

import os
import re
import sys
import gzip
import glob
import subprocess
import h5py
#import time
import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tronn.util.parallelize import setup_multiprocessing_queue
from tronn.util.parallelize import run_in_parallel

from subprocess import Popen, PIPE, STDOUT


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
        reverse_complemented,
        include_metadata=True,
        include_variant_metadata=False):
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
                    out.write('{0}\t{1}\t{2}\t{3};features={0}:{1}-{2}(+)\t1\t+\n'.format(
                        chrom, new_start, new_stop, metadata))
                    out.write('{0}\t{1}\t{2}\t{3};features={0}:{1}-{2}(-)\t1\t-\n'.format(
                        chrom, new_start, new_stop, metadata))
                else:
                    out.write('{0}\t{1}\t{2}\t{3};features={0}:{1}-{2}\n'.format(
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
        features_hf = hf.create_dataset(
            examples_key,
            (num_bins, 1, final_length, 4),
            dtype="u1")
        if include_metadata:
            metadata_hf = hf.create_dataset(
                'example_metadata',
                (num_bins,), dtype='S1000')
        if include_variant_metadata:
            variant_pos_hf = hf.create_dataset(
                "variant_relative_pos", (num_bins,))
            
        counter = 0
        with open(fasta_sequences, 'rb') as fp:
            for line in fp:
                
                if counter % 50000 == 0:
                    print counter
                
                # convert each sequence into one hot encoding
                # and load into hdf5 file
                sequence = line.strip().split()[1].upper()
                features_hf[counter,:,:,:] = one_hot_encode(sequence)

                # track the region name too.
                if include_metadata:
                    metadata = line.strip().split()[0]
                    metadata_hf[counter] = metadata

                # if variant, save out position (0-index)
                if include_variant_metadata:
                    metadata_dict = dict([
                        field.split("=")
                        for field in line.strip().split()[0].split(";")])
                    example_start = int(metadata_dict["features"].split(":")[1].split("-")[0])
                    variant_abs_pos = int(metadata_dict["snp_pos"].split(":")[1])
                    variant_relative_pos = variant_abs_pos - example_start
                    variant_pos_hf[counter] = variant_relative_pos
                
                counter += 1

    # now gzip the file to keep but reduce space usage
    os.system('gzip {}'.format(fasta_sequences))

    return None


# TODO make sure to close all these cleanly?
def sequence_string_to_onehot_converter(fasta):
    """sets up the pipe to convert a sequence string to onehot
    NOTE: you must unbuffer things to make sure they flow through the pipe!
    """
    # set up input pipe. feed using: pipe_in.write(interval), pipe_in.flush()
    pipe_in = subprocess.Popen(["cat", "-u", "-"], stdout=PIPE, stdin=PIPE, stderr=STDOUT)

    # get fasta sequence
    get_fasta_cmd = "bedtools getfasta -tab -fi {} -bed stdin".format(fasta)
    get_fasta = subprocess.Popen(get_fasta_cmd.split(), stdin=pipe_in.stdout, stdout=PIPE)

    # convert to upper with AWK
    to_upper_cmd = ['awk', '{print toupper($2); system("")}']
    to_upper = subprocess.Popen(to_upper_cmd, stdin=get_fasta.stdout, stdout=PIPE)

    # replace ACGTN with 01234
    sed_cmd = ['sed', "-u", 's/A/0/g; s/C/1/g; s/G/2/g; s/T/3/g; s/N/4/g']
    replace = subprocess.Popen(sed_cmd, stdin=to_upper.stdout, stdout=PIPE)

    # separate all by commas
    split_w_commas_cmd = ["sed", "-u", 's/./,&/g; s/,//']
    pipe_out = subprocess.Popen(split_w_commas_cmd, stdin=replace.stdout, stdout=PIPE)

    # set up close fn
    def close_fn():
        pipe_in.stdin.close()
        pipe_in.wait()
        get_fasta.wait()
        to_upper.wait()
        replace.wait()
        pipe_out.wait()
    
    
    return pipe_in, pipe_out, close_fn


def batch_string_to_onehot(array, pipe_in, pipe_out, batch_array):
    """given a string array, convert each to onehot
    """
    #batch_array = np.zeros((array.shape[0], 1000), dtype=np.uint8)
    
    for i in xrange(array.shape[0]):

        # TODO when featurizing, separate out interval - can save time here
        # get the feature from metadata - is this slow?
        metadata_dict = dict([
            val.split("=")
            for val in array[i][0].strip().split(";")])
        try:
            feature_interval = metadata_dict["features"].replace(":", "\t").replace("-", "\t")
            feature_interval += "\n"
            # pipe in and get out onehot
            pipe_in.stdin.write(feature_interval)
            pipe_in.stdin.flush()

            # check
            sequence = pipe_out.stdout.readline().strip()
            sequence = np.fromstring(sequence, dtype=np.uint8, sep=",") # THIS IS CRUCIAL
        except:
            # TODO fix this so that the information is in the metadata?
            sequence = np.array([4 for j in xrange(1000)], dtype=np.uint8)
            
        batch_array[i,:] = sequence

    return batch_array



def _interval_generator(
        interval_string,
        bin_size,
        stride,
        final_length):
    """helper function to go through an 
    interval and split up into bins
    """
    fields = interval_string.strip().split()
    chrom = fields[0]
    start = int(fields[1])
    stop = int(fields[2])

    extend_len = (final_length - bin_size) / 2
    
    mark = start
    while mark < stop:
        interval_start = mark - extend_len
        interval_stop = mark + bin_size + extend_len

        if interval_start < 0:
            continue
        
        bin_interval = "{}\t{}\t{}\n".format(chrom, interval_start, interval_stop)
        mark += stride
        yield bin_interval


def bed_to_sequence_iterator(
        bed_file,
        fasta,
        bin_size=200,
        stride=50,
        final_length=1000,
        batch_size=1):
    """bed file to batches of sequences
    """
    # set up converter
    converter_in, converter_out = sequence_string_to_onehot_converter(fasta)
    
    # go through bed file and get sequences
    with gzip.open(bed_file) as bed_fp:
        for line in bed_fp:
            line_fields = line.strip().split()
            intervals = _interval_generator(line, bin_size, stride, final_length)

            for interval in intervals:

                converter_in.stdin.write(interval)
                converter_in.stdin.flush()
                sequence = converter_out.stdout.readline().strip()
                sequence = np.fromstring(sequence, dtype=np.uint8, sep=",") # THIS IS CRUCIAL
                sequence = np.expand_dims(sequence, axis=0)
                
                # also set up example metadata coming out if no name
                fields = interval.strip().split()
                interval_metadata = "interval={}:{}-{}".format(fields[0], fields[1], fields[2])
                
                if len(line_fields) > 3:
                    example_metadata = "{};{}".format(line_fields[3], interval_metadata)
                else:
                    example_metadata = interval_metadata
                example_metadata = np.array(example_metadata).reshape((1,1))
            
                # yield
                yield example_metadata, sequence
            
    converter_in.stdout.close()

