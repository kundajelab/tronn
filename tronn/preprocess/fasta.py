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


def sequence_string_to_onehot_converter(fasta):
    """sets up the pipe to convert a sequence string to onehot
    NOTE: you must unbuffer things to make sure they flow through the pipe!

    """
    pipe_in = subprocess.Popen(["cat", "-u", "-"], stdout=PIPE, stdin=PIPE, stderr=STDOUT) # feed by in_pipe.write(interval), in_pipe.flush()
    
    get_fasta_cmd = "bedtools getfasta -tab -fi {} -bed stdin".format(fasta)
    get_fasta = subprocess.Popen(get_fasta_cmd.split(), stdin=pipe_in.stdout, stdout=PIPE)

    #to_upper_cmd = ["tr", "'[:lower:]'", "'[:upper:]'"]
    to_upper_cmd = ['awk', '{print toupper($0); system("")}']
    #to_upper_cmd = ["awk", '{print $0; system("")}']
    to_upper = subprocess.Popen(to_upper_cmd, stdin=get_fasta.stdout, stdout=PIPE)
    
    sed_cmd = ['sed', "-u", 's/A/0/g; s/C/1/g; s/G/2/g; s/T/3/g; s/N/4/g']
    pipe_out = subprocess.Popen(sed_cmd, stdin=to_upper.stdout, stdout=PIPE)

    #unbuffer_cmd = ["awk", '{print $0; system("")}']
    #pipe_out = subprocess.Popen(unbuffer_cmd, stdin=sed.stdout, stdout=PIPE)

    
    return pipe_in, pipe_out


def _map_to_int(sequence):
    """test
    """
    integer_type = np.int8 if sys.version_info[0] == 2 else np.int32
        
    sequence_length = len(sequence)
    sequence_npy = np.fromstring(sequence, dtype=integer_type)

    # one hot encode
    #print sequence
    integer_array = LabelEncoder().fit(
        np.array(('ACGTN',)).view(integer_type)).transform(
            sequence_npy.view(integer_type)).reshape(1, sequence_length)
    #print integer_array
    #quit()
    
    return integer_array


def batch_string_to_onehot(array, pipe_in, pipe_out):
    """given a string array, convert each to onehot
    """
    onehot_batch = []
    for i in xrange(array.shape[0]):
        
        # get the feature
        metadata_dict = dict([
            val.split("=") 
            for val in array[i][0].strip().split(";")])
        #try:
        feature_interval = metadata_dict["features"].replace(":", "\t").replace("-", "\t")
        feature_interval += "\n"
        # pipe in and get out onehot
        pipe_in.stdin.write(feature_interval)
        pipe_in.stdin.flush()

        # check
        sequence = pipe_out.stdout.readline().strip().split('\t')[1] #.upper().split("")
        print sequence
        quit()
        #except:
        # TODO fix this so that the information is in the metadata
            #sequence = "".join(["N" for i in xrange(1000)])
            #sequence = ["N" for i in xrange(1000)]
            
        #sequence = one_hot_encode(sequence)#.astype(np.float32)
        #sequence = _map_to_int(sequence)
        sequence = np.expand_dims(np.array(sequence), axis=0)
        onehot_batch.append(sequence)
    onehot_batch = np.concatenate(onehot_batch, axis=0)

    return onehot_batch



def bed_to_sequence_iterator(bed_file, fasta, batch_size=1):
    """bed file to batches of sequences
    """
    # set up pipe
    p = subprocess.Popen(["cat", "-"], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    get_fasta_cmd = "bedtools getfasta -tab -fi {} -bed stdin".format(fasta)
    print get_fasta_cmd.split()
    getfasta_p = subprocess.Popen(get_fasta_cmd.split(), stdin=p.stdout, stdout=PIPE)
    
    # return inidividuals (batch in another function? or use six)
    with gzip.open(bed_file) as bed_fp:
        for line in bed_fp:
            fields = line.strip().split()
            
            # set up interval and get from fasta
            interval = "{}\t{}\t{}\n".format(fields[0], fields[1], fields[2])
            p.stdin.write(interval)
            p.stdin.flush()
            sequence = getfasta_p.stdout.readline().strip().split('\t')[1].upper()
            sequence = one_hot_encode(sequence).astype(np.float32)
            
            # also set up example metadata coming out if no name
            if len(fields) > 3:
                example_metadata = fields[3]
            else:
                example_metadata = "{}:{}-{}".format(fields[0], fields[1], fields[2])
            example_metadata = np.array(example_metadata).reshape((1,1))
            
            # yield
            yield example_metadata, sequence
            
    p.stdin.close()
            
    return


#iterator = bed_to_sequences("test.bed.gz", "/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa")
#print iterator.next()
#print iterator.next()
