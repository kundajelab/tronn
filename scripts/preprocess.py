# functions useful for processing into NN format

import os
import sys
import gzip
import glob
import subprocess
import random
import h5py
import multiprocessing
import Queue
import operator
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import ggr_plotting


# =====================================================================
# GENERAL
# =====================================================================

def func_worker(queue):
    """Takes a tuple of (function, args) from queue and runs them

    queue: multiprocessing Queue where each elem is (function, args)
    """
    while not queue.empty():
        time.sleep(5.0) # to give queue time to get set up
        try:
            [func, args] = queue.get(timeout=0.1)
        except Queue.empty:
            time.sleep(5.0)
            continue
        func(*args) # run the function with appropriate arguments
        time.sleep(5.0)
        
    return None


def run_in_parallel(queue, parallel=12):
    """Takes a filled queue and runs in parallel

    queue: multiprocessing Queue where each elem is (function, args)
    parallel: how many to run in parallel
    """

    # Multiprocessing code
    pids = []
    for i in xrange(parallel):
        pid = os.fork()
        if pid == 0:
            func_worker(queue)
            os._exit(0)
        else:
            pids.append(pid)
            
    for pid in pids:
        os.waitpid(pid,0)

    return None


def generate_chrom_files(out_dir, peak_file, prefix):
    """Splits a gzipped peak file into its various chromosomes

    out_dir: output directory to put chrom files
    peak_file: BED file of form chr, start, stop (tab delim)
    prefix: desired prefix on chrom files
    """
    print "splitting into chromosomes..."

    current_chrom = ''
    with gzip.open(peak_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')
            chrom = fields[0]
            chrom_file = '{0}/{1}_{2}.bed.gz'.format(out_dir, prefix, chrom)
            with gzip.open(chrom_file, 'a') as out_fp:
                out_fp.write(line)
            
            # Tracking
            if chrom != current_chrom:
                print "Started {}".format(chrom)
                current_chrom = chrom

    return None


# =====================================================================
# BINNING
# =====================================================================

def bin_regions(in_file, out_file, bin_size, stride, method='plus_flank_negs'):
    """Bin regions based on bin size and stride

    in_file: BED file to bin
    out_file: name of output binned file
    bin_size: length of the bin 
    stride: how many base pairs to jump for next window
    method: how to do binning (add flanks or not)
    """
    print "binning regions for {}...".format(in_file)

    # Bin the files
    with gzip.open(out_file, 'w') as out:
        with gzip.open(in_file, 'rb') as fp:
            for line in fp:
                fields = line.strip().split('\t')
                chrom, start, stop = fields[0], int(fields[1]), int(fields[2])

                if method == 'naive':
                    # Just go from start of region to end of region
                    mark = start
                elif method == 'plus_flank_negs':
                    # Add 3 flanks to either side
                    start -= 3 * stride
                    stop += 3 * stride
                    mark = start

                while mark < stop:
                    # write out bins
                    out.write('{0}\t{1}\t{2}\n'.format(chrom, 
                                                       mark, 
                                                       mark + bin_size))
                    mark += stride

    return None


def bin_regions_chrom(in_dir, out_dir, prefix, bin_size, stride, parallel=12):
    """Utilize func_workder to run on chromosomes

    in_dir: directory with BED files to bin
    out_dir: directory to put binned files
    prefix: desired prefix on the output files
    bin_size: bin size
    stride: how many base pairs to jump for next bin
    parallel: how many processes in parallel
    """

    bin_queue = multiprocessing.Queue()

    # First find chromosome files
    chrom_files = glob.glob('{0}/*.bed.gz'.format(in_dir))
    print 'Found {} chromosome files'.format(len(chrom_files))

    # Then for each file, give the function and put in queue
    for chrom_file in chrom_files:
        chrom_prefix = chrom_file.split('/')[-1].split('.bed.gz')[0]
        bin_args = [chrom_file,
                    '{0}/{1}_binned.bed.gz'.format(out_dir, chrom_prefix),
                    bin_size,
                    stride]
        bin_queue.put([bin_regions, bin_args])

    # run the queue
    run_in_parallel(bin_queue, parallel=parallel)

    return None


# =====================================================================
# FEATURE GENERATION
# =====================================================================

def one_hot_encode(sequence):
    """One hot encode sequence from string to numpy array

    sequence: string of sequence, using [ACGTN] convention
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

    return one_hot_encoding.reshape(1, 1, sequence_length, 5)[:, :, :, [0, 1, 2, 4]]


def generate_examples(binned_file, binned_extended_file, fasta_sequences,
                      examples_file, bin_size, final_length, ref_fasta_file,
                      reverse_complemented):
    """Generate one hot encoded sequence examples from binned regions

    binned_file: BED file of binned (equal length) regions
    binned_extended_file: BED output file of extended size regions (created in this function)
    fasta_sequences: TAB delim output file of sequence strings for each bin
    examples_file: h5 output file where final examples are stored
    bin_size: bin size
    final_length: final length of sequence for example
    ref_fasta_file: reference FASTA file (to get sequences)
    reverse_complemented: Boolean for whether to generate RC sequences too
    """

    # First build extended binned file
    print "extending bins for {}...".format(binned_file)
    extend_length = (final_length - bin_size)/2
    with gzip.open(binned_extended_file, 'w') as out:
        with gzip.open(binned_file, 'rb') as fp:
            for line in fp:
                [chrom, start, stop] = line.strip().split('\t')
                if int(start) - extend_length < 0:
                    new_start = 0
                    new_stop = final_length
                else:
                    new_start = int(start) - extend_length
                    new_stop = int(stop) + extend_length

                if reverse_complemented:
                    out.write('{0}\t{1}\t{2}\t{0}:{1}-{2}\t1\t+\n'.format(chrom, new_start, new_stop))
                    out.write('{0}\t{1}\t{2}\t{0}:{1}-{2}\t1\t-\n'.format(chrom, new_start, new_stop))
                else:
                    out.write('{0}\t{1}\t{2}\n'.format(chrom, new_start, new_stop))

    # Then run get fasta to get sequences
    print "getting fasta sequences..."
    get_sequence = ("bedtools getfasta -s -fo {0} -tab "
                    "-fi {1} "
                    "-bed {2}").format(fasta_sequences,
                                       ref_fasta_file,
                                       binned_extended_file)
    print get_sequence
    os.system(get_sequence)

    # Then one hot encode and save into hdf5 file
    print "one hot encoding..."
    num_bins = int(subprocess.check_output(['wc','-l',
                                            fasta_sequences]).strip().split()[0])
    
    with h5py.File(examples_file, 'a') as hf:
        all_examples = hf.create_dataset('features',
                                         (num_bins, 1, final_length, 4))
        region_info = hf.create_dataset('regions',
                                        (num_bins,), dtype='S100')
        
        counter = 0
        with open(fasta_sequences, 'rb') as fp:
            for line in fp:
                
                if counter % 50000 == 0:
                    print counter

                sequence = line.strip().split()[1].upper()
                region = line.strip().split()[0]
                
                # convert each sequence into one hot encoding
                # and load into hdf5 file
                all_examples[counter,:,:,:] = one_hot_encode(sequence)

                # track the region name too.
                region_info[counter] = region

                counter += 1

    # TO DO now gzip the file
    os.system('gzip {}'.format(fasta_sequences))

    return None


def generate_examples_chrom(bin_dir, bin_ext_dir, fasta_dir, out_dir, prefix,
                            bin_size, final_length, ref_fasta_file,
                            reverse_complemented,
                            parallel=12):
    """Utilize func_workder to run on chromosomes

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
    """

    example_queue = multiprocessing.Queue()

    # First find chromosome files
    chrom_files = glob.glob('{0}/*.bed.gz'.format(bin_dir))
    print 'Found {} chromosome files'.format(len(chrom_files))

    # Then for each file, give the function and run
    for chrom_file in chrom_files:
        chrom_prefix = chrom_file.split('/')[-1].split('_bin')[0]
        ext_binned_file = '{0}/{1}_extbin.bed.gz'.format(bin_ext_dir,
                                                         chrom_prefix)
        regions_fasta_file = '{0}/{1}.fa'.format(fasta_dir,
                                                 chrom_prefix)
        examples_file = '{0}/{1}.h5'.format(out_dir,
                                            chrom_prefix)

        examples_args = [chrom_file, ext_binned_file, regions_fasta_file,
                         examples_file, bin_size, final_length, ref_fasta_file, reverse_complemented]

        if not os.path.isfile(examples_file):
            example_queue.put([generate_examples, examples_args])

    run_in_parallel(example_queue, parallel=parallel)

    return None


# =====================================================================
# LABEL GENERATION
# =====================================================================

def generate_labels(bin_dir, intersect_dir, prefix, label_files, fasta_file,
                    h5_ml_file, bin_size, final_length, method='half_peak'):
    """Generate labels
    
    bin_dir: directory where to put new binned files for labeling
    intersect_dir: directory to store temp intersect files from bedtools
    prefix: prefix for output files
    label_files: list of BED files to use for multitask labeling
    fasta_file: tab delim output from examples to use as regions
    h5_ml_file: where to store the labels
    bin_size: bin size
    final_length: final length of example
    method: how to count a label as positive
    """
    print "generating labels..."

    # Get relevant peak files of interest
    peak_list = label_files
    num_tasks = len(peak_list)
    label_set_names = [ os.path.basename(file_name).split('.narrowPeak')[0].split('.bed')[0] for file_name in peak_list ]
    print "found {} peak sets for labels...".format(len(peak_list))

    # Then generate new short bins for labeling
    fasta_prefix = fasta_file.split('/')[-1].split('.fa')[0]
    binned_file = '{0}/{1}_activecenter.bed.gz'.format(bin_dir, fasta_prefix)
    flank_length = (final_length - bin_size) / 2
    bin_count = 0
    with gzip.open(binned_file, 'w') as out:
        with gzip.open(fasta_file, 'r') as fp:
            for line in fp:
                fields = line.strip().split()
                chrom = fields[0].split(':')[0]
                start = int(fields[0].split(':')[1].split('-')[0])
                active_start = start + flank_length
                stop = int(fields[0].split(':')[1].split('-')[1].split('(')[0])
                active_stop = stop - flank_length
                out.write('{}\t{}\t{}\n'.format(chrom, active_start, active_stop))
                bin_count += 1
    
    # then for each, generate labels
    with h5py.File(h5_ml_file, 'a') as hf:

        if not 'labels' in hf:
            all_labels = hf.create_dataset('labels', (bin_count, num_tasks))
        else:
            all_labels = hf['labels']

        if not 'label_names' in hf:
            label_names = hf.create_dataset('label_names', (num_tasks,), dtype='S1000')
        hf['label_names'][:,] = label_set_names

        # initialize in-memory tmp label array
        tmp_labels_all = np.zeros((bin_count, num_tasks))

        # go through peak files to intersect and get positives/negatives
        counter = 0
        for peak_file in peak_list:

            peak_file_name = os.path.basename(peak_file).split('.narrowPeak')[0].split('bed.gz')[0]
            intersect_file_name = '{0}/{1}_{2}_intersect.bed.gz'.format(intersect_dir, prefix, peak_file_name)

            # Do the intersection to get a series of 1s and 0s
            if method == 'summit': # Must overlap with summit
                intersect = (
                    "zcat {0} | "
                    "awk -F '\t' '{{print $1\"\t\"$2+$10\"\t\"$2+$10+1}}' | "
                    "bedtools intersect -c -a {1} -b stdin | "
                    "gzip -c > "
                    "{2}").format(peak_file, binned_file, intersect_file_name)
            elif method == 'half_peak': # Bin must be 50% positive
                intersect = (
                    "bedtools intersect -f 0.5 -c -a <(zcat {0}) -b <(zcat {1}) | "
                    "gzip -c > "
                    "{2}").format(binned_file, peak_file, intersect_file_name)
            print '{0}: {1}'.format(prefix, intersect)
            os.system('GREPDB="{0}"; /bin/bash -c "$GREPDB"'.format(intersect))
            
            # Then for each intersect, store it in a numpy array
            # Use the indices to select the right parts and save to hdf5 file
            task_labels = pd.read_table(intersect_file_name,
                                        header=None,
                                        names=['Chr',
                                               'Start',
                                               'Stop',
                                               prefix]) # If this fails, use bedtools 2.23.0
            tmp_labels_all[:, counter] = (task_labels[prefix] >= 1.0).astype(int)

            # delete the intersect file and move counter up
            os.system('rm {}'.format(intersect_file_name))
            counter += 1
            
        # now store all values at once
        print "storing in file"
        for i in range(bin_count):
            if i % 10000 == 0:
                print i
            all_labels[i,:] = tmp_labels_all[i,:]

    return None


def generate_labels_chrom(bin_ext_dir, intersect_dir, prefix, label_files,
                          fasta_dir, out_dir, bin_size, final_length,
                          parallel=12):
    """Utilize func_worker to run on chromosomes

    bin_ext_dir: directory to store extended bin files
    intersect_dir: directory to put temp intersect files
    prefix: output prefix
    label_files: list of peak files to use for labeling
    fasta_dir: directory of tab delim FASTA sequence examples
    out_dir: h5 file directory
    bin_size: bin_size
    final_length: final length
    parallel: nunmber to run in parallel
    """

    label_queue = multiprocessing.Queue()

    # First find chromosome files
    chrom_files = glob.glob('{0}/*.fa.gz'.format(fasta_dir))
    print 'Found {} chromosome files'.format(len(chrom_files))

    # Then for each file, give the function and run
    for chrom_file in chrom_files:

        print chrom_file
        
        #if not ('chr1.fa' in chrom_file):
        #    continue

        chrom_prefix = chrom_file.split('/')[-1].split('.fa.gz')[0]
        regions_fasta_file = '{0}/{1}.fa.gz'.format(fasta_dir,
                                                    chrom_prefix)
        h5_ml_file = '{0}/{1}.h5'.format(out_dir,
                                            chrom_prefix)

        labels_args = [bin_ext_dir, intersect_dir, chrom_prefix, label_files, 
                       regions_fasta_file, h5_ml_file, bin_size, final_length]

        label_queue.put([generate_labels, labels_args])

    run_in_parallel(label_queue, parallel=parallel)

    return None


def generate_nn_dataset(celltype_master_regions,
                        univ_master_regions,
                        ref_fasta,
                        label_files,
                        work_dir,
                        prefix,
                        neg_region_num=200000,
                        bin_size=200,
                        stride=50,
                        final_length=1000,
                        softmax=False,
                        reverse_complemented=False):
    '''
    Convenient wrapper to run all relevant functions
    requires: ucsc_tools, bedtools
    '''

    # Select random set of negatives from univ master regions
    completely_neg_file = '{0}/{1}.negatives.bed.gz'.format(work_dir, prefix)
    select_negs = ("bedtools intersect -v -a {0} -b {1} | "
                   "shuf -n {2} | "
                   "awk '{{ print $1\"\t\"$2\"\t\"$3 }}' | "
                   "gzip -c > {3}").format(univ_master_regions,
                                           celltype_master_regions,
                                           neg_region_num,
                                           completely_neg_file)
    if not os.path.isfile(completely_neg_file):
        print select_negs
        os.system(select_negs)

    # merge in to have a file of positive and negative regions
    final_master = '{0}/{1}.master.ml.bed.gz'.format(work_dir, prefix)
    merge_pos_neg = ("zcat {0} {1} | "
                     "awk -F '\t' '{{ print $1\"\t\"$2\"\t\"$3 }}' | "
                     "sort -k1,1 -k2,2n | "
                     "bedtools merge -i stdin | "
                     "gzip -c > {2}").format(celltype_master_regions,
                                             completely_neg_file,
                                             final_master)
    if not os.path.isfile(final_master):
        print merge_pos_neg
        os.system(merge_pos_neg)


    # split into chromosomes, everything below done by chromosome
    chrom_master_dir = '{}/master_by_chrom'.format(work_dir)
    if not os.path.isfile('{0}/{1}_chrY.bed.gz'.format(chrom_master_dir, prefix)):
        os.system('mkdir -p {}'.format(chrom_master_dir))
        generate_chrom_files(chrom_master_dir, final_master, prefix)

    # bin the files
    # NOTE: this does not check for chromosome lengths and WILL contain inappropriate regions
    bin_dir = '{}/binned'.format(work_dir)
    if not os.path.isfile('{0}/{1}_chrY_binned.bed.gz'.format(bin_dir, prefix)):
        os.system('mkdir -p {}'.format(bin_dir))
        bin_regions_chrom(chrom_master_dir, bin_dir, prefix, bin_size, stride)

    # generate one-hot encoding sequence files (examples) and then labels
    regions_fasta_dir = '{}/regions_fasta'.format(work_dir)
    bin_ext_dir = '{}/bin_ext'.format(work_dir)
    intersect_dir = '{}/intersect'.format(work_dir)
    chrom_hdf5_dir = '{}/h5'.format(work_dir)
    
    if not os.path.isfile('{0}/{1}_chrY.h5'.format(chrom_hdf5_dir, prefix)):
        os.system('mkdir -p {}'.format(chrom_hdf5_dir))
        os.system('mkdir -p {}'.format(regions_fasta_dir))
        os.system('mkdir -p {}'.format(bin_ext_dir))
        os.system('mkdir -p {}'.format(intersect_dir))
        generate_examples_chrom(bin_dir, bin_ext_dir, regions_fasta_dir, chrom_hdf5_dir, prefix, bin_size, final_length, ref_fasta, reverse_complemented)
        generate_labels_chrom(bin_ext_dir, intersect_dir, prefix, label_files, regions_fasta_dir, chrom_hdf5_dir, bin_size, final_length)


    return None


def generate_interpretation_dataset(h5_in_dir, task_num, out_h5):
    '''
    This takes in a directory of data files, checks for specific tasks and saves out those
    files to a new hdf5 file that you can throw into the model for interpretation
    '''

    h5_files = glob.glob('{}/*.h5'.format(h5_in_dir))

    # first need to figure out how many examples match, to make 
    # a dataset of the right size in an hdf5 format
    total_pos = 0
    for i in range(len(h5_files)):

        with h5py.File(h5_files[i], 'r') as hf:
            total_pos += int(np.sum(hf['labels'][:,task_num]))

            if i == 0:
                num_tasks = hf['labels'].shape[1]
                final_length = hf['features'].shape[2]

    print total_pos, num_tasks, final_length

    # Now save these to a file
    with h5py.File(out_h5, 'w') as out:
        # create datasets
        features = out.create_dataset('features',
                                         (total_pos, 1, final_length, 4))
        labels = out.create_dataset('labels', (total_pos, num_tasks))
        label_names = out.create_dataset('label_names', (num_tasks,), dtype='S1000')
        regions = out.create_dataset('regions',
                                        (total_pos,),
                                        dtype='S100')

        start_idx = 0
        stop_idx = 0
        for i in range(len(h5_files)):
            print i
            with h5py.File(h5_files[i], 'r') as hf:

                if i == 0:
                    label_names[:] = hf['label_names'][:]

                pos_indices = np.nonzero(hf['labels'][:,task_num])[0]
                if len(pos_indices) == 0:
                    continue

                stop_idx = start_idx + len(pos_indices)

                # Save things into their places
                features[start_idx:stop_idx,:,:,:] = hf['features'][pos_indices,:,:,:]
                labels[start_idx:stop_idx,:] = hf['labels'][pos_indices,:]
                regions[start_idx:stop_idx] = hf['regions'][list(pos_indices)]

                start_idx += len(pos_indices)

        assert(stop_idx == total_pos)

    return None

def visualize_sequence(h5_file, num_task, out_dir):
    '''
    quick wrapper for Av viz tools
    '''

    with h5py.File(h5_file, 'r') as hf:
        
        sequences = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000]

        for sequence_idx in sequences:
            sequence = hf['importance_{}'.format(num_task)][sequence_idx,:,:,:]
            sequence_name = hf['region'][sequence_idx, 0].split('(')[0]
            ggr_plotting.plot_weights(sequence, '{0}/task_{1}.example_{2}.viz_all.png'.format(out_dir, num_task, sequence_idx))
        # quit()

        # for sequence_idx in sequences:
        #     start = 0
        #     skip = 100
        #     length = 200
        #     while start < 1000:
        #         sequence = hf['importance_{}'.format(num_task)][sequence_idx,:,start:(start+length),:]
        #         ggr_plotting.plot_weights(sequence, '{0}/task_{1}.example_{2}.viz_{3}.png'.format(out_dir, num_task, sequence_idx, start))
        #         start += skip

    return None


def region_generator(h5_file, start_idx, stop_idx, num_task):
    '''
    Build a generator to easily extract regions from an h5 file of importance scores
    '''

    with h5py.File(h5_file, 'r') as hf:
        regions = hf.get('regions')
        sequence_idx = 0
        region_idx = start_idx
        current_chrom = 'NA'
        current_region_start = 0
        current_region_stop = 0
        current_sequence = np.zeros((1,1))

        while region_idx < stop_idx:
            sequence = hf['importance_{}'.format(num_task)][sequence_idx,:,:,:]
            sequence = np.squeeze(sequence)
            sequence = sequence.transpose(1, 0)

            region = regions[sequence_idx, 0]
            chrom = region.split(':')[0]
            region_start = int(region.split(':')[1].split('-')[0])
            region_stop = int(region.split(':')[1].split('-')[1].split('(')[0])

            if (current_chrom == chrom) and (region_start < current_region_stop) and (region_stop > current_region_stop):
                # add on to current region
                offset = region_start - current_region_start

                # concat zeros
                current_sequence = np.concatenate((current_sequence, np.zeros((4, region_stop - current_region_stop))), axis=1)
                #normalizer = np.concatenate((normalizer, np.zeros((region_stop - current_region_stop,))))
                # add new data on top
                current_sequence[:,offset:] += sequence
                #normalizer[offset:,] += 1e-8 * np.ones((region_stop - region_start,))

                current_region_stop = region_stop

            else:
                # we're on a new region
                if current_sequence.shape[0] == 4:
                    region_name = '{0}:{1}-{2}'.format(current_chrom, current_region_start, current_region_stop)
                    yield current_sequence, region_name, region_idx
                    region_idx += 1

                # save new region
                current_chrom = chrom
                current_region_start = region_start
                current_region_stop = region_stop
                current_sequence = sequence
                #normalizer = 1e-8 * np.ones((region_stop - region_start,))

            sequence_idx += 1


def visualize_region(h5_file, num_task, out_dir):
    '''
    Aggregate examples into a region and visualize
    '''

    for sequence, name, idx in region_generator(h5_file, 0, 9, num_task):
        print sequence.shape
        ggr_plotting.plot_weights(sequence, '{0}/task_{1}.region_{2}.{3}.png'.format(out_dir, num_task, idx, name.replace(':', '-')))

    return None


# TODO function to extract regions of high impt and count kmers
def onehot_to_string(array):
    '''
    Go from numpy array to letters
    '''

    assert (array.shape[0] == 4)

    seq_list = ['']*array.shape[1]
    for i in range(array.shape[1]):
        # Convert into letter
        if array[0,i] != 0:
            seq_list[i] = 'A'
        elif array[1,i] != 0:
            seq_list[i] = 'C'
        elif array[2,i] != 0:
            seq_list[i] = 'G'
        elif array[3,i] != 0:
            seq_list[i] = 'T'
        else: 
            seq_list[i] = 'N'

    return ''.join(seq_list)


def build_kmer_clusters(kmer_dict, kmer_counts):
    '''
    Mutate kmer at position
    '''

    bases = ['A', 'C', 'G', 'T']

    # TODO sort the dictionary to get top kmers
    kmer_dict_sorted = sorted(kmer_dict.items(), key=operator.itemgetter(1), reverse=True)

    current_kmer_idx = 0
    current_kmer_community = 0
    kmer_communities = {}
    kmer_community_scores = {}
    seen_kmers = {}

    while kmer_dict_sorted[current_kmer_idx][1] > 0.1:
        # do aggregation
        current_kmer, current_kmer_score = kmer_dict_sorted[current_kmer_idx]

        try:
            seen = seen_kmers[current_kmer]
            current_kmer_idx += 1
            continue

        except:
            community = [current_kmer]
            seen_kmers[current_kmer] = 1

            # mutate the kmer (insert or mutate) and add if score is positive.
            for pos in range(len(current_kmer)):
                for base in bases:
                    if current_kmer[pos] != base:
                        new_kmer = list(current_kmer)
                        new_kmer[pos] = base
                        new_kmer = ''.join(new_kmer)
                        try:
                            seen = seen_kmers[new_kmer]
                            continue
                        except:
                            try:
                                new_kmer_score = kmer_dict[new_kmer]
                                if new_kmer_score > 0:
                                    current_kmer_score += new_kmer_score
                                    community.append(new_kmer)
                                    seen_kmers[new_kmer] = 1
                            except:
                                continue

                for base in bases:
                    new_kmer = base + str(current_kmer[:-1])
                    try:
                        seen = seen_kmers[new_kmer]
                        continue
                    except:
                        try:
                            new_kmer_score = kmer_dict[new_kmer]
                            if new_kmer_score > 0:
                                current_kmer_score += new_kmer_score
                                community.append(new_kmer)
                                seen_kmers[new_kmer] = 1
                        except:
                            continue

                for base in bases:
                    new_kmer = str(current_kmer[1:]) + base
                    try:
                        seen = seen_kmers[new_kmer]
                        continue
                    except:
                        try:
                            new_kmer_score = kmer_dict[new_kmer]
                            if new_kmer_score > 0:
                                current_kmer_score += new_kmer_score
                                community.append(new_kmer)
                                seen_kmers[new_kmer] = 1
                        except:
                            continue

            kmer_communities[current_kmer_community] = community
            kmer_community_scores[current_kmer_community] = current_kmer_score
            current_kmer_community += 1

    scores_sorted = sorted(kmer_community_scores.items(), key=operator.itemgetter(1), reverse=True)

    import ipdb
    ipdb.set_trace()

    return None

def extract_kmer_counts(h5_file, num_task, out_dir, k_val=8):
    '''
    Extracts kmers with high importance, adds them up
    Keeps track of kmer's contexts.
    '''

    print "running kmer counts"

    kmer_dict = {}
    kmer_counts = {}

    with h5py.File(h5_file, 'r') as hf:
        dataset_name = 'importance_{}'.format(num_task)
        print hf[dataset_name].shape

        for i in range(hf[dataset_name].shape[0]):
            if i % 100 == 0:
                print i

            sequence = np.squeeze(hf[dataset_name][i,:,:,:])
            sequence = sequence.transpose(1, 0)

            for j in range(hf[dataset_name].shape[2] - k_val):
                kmer = sequence[:,j:j+k_val]
                kmer_string = onehot_to_string(kmer)


                kmer_importance = np.sum(kmer[kmer > 0])

                try:
                    kmer_total = kmer_dict[kmer_string]
                    kmer_dict[kmer_string] = kmer_total + kmer_importance
                    kmer_count = kmer_counts[kmer_string]
                    kmer_counts[kmer_string] = kmer_count + 1
                except:
                    kmer_dict[kmer_string] = kmer_importance
                    kmer_counts[kmer_string] = 1

            if i % 100 == 0:
                kmer_dict_sorted = sorted(kmer_dict.items(), key=operator.itemgetter(1), reverse=True)
                print kmer_dict_sorted[0:30]
    
    return kmer_dict, kmer_counts


def extract_kmer_counts_maxpoints(h5_file, num_task, out_dir, k_val=8, num_max=3):
    '''
    Extracts kmers with high importance, adds them up
    Keeps track of kmer's contexts.
    '''

    print "running kmer counts"

    kmer_dict = {}
    kmer_counts = {}

    with h5py.File(h5_file, 'r') as hf:
        dataset_name = 'importance_{}'.format(num_task)
        print hf[dataset_name].shape

        for i in range(hf[dataset_name].shape[0]):
            if i % 100 == 0:
                print i

            sequence = np.squeeze(hf[dataset_name][i,:,:,:])
            sequence = sequence.transpose(1, 0)
            importances = np.sum(sequence, axis=0)
            importances_by_index = np.argsort(importances).tolist()
            importances_by_index.reverse()

            # Now get kmers
            for j in range(num_max):
                center = importances_by_index[j]
                extend = k_val / 2
                kmer = sequence[:,center-extend:center+extend]
                kmer_string = onehot_to_string(kmer)

                if ("TGA" in kmer_string) or ("TCA" in kmer_string):
                    continue
                    
                kmer_importance = np.sum(kmer[kmer > 0])

                try:
                    kmer_total = kmer_dict[kmer_string]
                    kmer_dict[kmer_string] = kmer_total + kmer_importance
                    kmer_count = kmer_counts[kmer_string]
                    kmer_counts[kmer_string] = kmer_count + 1
                except:
                    kmer_dict[kmer_string] = kmer_importance
                    kmer_counts[kmer_string] = 1

            if i % 100 == 0:
                kmer_dict_sorted = sorted(kmer_dict.items(), key=operator.itemgetter(1), reverse=True)
                print kmer_dict_sorted[0:30]
    
    return kmer_dict, kmer_counts



