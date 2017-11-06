"""Description: preprocessing functions that take standard
bioinformatic formats (BED, FASTA, etc) and format into hdf5
files that are input to deep learning models
"""

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



# =====================================================================
# Split to chromosomes
# =====================================================================

def split_bed_to_chrom_bed(out_dir, peak_file, prefix):
    """Splits a gzipped peak file into its various chromosomes

    Args:
      out_dir: output directory to put chrom files
      peak_file: BED file of form chr, start, stop (tab delim)
                 Does not need to be sorted
      prefix: desired prefix on chrom files

    Returns:
      None
    """
    logging.info("Splitting BED file into chromosome BED files...")
    assert os.path.splitext(peak_file)[1] == '.gz'
    
    current_chrom = ''
    with gzip.open(peak_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')
            chrom = fields[0]
            chrom_file = '{0}/{1}.{2}.bed.gz'.format(out_dir, prefix, chrom)
            with gzip.open(chrom_file, 'a') as out_fp:
                out_fp.write(line)
            
            # Tracking
            if chrom != current_chrom:
                logging.info("Started {}".format(chrom))
                current_chrom = chrom

    return None


# =====================================================================
# Binning
# =====================================================================

def bin_regions(
        in_file,
        out_file,
        bin_size,
        stride,
        method='plus_flank_negs'):
    """Bin regions based on bin size and stride

    Args:
      in_file: BED file to bin
      out_file: name of output binned file
      bin_size: length of the bin 
      stride: how many base pairs to jump for next window
      method: how to do binning (add flanks or not)

    Returns:
      None
    """
    logging.info("binning regions for {}...".format(in_file))
    assert os.path.splitext(in_file)[1] == '.gz'
    assert os.path.splitext(out_file)[1] == '.gz'

    # Open input and output files and bin regions
    with gzip.open(out_file, 'w') as out:
        with gzip.open(in_file, 'rb') as fp:
            for line in fp:
                fields = line.strip().split('\t')
                chrom, start, stop = fields[0], int(fields[1]), int(fields[2])

                if method == 'naive':
                    # Just go from start of region to end of region
                    mark = start
                    adjusted_stop = stop
                elif method == 'plus_flank_negs':
                    # Add 3 flanks to either side
                    mark = max(start - 3 * stride, 0)
                    adjusted_stop = stop + 3 * stride
                # add other binning strategies as needed here

                while mark < adjusted_stop:
                    # write out bins
                    out.write((
                        "{0}\t{1}\t{2}\t"
                        "active={0}:{1}-{2};"
                        "region={0}:{3}-{4}\n").format(
                            chrom, 
                            mark, 
                            mark + bin_size,
                            start,
                            stop))
                    mark += stride

    return None


def bin_regions_chrom(
        in_dir,
        out_dir,
        prefix,
        bin_size,
        stride,
        bin_method,
        parallel=12):
    """Utilize func_workder to run on chromosomes
    
    Args:
      in_dir: directory with BED files to bin
      out_dir: directory to put binned files
      prefix: desired prefix on the output files
      bin_size: bin size
      stride: how many base pairs to jump for next bin
      parallel: how many processes in parallel

    Returns:
      None
    """
    bin_queue = setup_multiprocessing_queue()

    # First find chromosome files
    chrom_files = glob.glob('{0}/*.bed.gz'.format(in_dir))
    logging.info('Found {} chromosome files'.format(len(chrom_files)))

    # Then for each file, give the function and put in queue
    for chrom_file in chrom_files:
        chrom_prefix = chrom_file.split('/')[-1].split('.bed.gz')[0]
        bin_args = [chrom_file,
                    '{0}/{1}.binned.bed.gz'.format(out_dir, chrom_prefix),
                    bin_size,
                    stride,
                    bin_method]
        bin_queue.put([bin_regions, bin_args])

    # run the queue
    run_in_parallel(bin_queue, parallel=parallel, wait=True)

    return None


# =====================================================================
# Feature generation
# =====================================================================

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


def generate_examples(
        binned_file,
        binned_extended_file,
        fasta_sequences,
        examples_file,
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
        features_hf = hf.create_dataset('features',
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

    # TO DO now gzip the file
    os.system('gzip {}'.format(fasta_sequences))

    return None


def generate_examples_chrom(
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

    # First find chromosome files
    chrom_files = glob.glob('{0}/*.bed.gz'.format(bin_dir))
    print 'Found {} chromosome files'.format(len(chrom_files))

    # Then for each file, give the function and run
    for chrom_file in chrom_files:
        chrom_prefix = chrom_file.split('/')[-1].split('.bin')[0]
        ext_binned_file = '{0}/{1}.extbin.bed.gz'.format(bin_ext_dir,
                                                         chrom_prefix)
        regions_fasta_file = '{0}/{1}.fa'.format(fasta_dir,
                                                 chrom_prefix)
        examples_file = '{0}/{1}.h5'.format(out_dir,
                                            chrom_prefix)

        examples_args = [chrom_file, ext_binned_file, regions_fasta_file,
                         examples_file, bin_size, final_length, ref_fasta_file,
                         reverse_complemented]

        if not os.path.isfile(examples_file):
            example_queue.put([generate_examples, examples_args])

    run_in_parallel(example_queue, parallel=parallel, wait=True)

    return None


# =====================================================================
# Label generation
# =====================================================================

def generate_labels(
        bin_dir,
        intersect_dir,
        prefix,
        label_files,
        fasta_file,
        h5_ml_file,
        method='half_peak',
        remove_substring_regex="ggr\.|GGR\.|\.filt"): 
    """Generate labels
    
    Args:
      bin_dir: directory where to put new binned files for labeling
      intersect_dir: directory to store temp intersect files from bedtools
      prefix: prefix for output files
      label_files: list of BED files to use for multitask labeling
      fasta_file: tab delim output from examples to use as regions
      h5_ml_file: where to store the labels
      bin_size: bin size
      final_length: final length of example
      method: how to count a label as positive

    Returns:
      None
    """
    print "generating labels..."

    # Get relevant peak files of interest
    peak_list = label_files
    num_tasks = len(peak_list)
    label_set_names = [
        "index={0};description={1}".format(
            i, re.sub(
                remove_substring_regex, "",
                os.path.basename(peak_list[i]).split('.narrowPeak')[0].split('.bed')[0]))
        for i in range(len(peak_list)) ]
    print "found {} peak sets for labels...".format(len(peak_list))

    # Then generate new short bins for labeling
    fasta_prefix = fasta_file.split('/')[-1].split('.fa')[0]
    binned_file = '{0}/{1}.activecenter.bed.gz'.format(bin_dir, fasta_prefix)
    bin_count = 0
    with gzip.open(binned_file, 'w') as out:
        with gzip.open(fasta_file, 'r') as fp:
            for line in fp:
                fields = line.strip().split()

                # extract bin only
                metadata_fields = fields[0].split("::")[0].split(";")
                region_bin = metadata_fields[1].split("=")[1] # active is field 1

                chrom = region_bin.split(":")[0]
                start = int(region_bin.split(':')[1].split('-')[0])
                stop = int(region_bin.split(':')[1].split('-')[1].split('.')[0])
                
                out.write('{}\t{}\t{}\n'.format(chrom, start, stop))
                bin_count += 1
    
    # then for each, generate labels
    with h5py.File(h5_ml_file, 'a') as hf:

        if not 'labels' in hf:
            all_labels = hf.create_dataset('labels', (bin_count, num_tasks))
        else:
            all_labels = hf['labels']

        if not 'label_metadata' in hf:
            label_names = hf.create_dataset('label_metadata', (num_tasks,),
                                            dtype='S1000')

        hf['label_metadata'][:,] = label_set_names

        # initialize in-memory tmp label array
        tmp_labels_all = np.zeros((bin_count, num_tasks))

        # go through peak files to intersect and get positives/negatives
        counter = 0
        for peak_file in peak_list:

            peak_file_name = os.path.basename(peak_file).split('.narrowPeak')[0].split('bed.gz')[0]
            intersect_file_name = '{0}/{1}_{2}_intersect.bed.gz'.format(
                intersect_dir, prefix, peak_file_name)

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
                    "bedtools intersect -f 0.5 -c "
                    "-a <(zcat {0}) "
                    "-b <(zcat {1}) | "
                    "gzip -c > "
                    "{2}").format(binned_file, peak_file, intersect_file_name)
                
            # TODO(dk) do a counts version (for regression)

                
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


def generate_labels_chrom(
        bin_ext_dir,
        intersect_dir,
        prefix,
        label_files,
        fasta_dir,
        out_dir,
        parallel=12):
    """Utilize func_worker to run on chromosomes

    Args:
      bin_ext_dir: directory to store extended bin files
      intersect_dir: directory to put temp intersect files
      prefix: output prefix
      label_files: list of peak files to use for labeling
      fasta_dir: directory of tab delim FASTA sequence examples
      out_dir: h5 file directory
      bin_size: bin_size
      final_length: final length
      parallel: nunmber to run in parallel

    Returns: 
      None
    """
    label_queue = setup_multiprocessing_queue()

    # First find chromosome files
    chrom_files = glob.glob('{0}/*.fa.gz'.format(fasta_dir))
    print 'Found {} chromosome files'.format(len(chrom_files))

    # Then for each file, give the function and run
    for chrom_file in chrom_files:
        print chrom_file

        chrom_prefix = chrom_file.split('/')[-1].split('.fa.gz')[0]
        regions_fasta_file = '{0}/{1}.fa.gz'.format(fasta_dir,
                                                    chrom_prefix)
        h5_ml_file = '{0}/{1}.h5'.format(out_dir,
                                            chrom_prefix)

        labels_args = [bin_ext_dir, intersect_dir, chrom_prefix, label_files, 
                       regions_fasta_file, h5_ml_file]

        label_queue.put([generate_labels, labels_args])

    run_in_parallel(label_queue, parallel=parallel, wait=True)

    return None

# =====================================================================
# Dataset generation
# =====================================================================

def generate_nn_dataset(
        celltype_master_regions,
        univ_master_regions,
        ref_fasta,
        label_files,
        work_dir,
        prefix,
        neg_region_num=None,
        use_dhs=True,
        use_random=False,
        chrom_sizes=None,
        bin_size=200,
        bin_method='plus_flank_negs',
        stride=50,
        final_length=1000,
        parallel=12,
        softmax=False,
        reverse_complemented=False):
    """Convenient wrapper to run all relevant functions
    requires: ucsc_tools, bedtools
    """
    os.system('mkdir -p {}'.format(work_dir))
    tmp_dir = "{}/tmp".format(work_dir)
    os.system("mkdir -p {}".format(tmp_dir))

    # set up negatives
    completely_neg_file = '{0}/{1}.negatives.bed.gz'.format(tmp_dir, prefix)
    if not os.path.isfile(completely_neg_file):

        # count regions in master regions
        total_master_regions = 0
        with gzip.open(celltype_master_regions) as fp:
            for line in fp:
                total_master_regions += 1
        
        # count regions in dhs regions
        total_dhs_regions = 0
        with gzip.open(univ_master_regions) as fp:
            for line in fp:
                total_dhs_regions += 1
        
        # determine settings for negative region total
        if neg_region_num is None:
            neg_region_num = total_master_regions

        # determine division of negatives
        if use_dhs and use_random:
            # split evenly
            neg_region_num = int(neg_region_num / 2.)

        # select negs from DHS regions
        if use_dhs:
            select_negs = (
                "bedtools intersect -v -a {0} -b {1} | "
                "shuf -n {2} | "
                "awk '{{ print $1\"\t\"$2\"\t\"$3 }}' | "
                "gzip -c >> {3}").format(
                    univ_master_regions,
                    celltype_master_regions,
                    neg_region_num,
                    completely_neg_file)
            print select_negs
            os.system(select_negs)

        # select negs randomly from genome
        if use_random:
            assert chrom_sizes is not None
            tmp_chrom_sizes = "{0}/{1}.tmp".format(tmp_dir, os.path.basename(chrom_sizes))
            setup_chrom_sizes = (
                "cat {0} | grep -v '_' | grep -v 'chrM' > "
                "{1}").format(chrom_sizes, tmp_chrom_sizes)
            print setup_chrom_sizes
            os.system(setup_chrom_sizes)
            random_neg_left = neg_region_num
            while random_neg_left > 0:
                random_negs_to_select = min(random_neg_left, total_master_regions)
                select_negs = (
                    "bedtools shuffle -i {0} -excl {0} -g {1} | "
                    "head -n {2} | "
                    "gzip -c >> {3}").format(
                        celltype_master_regions,
                        tmp_chrom_sizes,
                        random_negs_to_select,
                        completely_neg_file)
                print select_negs
                os.system(select_negs)
                random_neg_left -= random_negs_to_select

    # merge in to have a file of positive and negative regions
    final_master = '{0}/{1}.master.ml.bed.gz'.format(tmp_dir, prefix)
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
    chrom_master_dir = '{}/master_by_chrom'.format(tmp_dir)
    if not os.path.isfile('{0}/{1}.chrY.bed.gz'.format(chrom_master_dir, prefix)):
        os.system('mkdir -p {}'.format(chrom_master_dir))
        split_bed_to_chrom_bed(chrom_master_dir, final_master, prefix)

    # bin the files
    # NOTE: this does not check for chromosome lengths and WILL contain inappropriate regions
    bin_dir = '{}/binned'.format(tmp_dir)
    if not os.path.isfile('{0}/{1}.chrY.binned.bed.gz'.format(bin_dir, prefix)):
        os.system('mkdir -p {}'.format(bin_dir))
        bin_regions_chrom(chrom_master_dir, bin_dir, prefix,
                          bin_size, stride, bin_method, parallel=parallel)

    # generate one-hot encoding sequence files (examples) and then labels
    regions_fasta_dir = '{}/regions_fasta'.format(tmp_dir)
    bin_ext_dir = '{}/bin_ext'.format(tmp_dir)
    intersect_dir = '{}/intersect'.format(tmp_dir)
    chrom_hdf5_dir = '{}/h5'.format(work_dir)

    # now run example generation and label generation
    if not os.path.isfile('{0}/{1}.chrY.h5'.format(chrom_hdf5_dir, prefix)):
        os.system('mkdir -p {}'.format(chrom_hdf5_dir))
        os.system('mkdir -p {}'.format(regions_fasta_dir))
        os.system('mkdir -p {}'.format(bin_ext_dir))
        generate_examples_chrom(
            bin_dir,
            bin_ext_dir,
            regions_fasta_dir,
            chrom_hdf5_dir,
            prefix,
            bin_size,
            final_length,
            ref_fasta,
            reverse_complemented,
            parallel=parallel)
        os.system('mkdir -p {}'.format(intersect_dir))
        generate_labels_chrom(
            bin_ext_dir,
            intersect_dir,
            prefix,
            label_files,
            regions_fasta_dir,
            chrom_hdf5_dir,
            parallel=parallel)
        os.system("rm -r {}".format(intersect_dir))

    return '{}/h5'.format(work_dir)


def generate_variant_datasets(variant_file, ref_fasta_file, out_dir, prefix, seq_length=1000):
    """Creates 2 hdf5 files, one that has allele1 and the other with allele2
    Then, when prediction (in order), should produce two files that you can put
    together to get the fold changes easily
    """
    snp_chr_column = 4
    snp_pos_column = 5
    snp_name_column = 6
    snp_dist_column = 26
    allele1_column = 7
    allele2_column = 8

    # set up folders
    os.system("mkdir -p {0}/{1}.allele1 {0}/{1}.allele2".format(out_dir, prefix))
    
    # start from SNP file that has allele1 and allele2, as well as SNP distance from center of DHS peak
    allele1_bed = "{0}/{1}.allele1/{1}.allele1.bed.gz".format(out_dir, prefix)
    allele2_bed = "{0}/{1}.allele2/{1}.allele2.bed.gz".format(out_dir, prefix)
    with gzip.open(allele1_bed, "w") as out1:
        with gzip.open(allele2_bed, "w") as out2:
    
            with open(variant_file, "r") as fp:
                for line in fp:
                    if line.startswith("SNP"):
                        continue
                    
                    fields = line.strip().split('\t')
                    
                    # make a bed file line
                    if fields[snp_dist_column] == "NA":
                        dist = 0
                    else:
                        dist = int(fields[snp_dist_column])

                    chrom = fields[snp_chr_column]
                    snp_coord = int(fields[snp_pos_column])
                    center_coord = snp_coord + dist
                    start_coord = center_coord - int(seq_length/2)
                    stop_coord = center_coord + int(seq_length/2)
                    snp_pos_from_start = snp_coord - start_coord

                    # ignore any positions that are beyond 1000
                    if snp_pos_from_start <= 10:
                        continue
                    if snp_pos_from_start > 990:
                        continue

                    out1_line = "{0}\t{1}\t{2}\tfeatures={0}:{1}-{2};snp_name={3};snp_pos={4};allele={5}\n".format(
                        chrom,
                        start_coord,
                        stop_coord,
                        fields[snp_name_column],
                        snp_pos_from_start,
                        fields[allele1_column])
                    out1.write(out1_line)
                    
                    out2_line = "{0}\t{1}\t{2}\tfeatures={0}:{1}-{2};snp_name={3};snp_pos={4};allele={5}\n".format(
                        chrom,
                        start_coord,
                        stop_coord,
                        fields[snp_name_column],
                        snp_pos_from_start,
                        fields[allele2_column])
                    out2.write(out2_line)

    # now shift 5 bp on either side
    new_allele_beds = []
    for allele_bed in [allele1_bed, allele2_bed]:
        new_bed = "{}.shifts.bed.gz".format(allele_bed.split(".bed")[0])
        new_allele_beds.append(new_bed)
        with gzip.open(new_bed, "w") as out:
            with gzip.open(allele_bed) as fp:
                for line in fp:
                    fields = line.strip().split('\t')

                    # shift up and down by 5
                    for shift in xrange(-4,5):
                        metadata_fields = fields[3].split(";")
                        metadata_fields[2] = "snp_pos={}".format(int(metadata_fields[2].split("=")[1])-shift)
                        out.write("{}\t{}\t{}\t{}\n".format(
                            fields[0],
                            int(fields[1])+shift,
                            int(fields[2])+shift,
                            ";".join(metadata_fields)))

    # run getfasta
    fasta_files = []
    for allele_bed in new_allele_beds:
        fasta_sequences = "{}.fasta".format(allele_bed.split(".bed")[0])
        fasta_files.append(fasta_sequences)
        get_sequence = ("bedtools getfasta -s -fo {0} -tab -name "
                        "-fi {1} "
                        "-bed {2}").format(fasta_sequences,
                                           ref_fasta_file,
                                           allele_bed)
        print get_sequence
        os.system(get_sequence)
    
    # onehot encode, throw into hdf5 file
    h5_files = []
    for fasta_file in fasta_files:
        h5_file = "{}.h5".format(fasta_file.split(".fasta")[0])
        h5_files.append(h5_file)
        
        num_bins = int(subprocess.check_output(
            ['wc','-l', fasta_file]).strip().split()[0])
        
        with h5py.File(h5_file, "w") as hf:
            features_hf = hf.create_dataset('features',
                                            (num_bins, 1, seq_length, 4))
            metadata_hf = hf.create_dataset('example_metadata',
                                            (num_bins,), dtype='S1000')
        
            counter = 0
            with open(fasta_file, 'rb') as fp:
                for line in fp:
                
                    if counter % 50000 == 0:
                        print counter

                    sequence = line.strip().split()[1].upper()
                    metadata = line.strip().split()[0]
                    
                    allele_pos = int(metadata.split(";")[2].split("=")[1]) - 1 # 0 index
                    allele_basepair = metadata.split("::")[0].split(";")[3].split("=")[1]

                    # sequence with new basepair
                    sequence = sequence[:allele_pos] + allele_basepair + sequence[allele_pos+1:]

                    # convert each sequence into one hot encoding
                    # and load into hdf5 file
                    try:
                        features_hf[counter,:,:,:] = one_hot_encode(sequence)
                    except:
                        import ipdb
                        ipdb.set_trace()

                    # track the region name too.
                    metadata_hf[counter] = metadata
                
                    counter += 1

            # generate fake labels
            labels_hf = hf.create_dataset("labels",
                                          (num_bins, 1))
            labels_hf[:] = np.ones((num_bins, 1))
                    
    return

def generate_ordered_single_file_nn_dataset(
        celltype_master_regions,
        ref_fasta,
        label_files,
        work_dir,
        prefix,
        bin_size=200,
        bin_method='plus_flank_negs',
        stride=50,
        final_length=1000,
        parallel=12,
        reverse_complemented=False):
    """Convenient wrapper to run all relevant functions
    requires: ucsc_tools, bedtools
    NOTE: this version is used to produce a single hdf5 file with all data
    """
    os.system('mkdir -p {}'.format(work_dir))
    tmp_dir = "{}/tmp".format(work_dir)
    os.system("mkdir -p {}".format(tmp_dir))

    # bin the files
    # NOTE: this does not check for chromosome lengths and WILL contain inappropriate regions
    bin_dir = '{}/binned'.format(tmp_dir)
    binned_file = "{0}/{1}.binned.bed.gz".format(bin_dir, prefix)
    if not os.path.isfile(binned_file):
        os.system('mkdir -p {}'.format(bin_dir))
        bin_regions(celltype_master_regions, binned_file, bin_size, stride,
            method='plus_flank_negs')

    # generate one-hot encoding sequence files (examples) and then labels
    regions_fasta_dir = '{}/regions_fasta'.format(tmp_dir)
    bin_ext_dir = '{}/bin_ext'.format(tmp_dir)
    intersect_dir = '{}/intersect'.format(tmp_dir)
    chrom_hdf5_dir = '{}/h5'.format(work_dir)

    # now run example generation and label generation
    out_h5_file = "{0}/{1}.ordered.h5".format(chrom_hdf5_dir, prefix)
    if not os.path.isfile(out_h5_file):
        os.system('mkdir -p {}'.format(chrom_hdf5_dir))
        os.system('mkdir -p {}'.format(regions_fasta_dir))
        os.system('mkdir -p {}'.format(bin_ext_dir))
        binned_extended_file = "{0}/{1}.extbin.bed.gz".format(
            bin_ext_dir, os.path.basename(binned_file).split(".bed")[0])
        fasta_sequences = "{0}/{1}.fa".format(
            regions_fasta_dir,
            "{}.ordered".format(os.path.basename(binned_file).split(".binned")[0]))
        generate_examples(
            binned_file,
            binned_extended_file,
            fasta_sequences,
            out_h5_file,
            bin_size,
            final_length,
            ref_fasta,
            reverse_complemented)
        os.system('mkdir -p {}'.format(intersect_dir))
        generate_labels_chrom(
            bin_ext_dir,
            intersect_dir,
            prefix,
            label_files,
            regions_fasta_dir,
            chrom_hdf5_dir,
            parallel=parallel)
        os.system("rm -r {}".format(intersect_dir))

    return '{}/h5'.format(work_dir)



def generate_master_regions(out_file, label_peak_files):
    """Generate master regions file
    
    Args:
      out_file: BED file for master regions set
      label_peak_files: BED files to use as labels

    Returns:
      None
    """
    assert len(label_peak_files) > 0
    tmp_master = '{}.tmp.bed.gz'.format(out_file.split('.bed')[0].split('.narrowPeak')[0])
    for i in range(len(label_peak_files)):
        label_peak_file = label_peak_files[i]
        if i == 0:
            # for first file copy into master
            transfer = "cp {0} {1}".format(label_peak_file, out_file)
            print transfer
            os.system(transfer)
        else:
            # merge master with current bed
            merge_bed = ("zcat {0} {1} | "
                         "awk -F '\t' '{{ print $1\"\t\"$2\"\t\"$3 }}' | "
                         "sort -k1,1 -k2,2n | "
                         "bedtools merge -i - | "
                         "gzip -c > {2}").format(label_peak_file, out_file, tmp_master)
            print merge_bed
            os.system(merge_bed)

            # copy tmp master over to master
            transfer = "cp {0} {1}".format(tmp_master, out_file)
            print transfer
            os.system(transfer)
                
    os.system('rm {}'.format(tmp_master))

    return
