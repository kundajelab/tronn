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

from tronn.preprocess.bed import generate_master_regions
from tronn.preprocess.bed import bin_regions
#from tronn.preprocess.bed import split_bed_to_chrom_bed
from tronn.preprocess.bed import split_bed_to_chrom_bed_parallel
from tronn.preprocess.bed import bin_regions_in_data_dict
from tronn.preprocess.bed import generate_labels
from tronn.preprocess.bed import extract_active_centers

from tronn.preprocess.fasta import generate_examples_in_data_dict
from tronn.preprocess.fasta import generate_one_hot_sequences

from tronn.preprocess.bigwig import generate_signal_vals

from tronn.util.parallelize import setup_multiprocessing_queue
from tronn.util.parallelize import run_in_parallel


def setup_h5_dataset(
        master_bed_file,
        ref_fasta,
        h5_file,
        label_files={},
        signal_files={},
        bin_size=200,
        stride=50,
        final_length=1000,
        reverse_complemented=False,
        onehot_features_key="features",
        tmp_dir="."):
    """given a region file, set up dataset
    conventionally, this is 1 chromosome
    """
    # set up prefix
    prefix = os.path.basename(master_bed_file).split(
        ".narrrowPeak")[0].split(".bed")[0]
    
    # bin the master bed file
    bin_dir = "{}/bin-{}.stride-{}".format(
        tmp_dir, bin_size, stride)
    os.system("mkdir -p {}".format(bin_dir))
    bin_file = "{}/{}.bin-{}.stride-{}.bed.gz".format(
        bin_dir, prefix, bin_size, stride)
    if not os.path.isfile(bin_file):
        bin_regions(master_bed_file, bin_file, bin_size, stride, method="naive")
        
    # generate the one hot sequence encoding
    bin_ext_file = "{}.len-{}.bed.gz".format(
        bin_file.split(".bed")[0], final_length)
    fasta_sequences_file = "{}.fa".format(
        bin_ext_file.split(".bed")[0])
    if not os.path.isfile(h5_file):
        generate_one_hot_sequences(
            bin_file,
            bin_ext_file,
            fasta_sequences_file,
            h5_file,
            onehot_features_key,
            bin_size,
            final_length,
            ref_fasta,
            reverse_complemented)
        
    # extract out the active center again
    bin_active_center_file = "{}.active.bed.gz".format(
        bin_ext_file.split(".bed")[0])
    fasta_sequences_file = "{}.gz".format(fasta_sequences_file)
    extract_active_centers(bin_active_center_file, fasta_sequences_file)
    
    # generate BED annotations on the active center
    print label_files
    for key in label_files.keys():
        generate_labels(
            bin_active_center_file,
            label_files[key],
            key,
            h5_file)

    # generate bigwig annotations on the active center
    for key in signal_files:
        generate_signal_vals(
            bin_active_center_file,
            signal_files[key],
            key,
            h5_file)

    # TODO matrix annotations? ie using DESeq2 normalized matrix here?
        
    return h5_file


def select_flank_negatives(
        positives_bed_file,
        negatives_bed_file,
        bin_size,
        stride,
        num_flank_regions=3):
    """given a positives set, get negatives from the flanks
    """
    with gzip.open(positives_bed_file, "r") as fp:
        with gzip.open(negatives_bed_file, "w") as out:
            for line in fp:
                fields = line.strip().split('\t')
                chrom, start, stop = fields[0], int(fields[1]), int(fields[2])

                # get left flanks
                left_start = max(start - 3*stride, 0)
                left_stop = start

                # get right flanks
                right_start = stop
                right_stop = stop + 3*stride

                # write out
                metadata = "region={0}:{1}-{2};negative_type=flank_neg".format(
                    chrom, start, stop)
                out.write("{0}\t{1}\t{2}\t{3}\n".format(
                    chrom, left_start, left_stop, metadata))
                out.write("{0}\t{1}\t{2}\t{3}\n".format(
                    chrom, right_start, right_stop, metadata))

                # TODO slopbed?
                
    return None


def select_negatives_from_region_set(
        positives_bed_file,
        superset_bed_file,
        neg_region_num,
        negatives_bed_file):
    """given a superset bed file, select a number of regions
    """
    select_negs = (
        "bedtools intersect -v -a {0} -b {1} | "
        "shuf -n {2} | "
        "awk '{{ print $1\"\t\"$2\"\t\"$3 }}' | "
        "gzip -c > {3}").format(
            positives_bed_file,
            superset_bed_file,
            neg_region_num,
            negatives_bed_file)
    print select_negs
    os.system(select_negs)

    return None


def select_random_negatives(
        positives_bed_file,
        negatives_bed_file,
        chrom_sizes,
        neg_region_num,
        tmp_dir="."):
    """select random negatives from across teh genome
    """
    # set up chrom sizes - remove chrM
    tmp_chrom_sizes = "{0}/{1}.tmp".format(tmp_dir, os.path.basename(chrom_sizes))
    setup_chrom_sizes = (
        "cat {0} | grep -v '_' | grep -v 'chrM' > "
        "{1}").format(chrom_sizes, tmp_chrom_sizes)
    print setup_chrom_sizes
    os.system(setup_chrom_sizes)

    # get total region num from positives bed file
    num_positive_regions = 0
    with gzip.open(positives_bed_file, "r") as fp:
        for line in fp:
            num_positive_regions += 1
            
    # select random negatives
    # this uses bedtools shuffle, which maintains distribution of lengths
    # of the regions. as such, it relies on the positives bed file
    # if you are selecting more negatives than there are total regions,
    # you'll run this multiple times to maintain the lengths distribution
    # as much as possible.
    random_neg_left = neg_region_num
    while random_neg_left > 0:
        random_negs_to_select = min(random_neg_left, num_positive_regions)
        select_negs = (
            "bedtools shuffle -i {0} -excl {0} -g {1} | "
            "head -n {2} | "
            "gzip -c >> {3}").format(
                positives_bed_file,
                tmp_chrom_sizes,
                random_negs_to_select,
                negatives_bed_file)
        print select_negs
        os.system(select_negs)
        random_neg_left -= random_negs_to_select
    
    return None


def select_all_negatives(
        positives_bed_file,
        negatives_bed_file,
        chrom_sizes,
        tmp_dir="."):
    """select all negatives outside of the positives
    """
    # set up chrom sizes - remove chrM
    tmp_chrom_sizes = "{0}/{1}.tmp".format(tmp_dir, os.path.basename(chrom_sizes))
    setup_chrom_sizes = (
        "cat {0} | grep -v '_' | grep -v 'chrM' > "
        "{1}").format(chrom_sizes, tmp_chrom_sizes)
    print setup_chrom_sizes
    os.system(setup_chrom_sizes)

    # get complement
    select_negs = (
        "bedtools complement -i {0} -g {1} | "
        "gzip -c > {2}").format(
            positives_bed_file,
            tmp_chrom_sizes,
            negatives_bed_file)
    
    return None



def setup_negatives(
        positives_bed_file,
        dhs_bed_file,
        chrom_sizes,
        bin_size=200,
        stride=50,
        genome_wide=True):
    """wrapper to set up reasonable negative sets 
    for various needs
    """
    # set up
    prefix = positives_bed_file.split(".bed")[0]
    num_positive_regions = 0
    with gzip.open(positives_bed_file, "r") as fp:
        for line in fp:
            num_positive_regions += 1
    
    # flank negatives
    num_flank_regions = 3
    flank_negatives_bed_file = "{}.flank-negatives.bed.gz".format(
        prefix, num_flank_regions)
    select_flank_negatives(
        positives_bed_file,
        flank_negatives_bed_file,
        bin_size,
        stride,
        num_flank_regions=num_flank_regions)

    # DHS negatives
    dhs_negatives_bed_file = "{}.dhs-negatives.bed.gz".format(prefix)
    select_negatives_from_region_set(
        positives_bed_file,
        dhs_bed_file,
        int(num_positive_regions/2.),
        dhs_negatives_bed_file)

    # also infuse random negatives
    random_negatives_bed_file = "{}.random-negatives.bed.gz".format(prefix)
    select_random_negatives(
        positives_bed_file,
        random_negatives_bed_file,
        chrom_sizes,
        int(num_positive_regions/2.))

    # now merge all these to get training negative set
    training_negatives_bed_file = "{}.training-negatives.bed.gz".format(prefix)
    merge_cmd = (
        "zcat {0} {1} {2} | "
        "awk -F '\t' '{{print $1\"\t\"$2\"\t\"$3}}' | "
        "sort -k1,1 -k2,2n | "
        "bedtools merge -i stdin | "
        "gzip -c > {3}").format(
            flank_negatives_bed_file,
            dhs_negatives_bed_file,
            random_negatives_bed_file,
            training_negatives_bed_file)
    print merge_cmd
    os.system(merge_cmd)

    genomewide_negatives_bed_file = "{}.genomewide-negatives.bed.gz".format(prefix)
    if genome_wide:
        # then also set up genomic negatives (for evaluation)
        select_all_negatives(
            positives_bed_file,
            genomewide_negatives_bed_file,
            chrom_sizes)
        
    return training_negatives_bed_file, genomewide_negatives_bed_file


def generate_h5_datasets(
        positives_bed_file,
        ref_fasta,
        chrom_sizes,
        label_files,
        signal_files,
        prefix,
        work_dir,
        bin_size=200,
        stride=50,
        final_length=1000,
        superset_bed_file=None,
        reverse_complemented=False,
        genome_wide=True,
        parallel=24):
    """generate a full h5 dataset
    """
    if False:
        # first select negatives
        training_negatives_bed_file, genomewide_negatives_bed_file = setup_negatives(
            positives_bed_file,
            superset_bed_file,
            chrom_sizes,
            bin_size=bin_size,
            stride=stride)

        # then split all of these into chromosomes
        if genome_wide:
            all_bed_files = [
                positives_bed_file,
                training_negatives_bed_file,
                genomewide_negatives_bed_file]
        else:
            all_bed_files = [
                positives_bed_file,
                training_negatives_bed_file]

    chrom_dir = "{}/by_chrom".format(work_dir)
    #os.system("mkdir -p {}".format(chrom_dir))
    #split_bed_to_chrom_bed_parallel(all_bed_files, chrom_dir)

    # grab all of these and process in parallel
    h5_dir = "{}/h5".format(work_dir)
    os.system("mkdir -p {}".format(h5_dir))
    #chrom_bed_files = glob.glob("{}/*.bed.gz".format(chrom_dir))
    chrom_bed_files = glob.glob("{}/*chrY*.bed.gz".format(chrom_dir))
    logging.info("Found {} bed files".format(chrom_bed_files))
    h5_queue = setup_multiprocessing_queue()
    for bed_file in chrom_bed_files:
        h5_file = "{}/{}.h5".format(
            h5_dir,
            os.path.basename(bed_file).split(".bed")[0])
        process_args = [
            bed_file,
            ref_fasta,
            h5_file,
            label_files,
            signal_files,
            bin_size,
            stride,
            final_length,
            reverse_complemented,
            "features",
            "."]
        h5_queue.put([setup_h5_dataset, process_args])
        
    # run the queue
    run_in_parallel(h5_queue, parallel=parallel, wait=True)

    return None


def generate_variant_h5_dataset():
    """take in a variant file and generate a dataset
    """

    # should be the same as above, just make two adjustments

    # 1) when generating the BED files, include extra metadata in the names column


    # 2) after generating fasta files, go in and adjust the relevant base pair.
    

    return




# OLD CODE





def generate_nn_dataset(
        celltype_master_regions, # master bed file
        univ_master_regions, # superset bed file
        ref_fasta, # annotation
        label_files,
        work_dir,
        prefix,
        neg_region_num=None,
        use_dhs=True,
        use_random=False,
        chrom_sizes=None, # isn't this required now
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
        
        # if still nothing, copy over cell type file
        if not os.path.isfile(completely_neg_file):
            os.system("cp {} {}".format(celltype_master_regions, completely_neg_file))


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

    # TODO separate out top part from here?
        
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


# TODO ideally generate just 1 h5 file with two feature sets (features_ref, features_alt) and then use the keys to switch
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
