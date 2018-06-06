# descriptions: preprocess information from BED files

import os
import h5py
import gzip
import glob
import logging

import numpy as np
import pandas as pd

from tronn.util.parallelize import setup_multiprocessing_queue
from tronn.util.parallelize import run_in_parallel


def generate_master_regions(master_bed_file, bed_files):
    """Generate master regions file
    
    Args:
      master_bed_file: BED file for master regions set
      bed_files: BED files to combine as master region set

    Returns:
      None
    """
    logging.info("generate master regions file")
    assert len(bed_files) > 0
    tmp_master_bed_file = '{}.tmp.bed.gz'.format(
        master_bed_file.split('.bed')[0].split('.narrowPeak')[0])
    
    for i in xrange(len(bed_files)):
        bed_file = bed_files[i]
        if i == 0:
            # for first file copy into master
            transfer = "cp {0} {1}".format(bed_file, master_bed_file)
            print transfer
            os.system(transfer)
        else:
            # merge master with current bed
            merge_bed = (
                "zcat {0} {1} | "
                "awk -F '\t' '{{ print $1\"\t\"$2\"\t\"$3 }}' | "
                "sort -k1,1 -k2,2n | "
                "bedtools merge -i - | "
                "gzip -c > {2}").format(
                    bed_file,
                    master_bed_file,
                    tmp_master_bed_file)
            print merge_bed
            os.system(merge_bed)

            # copy tmp master over to master
            transfer = "cp {0} {1}".format(tmp_master_bed_file, master_bed_file)
            print transfer
            os.system(transfer)
                
    os.system('rm {}'.format(tmp_master_bed_file))

    return None


def split_bed_to_chrom_bed(
        out_dir,
        peak_file,
        prefix):
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
    data_dict = {}
    with gzip.open(peak_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')
            chrom = fields[0]
            chrom_file = '{0}/{1}.{2}.bed.gz'.format(out_dir, prefix, chrom)
            data_dict[chrom] = {"bed": chrom_file}
            with gzip.open(chrom_file, 'a') as out_fp:
                out_fp.write(line)
            
            # Tracking
            if chrom != current_chrom:
                logging.info("Started {}".format(chrom))
                current_chrom = chrom

    return data_dict


def split_bed_to_chrom_bed_parallel(
        bed_files,
        out_dir,
        parallel=12):
    """
    """
    split_queue = setup_multiprocessing_queue()

    for bed_file in bed_files:
        prefix = os.path.basename(bed_file).split(".narrowPeak")[0].split(".bed")[0]
        split_args = [
            out_dir,
            bed_file,
            prefix]
        split_queue.put([split_bed_to_chrom_bed, split_args])

    # run the queue
    run_in_parallel(split_queue, parallel=parallel, wait=True)

    return None


def bin_regions(
        in_file,
        out_file,
        bin_size,
        stride,
        method='naive'):
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
                else:
                    raise Exception
                
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


def extract_active_centers(binned_file, fasta_file):
    """given a fasta file produced in this pipeline, extract out
    the active center 
    """
    bin_count = 0
    with gzip.open(binned_file, 'w') as out:
        with gzip.open(fasta_file, 'r') as fp:
            for line in fp:
                fields = line.strip().split()

                # extract bin only
                metadata = fields[0].split("::")[0]
                metadata_fields = metadata.split(";")
                region_bin = metadata_fields[1].split("=")[1] # active is field 1

                chrom = region_bin.split(":")[0]
                start = int(region_bin.split(':')[1].split('-')[0])
                stop = int(region_bin.split(':')[1].split('-')[1].split('.')[0])
                
                out.write('{}\t{}\t{}\t{}\n'.format(chrom, start, stop, metadata))
                bin_count += 1
    
    return bin_count



def generate_labels(
        bed_file,
        label_bed_files,
        key,
        h5_file,
        method="half_peak",
        tmp_dir="."):
    """given a bed file and label files, intersect 
    and return an array of labels
    """
    os.system("mkdir -p {}".format(tmp_dir))
    num_files = len(label_bed_files)
    
    # get metadata
    file_metadata = [
        "index={0};file={1}".format(
            i, os.path.basename(label_bed_files[i]))
        for i in xrange(len(label_bed_files))]

    # for each label bed file
    for i in xrange(len(label_bed_files)):
        label_bed_file = label_bed_files[i]

        # generate intersect
        #out_tmp_file = "{}/{}_x_{}.bed.gz".format(
        #    tmp_dir,
        #    os.path.basename(bed_file).split(".narrowPeak")[0].split(".bed.gz")[0],
        #    os.path.basename(label_bed_file).split(".narrowPeak")[0].split(".bed.gz")[0])
        out_tmp_file = "{}/{}.intersect.bed.gz".format(
            tmp_dir,
            os.path.basename(label_bed_file).split(".narrowPeak")[0].split(".bed.gz")[0])
        
        # Do the intersection to get a series of 1s and 0s
        if method == 'summit': # Must overlap with summit
            intersect = (
                "zcat {0} | "
                "awk -F '\t' '{{print $1\"\t\"$2+$10\"\t\"$2+$10+1}}' | "
                "bedtools intersect -c -a {1} -b stdin | "
                "gzip -c > "
                "{2}").format(label_bed_file, bed_file, out_tmp_file)
        elif method == 'half_peak': # Bin must be 50% positive
            intersect = (
                "bedtools intersect -f 0.5 -c "
                "-a <(zcat {0}) "
                "-b <(zcat {1}) | "
                "gzip -c > "
                "{2}").format(bed_file, label_bed_file, out_tmp_file)
                
        # TODO(dk) do a counts version (for regression)
        print '{0}: {1}'.format(bed_file, intersect)
        os.system('GREPDB="{0}"; /bin/bash -c "$GREPDB"'.format(intersect))
        
        # Then for each intersect, store it in h5
        bed_labels = pd.read_table(
            out_tmp_file,
            header=None,
            names=['Chr', 'Start', 'Stop', "pos"]) 
        if i == 0:
            num_rows = bed_labels.shape[0]
            labels = np.zeros((num_rows, num_files), dtype=np.bool)
            labels[:,i] = (bed_labels["pos"] >= 1.0).astype(np.bool)
        else:
            labels[:,i] = (bed_labels["pos"] >= 1.0).astype(np.bool)

        # delete tmp file
        os.system("rm {}".format(out_tmp_file))

    # add data into h5 file
    with h5py.File(h5_file, "a") as hf:
        hf.create_dataset(
            key,
            dtype="u1",
            shape=labels.shape)
            #data=np.zeros((num_rows, num_files), dtype=np.bool))
        print "storing in file"
        for i in range(num_rows):
            if i % 10000 == 0:
                print i
            hf[key][i,:] = labels[i,:]
        hf[key].attrs["filenames"] = file_metadata

    # remove tmp dir?
                
    return None
