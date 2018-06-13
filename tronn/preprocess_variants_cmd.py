"""Description: preprocessing functions that take standard
bioinformatic formats (BED, FASTA, etc) and format into hdf5
files that are input to deep learning models
"""

import os
import glob
import time
import logging

from tronn.preprocess.preprocess import generate_variant_h5_dataset


def parse_files(file_list):
    """given an arg string, parse out into a dict
    assumes a format of: key=file1,file2,..;param1=val,param2=val,...
    """
    file_dict = {}
    for file_set in file_list:
        # split out files and params
        key, vals = file_set.split(":")
        files_and_params = vals.split(";")
        files = files_and_params[0].split(",")
        if len(files_and_params) == 1:
            # no params - make empty dict
            params = {}
        else:
            # has params - adjust
            params = files_and_params[1]
            params = dict(
                [param.split("=")
                 for param in params.split(",")])
        file_dict[key] = (files, params)
    
    return file_dict


def run(args):
    """Main function to generate dataset
    """
    # set up
    logger = logging.getLogger(__name__)
    logger.info("Preprocessing data")
    logging.info("Remember to load both bedtools and ucsc_tools!")
    start = time.time()

    # tmp dir
    if args.tmp_dir is None:
        args.tmp_dir = "{}/tmp".format(args.out_dir)
    os.system("mkdir -p {}".format(args.tmp_dir))

    # parse labels and signals
    labels = parse_files(args.labels)
    signals = parse_files(args.signals)
    total_labels = [len(labels[key][0]) for key in labels.keys()]
    total_signals = [len(signals[key][0]) for key in signals.keys()]
    logging.info("Using {} label files".format(total_labels))
    logging.info("Using {} signal files".format(total_signals))
            
    # generate nn dataset
    generate_variant_h5_dataset(
        args.vcf_file,
        args.annotations["ref_fasta"],
        args.annotations["chrom_sizes"],
        labels,
        signals,
        args.prefix,
        args.out_dir,
        reverse_complemented=args.rc,
        tmp_dir=args.tmp_dir)  

    end = time.time()
    print "Execution time: {}".format(end - start)
    print "DONE"
    
    return None

