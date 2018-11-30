"""Description: preprocessing functions that take standard
bioinformatic formats (BED, FASTA, etc) and format into hdf5
files that are input to deep learning models
"""

import os
import time
import logging

from tronn.preprocess.bed import generate_master_regions
from tronn.preprocess.preprocess import generate_h5_datasets


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

    # print total labels and signals (for debug as needed)
    total_labels = [len(args.labels[key][0]) for key in args.labels.keys()]
    total_signals = [len(args.signals[key][0]) for key in args.signals.keys()]
    logging.info("Using {} label files".format(total_labels))
    logging.info("Using {} signal files".format(total_signals))
    
    # generate master bed file if one is not given
    if args.master_bed_file is None:
        master_file_labels = []
        if len(args.master_label_keys) == 0:
            args.master_label_keys = args.labels.keys()
        for key in args.master_label_keys:
            master_file_labels += args.labels[key][0]
        master_regions_bed = '{0}/{1}.master.bed.gz'.format(
            args.tmp_dir, args.prefix)
        final_master_regions_bed = '{0}/{1}.master.bed.gz'.format(
            args.out_dir, args.prefix)
        if not os.path.isfile(final_master_regions_bed):
            generate_master_regions(master_regions_bed, master_file_labels)
        os.system("cp {} {}".format(master_regions_bed, final_master_regions_bed))
        master_regions_bed = final_master_regions_bed
    else:
        master_regions_bed = args.master_bed_file    
        
    # generate nn dataset
    generate_h5_datasets(
        master_regions_bed,
        args.annotations["ref_fasta"],
        args.annotations["chrom_sizes"],
        args.labels,
        args.signals,
        args.prefix,
        args.out_dir,
        superset_bed_file=args.annotations["univ_dhs"],
        reverse_complemented=args.rc,
        genome_wide=args.genomewide,
        parallel=args.parallel,
        tmp_dir=args.tmp_dir,
        normalize_signals=True)  

    # track how long it took
    end = time.time()
    logging.info("Execution time: {}".format(end - start))
    logging.info("DONE")

    return None

