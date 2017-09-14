"""Description: preprocessing functions that take standard
bioinformatic formats (BED, FASTA, etc) and format into hdf5
files that are input to deep learning models
"""

import os
import time

from tronn.preprocess import generate_master_regions
from tronn.preprocess import generate_nn_dataset


def run(args):
    """Main function to generate dataset
    """
    start = time.time()
    print "Remember to load both bedtools and ucsc_tools!"
    print "Using {} label sets".format(len(args.labels))

    # generate master bed file
    master_regions_bed = '{0}/{1}.master.bed.gz'.format(args.out_dir, args.prefix)
    if not os.path.isfile(master_regions_bed):
        generate_master_regions(master_regions_bed, args.labels)

    # then run generate nn dataset
    generate_nn_dataset(master_regions_bed,
                        args.annotations["univ_dhs"],
                        args.annotations["ref_fasta"],
                        args.labels,
                        args.out_dir,
                        args.prefix,
                        parallel=args.parallel,
                        neg_region_num=args.univ_neg_num,
                        reverse_complemented=args.rc)

    # TODO utilize the kmerize function from wkm
    if args.kmerize:
        # run kmerize function and save to hdf5 files
        print "kmerize!"
        from tronn.interpretation.wkm import kmerize_parallel
        os.system('mkdir -p {}/data/h5_kmer'.format(args.out_dir))
        kmerize_parallel('{}/data/h5'.format(args.out_dir),
                         '{}/data/h5_kmer'.format(args.out_dir))
        

    end = time.time()
    print "Execution time: {}".format(end - start)
    
    return None
