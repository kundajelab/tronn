"""Description: preprocessing functions that take standard
bioinformatic formats (BED, FASTA, etc) and format into hdf5
files that are input to deep learning models
"""

import os
import glob
import time
import h5py

from tronn.graphs import TronnGraph
from tronn.datalayer import load_data_from_filename_list

from tronn.preprocess import generate_master_regions
from tronn.preprocess import generate_nn_dataset

from tronn.interpretation.kmers import kmerize_parallel
from tronn.interpretation.kmers import kmerize_gpu

from tronn.nets.kmer_nets import gkmerize

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
                        use_dhs=not args.no_dhs_negs,
                        use_random=args.random_negs,
                        chrom_sizes=args.annotations["chrom_sizes"],
                        bin_method="naive" if args.no_flank_negs else "plus_flank_negs",
                        reverse_complemented=args.rc)

    if args.kmerize:
        # run kmerize function and save to hdf5 files
        print "kmerize!"
        os.system('mkdir -p {}/h5_kmer'.format(args.out_dir))
        kmerize_parallel('{}/h5'.format(args.out_dir),
                         '{}/h5_kmer'.format(args.out_dir),
                         parallel=args.parallel)



    quit()
    # TODO utilize the kmerize function from wkm
    if args.kmerize:
        h5_files = glob.glob("{}/h5/*".format(args.out_dir))
        os.system('mkdir -p {}/h5_kmer'.format(args.out_dir))
        batch_size = 64
        
        for h5_file in h5_files:

            h5_kmer_file = "{}/h5_kmer/{}.kmer.h5".format(
                args.out_dir, os.path.basename(h5_file).split(".h5")[0])
            print "generating:", h5_kmer_file
            
            with h5py.File(h5_file, "r") as hf:
                total_examples = hf["example_metadata"].shape[0]
            
            kmerize_graph = TronnGraph(
                {"data":[h5_file]},
                [],
                load_data_from_filename_list,
                gkmerize,
                {"kmer_len":6},
                batch_size,
                shuffle_data=False,
                ordered_num_epochs=2) # this last bit is to make sure we get the last batch

            kmerize_gpu(kmerize_graph, h5_kmer_file, total_examples, batch_size=batch_size)

            quit()
            

    if False:
    #if args.kmerize:
        # run kmerize function and save to hdf5 files
        print "kmerize!"
        os.system('mkdir -p {}/h5_kmer'.format(args.out_dir))
        kmerize_parallel('{}/h5'.format(args.out_dir),
                         '{}/h5_kmer'.format(args.out_dir))

        
    # and here also utilize motif info as input if desired
    
    

    end = time.time()
    print "Execution time: {}".format(end - start)
    
    return None
