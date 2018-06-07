"""Description: preprocessing functions that take standard
bioinformatic formats (BED, FASTA, etc) and format into hdf5
files that are input to deep learning models
"""

import os
import glob
import time
import logging

from tronn.preprocess.bed import generate_master_regions
from tronn.preprocess.preprocess import generate_h5_datasets


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
    logging.info("Using {} label sets".format(len(labels["labels"][0])))
    
    # generate master bed file
    master_regions_bed = '{0}/{1}.master.bed.gz'.format(
        args.out_dir, args.prefix)
    if not os.path.isfile(master_regions_bed):
        generate_master_regions(master_regions_bed, args.labels)

    # generate nn dataset
    generate_h5_datasets(
        master_regions_bed,
        args.annotations["ref_fasta"],
        args.annotations["chrom_sizes"],
        labels,
        signals,
        args.prefix,
        args.out_dir,
        superset_bed_file=args.annotations["univ_dhs"],
        reverse_complemented=args.rc,
        genome_wide=args.genomewide,
        parallel=args.parallel,
        tmp_dir=args.tmp_dir)  

    end = time.time()
    print "Execution time: {}".format(end - start)
    print "DONE"

    quit()

    if not args.genomewide:
        # then run generate nn dataset
        generate_nn_dataset(
            master_regions_bed,
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

    else:
        # then run generate nn dataset
        generate_nn_dataset(
            master_regions_bed,
            None,
            args.annotations["ref_fasta"],
            args.labels,
            args.out_dir,
            args.prefix,
            parallel=args.parallel,
            neg_region_num=None,
            use_dhs=False,
            use_random=False,
            chrom_sizes=args.annotations["chrom_sizes"],
            bin_method="naive",
            reverse_complemented=args.rc)

        # and generate the negatives also
        generate_genomewide_negatives_dataset(
            master_regions_bed,
            args.annotations["ref_fasta"],
            args.labels,
            args.out_dir,
            "{}.negs".format(args.prefix),
            parallel=args.parallel,
            chrom_sizes=args.annotations["chrom_sizes"])

    quit()

    

    if False:
        from tronn.graphs import TronnGraph
        from tronn.datalayer import load_data_from_filename_list
        
        from tronn.interpretation.kmers import kmerize_parallel
        from tronn.interpretation.kmers import kmerize_gpu
        
        from tronn.nets.kmer_nets import gkmerize

        # run kmerize function and save to hdf5 files
        print "kmerize!"
        os.system('mkdir -p {}/h5_kmer'.format(args.out_dir))
        kmerize_parallel('{}/h5'.format(args.out_dir),
                         '{}/h5_kmer'.format(args.out_dir),
                         parallel=args.parallel)

    # TODO utilize the kmerize function from wkm
    if False:
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

    if False:
    #if args.kmerize:
        # run kmerize function and save to hdf5 files
        print "kmerize!"
        os.system('mkdir -p {}/h5_kmer'.format(args.out_dir))
        kmerize_parallel('{}/h5'.format(args.out_dir),
                         '{}/h5_kmer'.format(args.out_dir))

    end = time.time()
    print "Execution time: {}".format(end - start)
    print "DONE"
    
    return None


def run_variants(args):
    """Run variants 
    """
    generate_variant_datasets(
        args.variant_file,
        args.annotations["ref_fasta"],
        args.out_dir,
        args.prefix)
    
    return None
