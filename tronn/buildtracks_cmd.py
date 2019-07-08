"""description: command for plotting regions
"""

import os
import json
import h5py
import logging

from tronn.preprocess.bed import bin_regions_sharded
from tronn.interpretation.inference import run_inference

def run(args):
    """command to plot a region
    """
    # set up
    logger = logging.getLogger(__name__)
    logger.info("Building bigwigs on regions")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir

    # TODO eventually put this in args
    args.bin_width = 200
    args.stride = 50
    args.final_length = 1000
        
    # check BED file, and shard as needed
    sharded_bed_file = "{}/inputs.sharded.bed.gz".format(args.tmp_dir)
    bin_regions_sharded(
        args.bed_file,
        "{}/{}".format(args.out_dir, args.prefix),
        args.bin_width,
        args.stride,
        args.final_length,
        args.chromsizes)
    args.bed_file = sharded_bed_file
    
    # collect a prediction sample for cross model quantile norm
    if args.model["name"] == "ensemble":
        true_sample_size = args.sample_size
        args.sample_size = 1000
        run_inference(args, warm_start=True)
        args.sample_size = true_sample_size

    # run inference
    inference_files = run_inference(args)

    # convert to bp resolution bed file

    # sort

    # get mean per base pair



    return