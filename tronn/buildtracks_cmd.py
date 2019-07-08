"""description: command for plotting regions
"""

import os
import glob
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
    
    # check BED files, and bin/shard as needed
    assert args.data_format == "bed"
    for data_file in args.data_files:
        bin_regions_sharded(
            data_file,
            "{}/{}".format(args.tmp_dir, os.path.basename(data_file).split(".bed")[0]),
            args.bin_width,
            args.stride,
            args.final_length,
            args.chromsizes)
    args.data_files = sorted(glob.glob("{}/*filt.bed.gz".format(
        args.tmp_dir)))
    logging.info(";".join(args.data_files))
    
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
