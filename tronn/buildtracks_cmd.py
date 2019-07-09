"""description: command for plotting regions
"""

import os
import glob
import json
import h5py
import logging

from tronn.preprocess.bed import bin_regions_sharded
from tronn.interpretation.inference import run_inference
from tronn.outlayer import h5_to_bigwig

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
    assert args.data_format == "bed"
    args.bin_width = 200
    args.stride = 50
    args.final_length = 1000
    args.fifo = True
    
    # collect a prediction sample for cross model quantile norm
    args.processed_inputs = False
    if args.model["name"] == "ensemble":
        true_sample_size = args.sample_size
        args.sample_size = 1000
        run_inference(args, warm_start=True)
        args.sample_size = true_sample_size

    # run inference
    inference_files = run_inference(args)

    # make bigwigs
    out_prefix = "{}/{}".format(args.out_dir, args.prefix)
    h5_to_bigwig(inference_files[0], out_prefix, args.chromsizes)
    
    return
