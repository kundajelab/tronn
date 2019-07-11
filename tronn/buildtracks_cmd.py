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

    # input MUST be BED file of regions to process
    assert args.data_format == "bed"
    
    # force input to be ordered
    logging.info("forcing input to be ordered.")
    args.fifo = True

    # adjust BED file for strides, to make
    # sure outputs don't clash when sequentially merging
    clean_bed_file = "{}/regions.cleaned.bed.gz".format(args.out_dir)
    merge_distance = 2 * args.num_flanks * args.stride
    if args.data_files[0].endswith(".gz"):
        open_cmd = "zcat"
    else:
        open_cmd = "cat"
    merge_cmd = (
        "{} {} | "
        "sort -k1,1 -k2,2n | "
        "bedtools merge -d {} -i stdin | "
        "gzip -c > {}").format(
            open_cmd,
            " ".join(args.data_files),
            merge_distance,
            clean_bed_file)
    args.data_files = [clean_bed_file]
    
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
