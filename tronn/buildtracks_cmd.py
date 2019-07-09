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

    # convert to bp resolution bed file
    # this seems wasteful, faster way to do this?
    # but potentially not, if only keeping nonzero positions

    # pad as needed up to the next stride:
    # ex if stride 50 and length 160, pad to 200
    # or snip down? so if 160, then half is 80, and need to go down to 50
    # so then take it down to 100
    
    # reshape by stride
    # ie {N, 10, 100}, np.reshape(N, 10, -1, 50) -> (N, 10, 2, 50)
    
    # then need to figure out how to grab across axis 2
    # if axis_len is 2, then: for loop 
    # take sum of [i,:,axis_len-1] and [i+1,:,axis_len-2]

    # get mean per base pair (bedtools merge)



    return
