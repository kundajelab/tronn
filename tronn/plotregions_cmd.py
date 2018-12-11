"""description: command for plotting regions
"""

import logging

from tronn.preprocess.bed import bin_regions_sharded

def run(args):
    """command to plot a region
    """
    # set up
    logger = logging.getLogger(__name__)
    logger.info("plotting region(s)...")

    # check whether region id or bed is defined, set up data loader
    # accordingly
    # need to build a function to create a tiling array
    # or maybe just convert to a BED file and have the bed file reader manage it?
    if args.region_id is not None:
        # put into tmp bed file

        # bin regions -> BED file


        
        
        pass
    elif args.bed_input is not None:

        # bin regions -> BED file
        
        pass
    else:
        raise ValueError, "no appropriate input defined to plot!"
    
    # put the binned BED file into BED dataloader
    
    
    # instantiate the model for inference
    # use an inference stack with JUST the importance scores


    # infer but NOT to hdf5? infer to array



    return
