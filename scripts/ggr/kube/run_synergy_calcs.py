#!/usr/bin/env python

import os
import sys
import glob

import pandas as pd

def main():
    """run synergy calculations <- separate bc need R and such
    """
    # params
    INFER_DIR = sys.argv[1]

    # get synergy files
    synergy_files = glob.glob("{}/*/ggr.synergy.h5".format(INFER_DIR))
    
    for synergy_file in synergy_files:

        # set up dir/prefix
        out_dir = "{}/calc_synergy".format(os.path.dirname(synergy_file))
        prefix = os.path.dirname(synergy_file).split("/")[-1]

        # calculate
        calc_synergy = "calculate_mutagenesis_effects.py --synergy_file {} --calculations 11/10 01/00 -o {} --refine --prefix {}".format(
            synergy_file,
            out_dir,
            prefix)
        print calc_synergy
        #os.system(calc_synergy)

    return

main()
