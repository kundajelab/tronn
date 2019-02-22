#!/usr/bin/env python

import os
import sys
import glob

import numpy as np
import pandas as pd

def main():
    """run synergy calculations <- separate bc need R and such
    """
    # params
    INFER_DIR = sys.argv[1]

    # get synergy files
    synergy_files = glob.glob("{}/*/ggr.synergy.h5".format(INFER_DIR))

    # other convenience set ups
    plots_dir = "{}/plots".format(INFER_DIR)
    os.system("mkdir -p {}".format(plots_dir))
    grammars_dir = "{}/grammars.synergy_filt".format(INFER_DIR)
    os.system("mkdir -p {}".format(grammars_dir))

    # go through synergy files
    for synergy_file in synergy_files:

        # check whether 2 or 3
        with h5py.File(synergy_file, "r") as hf:
            num_motifs = int(np.sqrt(hf["sequence.motif_mut.string"].shape[1]))
        
        # set up dir/prefix
        out_dir = "{}/calc_synergy".format(os.path.dirname(synergy_file))
        prefix = os.path.dirname(synergy_file).split("/")[-1]

        # set up calculations
        if num_motifs == 2:
            calculations = [
                "11/10 01/00",
                "11/01 10/00"]
        elif num_motifs == 3:
            calculations = [
                "101/001 100/000" # add first one in all contexts
                "110/010 100/000"
                "111/011 101/001"
                "111/011 110/010"
                
                "011/001 010/000" # add middle one in all contexts
                "110/100 010/000",
                "111/101 011/001",
                "111/101 110/100"

                "011/010 001/000", # add last one in all contexts
                "101/100 001/000",
                "111/110 011/010",
                "111/110 101/100"]
            
        # calculate
        for calculation_idx in range(len(calculations)):
            calcuation = calculations[calculation_idx]
            calc_synergy = "calculate_mutagenesis_effects.py --synergy_file {} --calculations {} -o {} --refine --prefix {}".format(
                synergy_file,
                calculation,
                out_dir,
                "{}.calc-{}".format(prefix, calculation_idx))
            print calc_synergy
            os.system(calc_synergy)

        # cp plots out to separate folder for easy download
        copy_plots = "cp {}/*pdf {}".format(out_dir, plots_dir)
        os.system(copy_plots)
        
        # and copy grammars to separate folder for easy annotation
        copy_grammar = "cp {}/*gml {}".format(out_dir, grammars_dir)
        os.system(copy_grammar)

    # annotate?
        
    return

main()
