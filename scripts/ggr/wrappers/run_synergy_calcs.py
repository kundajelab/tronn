#!/usr/bin/env python

import os
import sys
import h5py
import glob

import numpy as np
import pandas as pd

def main():
    """run synergy calculations <- separate bc need R and such
    """
    # inputs
    out_dir = sys.argv[1]
    grammar_summary_file = sys.argv[2]
    os.system("mkdir -p {}".format(out_dir))
    
    # get synergy files
    grammars_df = pd.read_table(grammar_summary_file)
    synergy_files = []
    for grammar_idx in range(grammars_df.shape[0]):
        grammar = grammars_df.iloc[grammar_idx]["filename"]
        grammar_prefix = os.path.basename(grammar).split(".gml")[0]
        
        # get the synergy file
        synergy_file = glob.glob("synergy*/{}/ggr.synergy.h5".format(grammar_prefix))
        if len(synergy_file) != 1:
            print grammar
        else:
            synergy_file = synergy_file[0]
            synergy_files.append(synergy_file)

        # check that it actually exists and is complete
        try:
            with h5py.File(synergy_file, "r") as hf:
                pwm_names = hf["logits.motif_mut"].attrs["pwm_names"]
        except:
            print "synergy file not readable/does not exist!"
                
    print "total synergy files: {}".format(len(synergy_files))

    # other convenience set ups
    plots_dir = "{}/plots".format(out_dir)
    os.system("mkdir -p {}".format(plots_dir))
    grammars_dir = "{}/grammars.synergy_only".format(out_dir)
    os.system("mkdir -p {}".format(grammars_dir))

    # go through synergy files
    total = 0
    for synergy_file in synergy_files:

        # check whether 2 or 3
        with h5py.File(synergy_file, "r") as hf:
            num_motifs = int(np.log2(hf["sequence.motif_mut.string"].shape[1]))
            
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
                "101/001 100/000", # add first one in all contexts
                "110/010 100/000",
                "111/011 101/001",
                "111/011 110/010",
                
                "011/001 010/000", # add middle one in all contexts
                "110/100 010/000",
                "111/101 011/001",
                "111/101 110/100",

                "011/010 001/000", # add last one in all contexts
                "101/100 001/000",
                "111/110 011/010",
                "111/110 101/100"]
            
        # calculate
        for calculation_idx in range(len(calculations)):
            calculation = calculations[calculation_idx]
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
        
    return

main()
