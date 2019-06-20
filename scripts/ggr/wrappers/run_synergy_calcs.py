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
        synergy_file = glob.glob("sims.synergy/{}/ggr.synergy.h5".format(grammar_prefix))
        #synergy_file = glob.glob("synergy*/{}/ggr.synergy.h5".format(grammar_prefix))
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
    group_summary_file = "{}/interactions.summary.txt".format(out_dir)
    print group_summary_file
    header_str = "pwm1\tpwm2\tnum_examples\tsig\tbest_task_index\tactual\texpected\tdiff\tpval\tcategory"
    write_header = "echo '{}' > {}".format(header_str, group_summary_file)
    print write_header
    os.system(write_header)
    for synergy_file in synergy_files:

        # check whether 2 or 3
        with h5py.File(synergy_file, "r") as hf:
            num_motifs = int(np.log2(hf["sequence.motif_mut.string"].shape[1]))
            
        # set up dir/prefix
        out_dir = "{}/calc_synergy".format(os.path.dirname(synergy_file))
        prefix = os.path.dirname(synergy_file).split("/")[-1]

        # set up calculations
        if num_motifs == 2:
            calculations = "11"
        elif num_motifs == 3:
            calculations = "110 101 011"
            
        # calculate
        #for calculation_idx in range(len(calculations)):
        #    calculation = calculations[calculation_idx]
        calc_synergy = "calculate_mutagenesis_effects.py --synergy_file {} --calculations {} -o {} --refine --prefix {}".format(
            synergy_file,
            calculations,
            out_dir,
            prefix)
        print calc_synergy
        os.system(calc_synergy)
        print ""

        # extract the calculation results into a summary file
        if num_motifs == 2:
            summary_file = "{}/{}.interactions.txt".format(out_dir, prefix)
            get_summary = "cat {} | awk 'NR>1{{ print }}' >> {}".format(
                summary_file, group_summary_file)
            print get_summary
            os.system(get_summary)

        # cp plots out to separate folder for easy download
        copy_plots = "cp {}/*pdf {}".format(out_dir, plots_dir)
        #os.system(copy_plots)
        
        # and copy grammars to separate folder for easy annotation
        copy_grammar = "cp {}/*gml {}".format(out_dir, grammars_dir)
        #os.system(copy_grammar)

    # plot the expected vs observed results
    
    
    return

main()
