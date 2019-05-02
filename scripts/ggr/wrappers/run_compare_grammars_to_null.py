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

    # synergy_only
    if True:
        synergy_only = "--synergy_only"
    else:
        synergy_only = ""
        
    # variant files
    variant_dir = "/datasets/ggr/1.0.0d/annotations/variants.validation"
    eqtl_bed_file = "{}/gtex.skin.atac_filt.bed.gz".format(variant_dir)
    eqtl_effects_file = "{}/Skin_Not_Sun_Exposed_Suprapubic.v7.signif_variant_gene_pairs.txt.gz".format(
        variant_dir)
    
    # read in grammar summary
    grammars_df = pd.read_table(grammar_summary_file)

    # for each grammar, analyze
    for grammar_idx in range(grammars_df.shape[0]):

        # get grammar file
        grammar_file = grammars_df.iloc[grammar_idx]["filename"]
        grammar_prefix = os.path.basename(grammar_file).split(".gml")[0]
        
        # get score files
        trajs = grammar_prefix.split(".")[1].split("_x_")
        score_files = ["{}/ggr.dmim.h5".format(traj) for traj in trajs]

        # get synergy file
        synergy_file = "synergy/{}/ggr.synergy.h5".format(grammar_prefix)

        # commands
        grammar_out_dir = "{}/{}".format(out_dir, grammar_prefix)
        if not os.path.isdir(grammar_out_dir):
            command = (
                "compare_grammar_to_null.py "
                "--grammar {} "
                "--score_files {} "
                "--synergy_file {} "
                "--eqtl_bed_file {} "
                "--eqtl_effects_file {} "
                "-o {} "
                "--prefix {}").format(
                    grammar_file,
                    " ".join(score_files),
                    synergy_file,
                    eqtl_bed_file,
                    eqtl_effects_file,
                    grammar_out_dir,
                    grammar_prefix)
            print command
            os.system(command)
        
        # also run synergy only
        grammar_out_dir_synergy = "{}/{}.synergy_only".format(out_dir, grammar_prefix)
        if not os.path.isdir(grammar_out_dir_synergy):
            command = (
                "compare_grammar_to_null.py "
                "--grammar {} "
                "--score_files {} "
                "--synergy_file {} "
                "--eqtl_bed_file {} "
                "--eqtl_effects_file {} "
                "-o {} "
                "--prefix {} "
                "--synergy_only").format(
                    grammar_file,
                    " ".join(score_files),
                    synergy_file,
                    eqtl_bed_file,
                    eqtl_effects_file,
                    grammar_out_dir_synergy,
                    grammar_prefix)
            print command
            os.system(command)
        
    # finally, aggregate
    # for ATAC/H3K27ac/variants, gather all files, sort, grab the commented lines
    # then go through and collect points, with null removed
        
    return

main()
