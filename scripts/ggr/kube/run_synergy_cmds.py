#!/usr/bin/env python

import os
import sys

import pandas as pd

def main():
    """run synergy commands
    """
    # first get the desired grammars table
    MODEL_DIR = sys.argv[1]
    MODEL = sys.argv[2]
    INFER_DIR = sys.argv[3]
    OUT_DIR = sys.argv[4]

    # TODO adjust work dir
    grammars_file = "{}/grammars.annotated/grammar_summary.filt.dedup.txt".format(INFER_DIR)
    
    # now read in table
    grammars_df = pd.read_table(grammars_file)
    print grammars_df

    # now set up run for each line
    for grammar_idx in range(grammars_df.shape[0]):

        # get grammar file
        grammar_file = grammars_df["filename"].iloc[grammar_idx]
        grammar_file = "{}/{}".format(INFER_DIR, grammar_file)
        print grammar_file
        
        # determine traj labels
        traj_labels = os.path.basename(grammar_file).split(".")[1].split("_x_")

        # adjust data files if needed
        if len(traj_labels) > 1:
            data_files = "\"--data_files"
            for traj_label in traj_labels:
                data_files += " {}/dmim.{}/ggr.dmim.h5".format(INFER_DIR, traj_label)
            data_files +="\""
        else:
            data_files = "\"\""

        # adjust out dir
        out_dir = "{}/{}".format(
            OUT_DIR,
            os.path.basename(grammar_file).split(".gml")[0])
        os.system("mkdir -p {}".format(out_dir))
            
        # go to synergy run script
        run_cmd = "/datasets/software/git/tronn/scripts/ggr/kube/synergy.ggr.bash {} {} {} {} {} {}".format(
            MODEL_DIR,
            MODEL,
            "{}/dmim.{}".format(INFER_DIR, traj_labels[0]),
            data_files,
            grammar_file,
            out_dir)
        print run_cmd
        os.system(run_cmd)
    
    return

main()
