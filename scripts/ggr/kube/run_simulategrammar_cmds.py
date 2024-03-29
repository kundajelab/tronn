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
    grammars_file = "{}/grammars.annotated.manual_filt.merged.final/grammars_summary.txt".format(INFER_DIR)
    bash_script = "/datasets/software/git/tronn/scripts/ggr/kube/simulategrammars.ggr.bash"
    
    # now read in table
    grammars_df = pd.read_table(grammars_file)
    print grammars_df

    # now set up run for each line
    for grammar_idx in range(grammars_df.shape[0]):

        # get grammar file
        grammar_file = grammars_df["filename"].iloc[grammar_idx]
        grammar_file = "{}/{}".format(INFER_DIR, grammar_file)
        print grammar_file

        # get traj label for dir with prediction sample
        traj_label = os.path.basename(grammar_file).split(".")[1].split("_x")[0]
        traj_dir = "{}/{}".format(INFER_DIR, traj_label)
        
        # adjust out dir
        out_dir = "{}/{}".format(
            OUT_DIR,
            os.path.basename(grammar_file).split(".gml")[0])
        os.system("mkdir -p {}".format(out_dir))

        # if already exists don't rerun
        if os.path.isfile("{}/ggr.simulategrammar.h5".format(out_dir)):
            continue
        
        # go to synergy run script
        run_cmd = "{0} {1} {2} {3} {4} {5}".format(
            bash_script,
            MODEL_DIR,
            MODEL,
            grammar_file,
            traj_dir,
            out_dir)
        print run_cmd
        os.system(run_cmd)

    return

main()
