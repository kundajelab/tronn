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
    # WARNING: remember for synergy with real inputs you need to do BEFORE MERGE (not true for sims)
    #grammars_file = "{}/grammars.annotated.manual_filt/grammars_summary.txt".format(INFER_DIR)
    grammars_file = "{}/grammars.annotated.manual_filt.merged.final/grammars_summary.txt".format(INFER_DIR)
    bash_script = "/datasets/software/git/tronn/scripts/ggr/kube/synergy.ggr.bash"
    
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
                data_files += " {}/{}/ggr.dmim.h5".format(INFER_DIR, traj_label)
            data_files +="\""
        else:
            data_files = "\"\""

        # adjust out dir
        out_dir = "{}/{}".format(
            OUT_DIR,
            os.path.basename(grammar_file).split(".gml")[0])
        os.system("mkdir -p {}".format(out_dir))

        # if already exists don't rerun
        if os.path.isfile("{}/ggr.synergy.h5".format(out_dir)):
            continue
        
        # adjust batch size in synergy run script based on num mut motifs
        mut_nodes = grammars_df["nodes"].iloc[grammar_idx].split(",")
        if len(mut_nodes) == 2:
            batch_size = 12
        elif len(mut_nodes) == 3:
            batch_size = 6
        else:
            print "mistake!"
        adjust_batch_size = "sed -i -e 's/batch_size [0-9]\+/batch_size {}/g' {}".format(batch_size, bash_script)
        print adjust_batch_size
        os.system(adjust_batch_size)

        # adjust mutate type
        if "point" in INFER_DIR:
            mutate_type = "point"
        else:
            mutate_type = "shuffle"
        
        # go to synergy run script
        run_cmd = "{0} {1} {2} {3} {4} {5} {6} {7}".format(
            bash_script,
            MODEL_DIR,
            MODEL,
            "{}/{}".format(INFER_DIR, traj_labels[0]),
            data_files,
            grammar_file,
            mutate_type,
            out_dir)
        print run_cmd
        os.system(run_cmd)

    return

main()
