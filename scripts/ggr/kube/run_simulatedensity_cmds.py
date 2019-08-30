#!/usr/bin/env python

import os
import sys
import logging

import pandas as pd

def main():
    """run synergy commands
    """
    # args
    MODEL_DIR = sys.argv[1]
    MODEL = sys.argv[2]
    INFER_DIR = sys.argv[3]
    OUT_DIR = sys.argv[4]
    motifs_file = "{}/motifs.sig/motifs.adjust.diff.rna_filt.dmim/summary/ggr.pwms_patterns_summary.txt".format(
        INFER_DIR)
    bash_script = "/datasets/software/git/tronn/scripts/ggr/kube/sims.predict.ggr.bash"

    # prediction sample
    prediction_sample_dir = "/datasets/inference.2019-02-05/motifs.input_x_grad.background"
    
    # first get the desired motif list
    motifs = pd.read_csv(motifs_file, sep="\t", header=0, index_col=0)
    motifs = list(motifs.index)

    # go through motifs
    for motif in motifs:

        # set up out dir (but if it exists don't run)
        out_dir = "{}/{}".format(OUT_DIR, motif)
        if os.path.isdir(out_dir):
            continue
        os.system("mkdir -p {}".format(out_dir))

        # set up input file
        input_file = "{}/input.txt".format(out_dir)
        pd.DataFrame(data=[motif]).to_csv(
            input_file, header=False, index=False)
        
        # set up run cmd
        run_cmd = "{0} {1} {2} {3} {4} {5}".format(
            bash_script,
            MODEL_DIR,
            MODEL,
            prediction_sample_dir,
            input_file,
            out_dir)
        print run_cmd
        os.system(run_cmd)
        
        quit()

    return

main()
