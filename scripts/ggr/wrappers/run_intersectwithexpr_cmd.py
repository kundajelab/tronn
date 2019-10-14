#!/usr/bin/env python

import os

def main():
    """intersect with RNA
    run inside nautilus
    """
    # in case inputs are elsewhere
    #SCANMOTIFS_DIR = "/datasets/inference.2019-02-05"
    SCANMOTIFS_DIR = "/mnt/lab_data/kundaje/users/dskim89/ggr/nn/inference.2019-10-08"

    # set up
    foreground_main_groups = ["early", "mid", "late"]
    foreground_files = ["{}/motifs.input_x_grad.dynamic.{}/ggr.scanmotifs.h5".format(SCANMOTIFS_DIR, val)
                        for val in foreground_main_groups]
    diff_dir = "{}/motifs.sig/motifs.adjust.diff".format(SCANMOTIFS_DIR)

    # GGR files
    GGR_DIR = "/mnt/lab_data/kundaje/users/dskim89/ggr/integrative/v1.0.0a"
    pwms = "{}/annotations/HOCOMOCOv11_core_pwms_HUMAN_mono.renamed.nonredundant.txt".format(GGR_DIR)
    pwm_metadata = "{}/annotations/HOCOMOCOv11_core_annotation_HUMAN_mono.nonredundant.expressed.txt".format(GGR_DIR)
    rna = "{}/results/rna/timeseries/matrices/ggr.rna.counts.pc.expressed.timeseries_adj.pooled.rlog.mat.txt.gz".format(GGR_DIR)
    #rna = "{}/data/ggr.rna.counts.pc.expressed.timeseries_adj.pooled.rlog.dynamic.traj.mat.txt.gz".format(GGR_DIR)
    
    # params
    min_cor = 0.75
    use_max_pwm_length = "--max_pwm_length 17 "
    out_dir = "{}/motifs.sig/motifs.adjust.diff.rna_filt".format(SCANMOTIFS_DIR)
    
    cmd = "intersect_pwms_and_rna.py "
    cmd += "--dataset_files {} ".format(" ".join(foreground_files))
    cmd += "--pvals_file {}/pvals.h5 ".format(diff_dir)
    cmd += "--pwm_file {} ".format(pwms)
    cmd += "--pwm_metadata_file {} ".format(pwm_metadata)
    cmd += "--cor_thresh {} ".format(min_cor)
    cmd += "--other_targets ATAC_SIGNALS.NORM=0,2,3,4,5,6,9,10,12 logits=0,2,3,4,5,6,9,10,12 "
    cmd += "--rna_expression_file {} ".format(rna)
    cmd += use_max_pwm_length
    cmd += "-o {} ".format(out_dir)
    print cmd
    os.system(cmd)

    # also do the summary here too
    cmd = "ggr_summarize_motifs.py "
    cmd += "--data_file {}/pvals.rna_filt.corr_filt.h5 ".format(out_dir)
    cmd += "-o {}/summary ".format(out_dir)
    cmd += "--prefix ggr"
    print cmd
    os.system(cmd)
    
    
    return

main()
