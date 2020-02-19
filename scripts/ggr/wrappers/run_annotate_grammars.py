#!/usr/bin/env python

import os
import glob
import sys

def main():
    """annotate all grammars
    """
    #WORK_DIR = sys.argv[1]
    WORK_DIR = "/mnt/lab_data3/dskim89/ggr/nn/2020-01-13/grammars.muts"
    GGR_DIR = "/mnt/lab_data/kundaje/users/dskim89/ggr/integrative/v1.0.0a"
    
    # annotations
    region_signal_mat_file = "{}/data/ggr.atac.ends.counts.pooled.rlog.dynamic.traj.mat.txt.gz".format(GGR_DIR)
    rna_signal_mat_file = "{}/data/ggr.rna.counts.pc.expressed.timeseries_adj.pooled.rlog.dynamic.traj.mat.txt.gz".format(GGR_DIR)
    links_file = "{}/results/linking/proximity/ggr.linking.ALL.overlap.interactions.txt.gz".format(GGR_DIR)
    tss_file = "{}/annotations/ggr.rna.tss.expressed.bed.gz".format(GGR_DIR)
    foreground_rna = rna_signal_mat_file
    background_rna = "{}/data/ggr.rna.counts.pc.expressed.mat.txt.gz".format(GGR_DIR)
    #pwms = "{}/HOCOMOCOv11_core_pwms_HUMAN_mono.renamed.nonredundant.txt".format(GGR_DIR) # any need for this? would only be for plotting
    pwm_metadata = "{}/annotations/HOCOMOCOv11_core_annotation_HUMAN_mono.nonredundant.expressed.txt".format(GGR_DIR)

    # get all grammars
    grammar_select = "{}/grammars.TRAJ*/*gml".format(WORK_DIR)
    grammar_files = sorted(glob.glob(grammar_select))
    print len(grammar_files)

    # out dir
    out_dir = "{}/grammars.annotated".format(WORK_DIR)

    # run cmd
    cmd = "ggr_annotate_grammars.py "
    cmd += "--grammars {} ".format(grammar_select)
    cmd += "--region_signal_mat_file {} ".format(region_signal_mat_file)
    cmd += "--rna_signal_mat_file {} ".format(rna_signal_mat_file)
    cmd += "--links_file {} ".format(links_file)
    cmd += "--tss_file {} ".format(tss_file)
    cmd += "--foreground_rna {} ".format(foreground_rna)
    cmd += "--background_rna {} ".format(background_rna)
    #cmd += "--no_go_terms "
    cmd += "--pwm_metadata {} ".format(pwm_metadata)
    cmd += "-o {} ".format(out_dir)
    cmd += "--tmp_dir {}/tmp".format(out_dir)
    print cmd
    os.system(cmd)

    return


main()
