#!/usr/bin/env python

import os
import glob
import sys

def main():
    """build grammars
    """
    # dirs
    WORK_DIR = "/mnt/lab_data3/dskim89/ggr/nn/2020-01-13/mutatemotifs/shuffle"
    SIG_MOTIF_DIR = "/mnt/lab_data3/dskim89/ggr/nn/2019-03-12.freeze/motifs.sig/motifs.adjust.diff.rna_filt.dmim"
    OUT_DIR = "."

    # params
    min_support = 1000 # legacy - 600
    scan_type = "muts" # hits, impts, muts
    
    # set up
    dmim_files = sorted(glob.glob("{}/*/ggr.mutatemotifs.h5".format(WORK_DIR)))
    dmim_dirs = [os.path.dirname(dmim_file) for dmim_file in dmim_files]
    out_dirname = "{}/grammars.{}".format(OUT_DIR, scan_type)
    
    # for each folder, build grammars
    for i in range(len(dmim_files)):
        dmim_file = dmim_files[i]
        dmim_dir = dmim_dirs[i]

        # cmd
        cmd = "tronn buildgrammars "
        cmd += "--scan_file {}/ggr.mutatemotifs.h5 ".format(dmim_dir)
        cmd += "--scan_type {} ".format(scan_type)
        cmd += "--sig_pwms_file {}/pvals.rna_filt.corr_filt.h5 ".format(SIG_MOTIF_DIR)
        cmd += "--rc_pwms_present "
        cmd += "--foreground_targets {} ".format(dmim_dir.split("/")[-1])
        cmd += "--min_support_fract 0.1 "
        cmd += "--min_support {} ".format(min_support)
        #cmd += "--keep_grammars HCLUST-169_TFAP2A.UNK.0.A,HCLUST-184_KLF12.UNK.0.A "
        cmd += "--ignore_pwms MECOM MBD2 SNAI PBX ARNTL ZNF350 ZNF528 ZNF140 MEIS CUX NFIA NANOG SMARC ZFP82 ZNF667 ZNF547 ZNF317 ZNF322 "
        cmd += "--aux_data_key ATAC_SIGNALS.NORM logits.norm "
        cmd += "-o {}/grammars.{} ".format(out_dirname, dmim_dir.split("/")[-1])
        cmd += "--prefix ggr "

        print cmd
        os.system(cmd)
        
    return


main()
