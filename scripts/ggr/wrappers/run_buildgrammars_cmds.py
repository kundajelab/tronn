#!/usr/bin/env python

import os
import glob
import sys

def main():
    """run inside nautilus to build grammars
    """
    WORK_DIR = sys.argv[1]
    MOTIF_DIR = sys.argv[2]

    # set up
    dmim_files = sorted(glob.glob("{}/*/ggr.mutatemotifs.h5".format(WORK_DIR)))
    dmim_dirs = [os.path.dirname(dmim_file) for dmim_file in dmim_files]
    out_dirname = "grammars"

    # for each folder, build grammars
    for i in range(len(dmim_files)):
        dmim_file = dmim_files[i]
        dmim_dir = dmim_dirs[i]
        
        cmd = "tronn buildgrammars "
        cmd += "--scan_file {}/ggr.mutatemotifs.h5 ".format(dmim_dir)
        cmd += "--sig_pwms_file {}/motifs.adjust.diff.rna_filt/pvals.rna_filt.corr_filt.h5 ".format(MOTIF_DIR)
        cmd += "--foreground_targets {} ".format(dmim_dir.split("/")[-1])
        cmd += "--min_support_fract 0.1 "
        cmd += "--min_support 600 "
        #cmd += "--keep_grammars HCLUST-169_TFAP2A.UNK.0.A,HCLUST-184_KLF12.UNK.0.A "
        cmd += "--ignore_pwms MECOM MBD2 SNAI PBX ARNTL ZNF350 ZNF528 ZNF140 MEIS CUX NFIA NANOG SMARC ZFP82 ZNF667 ZNF547 ZNF317 ZNF322 "
        cmd += "--aux_data_key ATAC_SIGNALS.NORM logits.norm "
        cmd += "-o grammars.{} ".format(dmim_dir.split("/")[-1])
        cmd += "--prefix ggr "

        print cmd
        os.system(cmd)
        
    return


main()
