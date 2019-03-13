
import os
import glob
import sys

def main():
    """run inside nautilus to build grammars
    """
    WORK_DIR = sys.argv[1]
    MOTIF_DIR = sys.argv[2]

    # set up
    dmim_files = sorted(glob.glob("{}/*/ggr.dmim.h5".format(WORK_DIR)))
    dmim_dirs = [os.path.dirname(dmim_file) for dmim_file in dmim_files]
    out_dirname = "grammars"

    # for each folder, build grammars
    for i in range(len(dmim_files)):
        dmim_file = dmim_files[i]
        dmim_dir = dmim_dirs[i]
        
        cmd = "tronn buildgrammars "
        cmd += "--scan_type dmim "
        cmd += "--scan_file {}/ggr.dmim.h5 ".format(dmim_dir)
        cmd += "--sig_pwms_file {}/motifs.adjust.diff.rna_filt.dmim/pvals.rna_filt.corr_filt.h5 ".format(MOTIF_DIR)
        cmd += "--foreground_targets {} ".format(dmim_dir.split("/")[-1])
        cmd += "--keep_grammars HCLUST-169_TFAP2A.UNK.0.A,HCLUST-184_KLF12.UNK.0.A "
        cmd += "--aux_data_key ATAC_SIGNALS.NORM logits.norm "
        cmd += "-o {}/{} ".format(dmim_dir, out_dirname)
        cmd += "--prefix ggr "

        print cmd
        os.system(cmd)
        
    return


main()
