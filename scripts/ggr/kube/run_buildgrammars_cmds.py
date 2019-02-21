
import os
import sys

def main():
    """run inside nautilus to build grammars
    """
    WORK_DIR = sys.argv[1]
    
    indices = range(15)
    indices.remove(6)
    print indices
    out_dirname = "grammars"
    
    for index in indices:
        dmim_dir = "{}/dmim.TRAJ_LABELS-{}".format(WORK_DIR, index)
        
        cmd = "tronn buildgrammars "
        cmd += "--scan_type dmim "
        cmd += "--scan_file {}/ggr.dmim.h5 ".format(dmim_dir)
        cmd += "--sig_pwms_file {}/motifs.rna_filt.dmim/pvals.rna_filt.corr_filt.h5 ".format(WORK_DIR)
        cmd += "--foreground_targets TRAJ_LABELS-{} ".format(index)
        cmd += "--aux_data_key ATAC_SIGNALS.NORM logits.norm "
        cmd += "-o {}/{} ".format(dmim_dir, out_dirname)
        cmd += "--prefix ggr "

        print cmd
        os.system(cmd)

    return


main()
