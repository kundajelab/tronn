
import os

def main():
    """intersect with RNA
    run inside nautilus
    """
    # in case inputs are elsewhere
    SCANMOTIFS_DIR = "/datasets/inference.2019-02-05"

    foreground_main_groups = ["early", "mid", "late"]
    foreground_files = ["{}/motifs.input_x_grad.{}/ggr.scanmotifs.h5".format(SCANMOTIFS_DIR, val)
                        for val in foreground_main_groups]
    diff_dir = "motifs.adjust.diff"
    pwms = "/datasets/ggr/1.0.0d/annotations/HOCOMOCOv11_core_pwms_HUMAN_mono.renamed.nonredundant.txt"
    pwm_metadata = "/datasets/ggr/1.0.0d/annotations/HOCOMOCOv11_core_annotation_HUMAN_mono.nonredundant.expressed.txt"
    min_cor = 0.75
    rna = "/datasets/ggr/1.0.0d/annotations/ggr.rna.counts.pc.expressed.timeseries_adj.pooled.rlog.mat.txt.gz"
    use_max_pwm_length = "--max_pwm_length 17 "
    #use_max_pwm_length = ""
    out_dir = "motifs.adjust.diff.rna_filt.dmim"
    
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
    
    return

main()
