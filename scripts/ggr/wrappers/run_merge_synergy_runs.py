
import os
import glob
import sys

def main():
    """annotate all grammars
    """
    WORK_DIR = sys.argv[1]
    GGR_DIR = "/datasets/ggr/1.0.0d/annotations"
    
    # annotations
    tss = "{}/ggr.rna.tss.expressed.bed.gz".format(GGR_DIR)
    foreground_rna = "{}/ggr.rna.counts.pc.expressed.timeseries_adj.pooled.rlog.dynamic.mat.txt.gz".format(GGR_DIR)
    background_rna = "{}/ggr.rna.counts.pc.rlog.expressed.txt.gz".format(GGR_DIR)
    pwms = "{}/HOCOMOCOv11_core_pwms_HUMAN_mono.renamed.nonredundant.txt".format(GGR_DIR)
    pwm_metadata = "{}/HOCOMOCOv11_core_annotation_HUMAN_mono.nonredundant.expressed.txt".format(GGR_DIR)

    # debug
    grammar_files = glob.glob("{}/grammars.annotated.manual_filt/*gml".format(WORK_DIR))
    print len(grammar_files)

    # cmd
    cmd = "ggr_merge_synergy_runs.py "
    cmd += "--grammar_summaries {}/grammars.annotated.manual_filt/grammars_summary.txt ".format(WORK_DIR)
    cmd += "--synergy_dirs {}/synergy ".format(WORK_DIR)
    cmd += "--tss {} ".format(tss)
    cmd += "--foreground_rna {} ".format(foreground_rna)
    cmd += "--background_rna {} ".format(background_rna)
    cmd += "--pwms {} ".format(pwms)
    cmd += "--pwm_metadata {} ".format(pwm_metadata)
    cmd += "--merged_synergy_dir {}/synergy ".format(WORK_DIR)
    cmd += "-o {}/grammars.annotated.manual_filt.merged".format(WORK_DIR)
    print cmd
    os.system(cmd)

    return


main()
