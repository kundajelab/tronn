
import os
import glob
import sys

def main():
    """annotate all grammars
    """
    WORK_DIR = sys.argv[1]
    ANNOT_DIR = "/datasets/annotations.hg19"
    GGR_DIR = "/datasets/ggr/1.0.0d/annotations"
    
    # annotations
    tss = "{}/hg19.tss.gencode19.bed.gz".format(GGR_DIR)
    foreground_rna = "{}/ggr.rna.counts.pc.expressed.timeseries_adj.pooled.rlog.dynamic.mat.txt.gz".format(GGR_DIR)
    background_rna = "{}/ggr.rna.counts.pc.rlog.expressed.txt.gz".format(GGR_DIR)
    pwms = "{}/HOCOMOCOv11_core_pwms_HUMAN_mono.renamed.nonredundant.txt".format(GGR_DIR)
    pwm_metadata = "{}/HOCOMOCOv11_core_annotation_HUMAN_mono.nonredundant.expressed.txt".format(GGR_DIR)
    
    grammar_files = glob.glob("{}/dmim*/grammars.dmim/*gml".format(WORK_DIR))
    print len(grammar_files)

    cmd = "ggr_annotate_grammars.py "
    cmd += "--grammars {}/*/grammars/*gml ".format(WORK_DIR)
    cmd += "--tss {} ".format(tss)
    cmd += "--foreground_rna {} ".format(foreground_rna)
    cmd += "--background_rna {} ".format(background_rna)
    cmd += "--pwms {} ".format(pwms)
    cmd += "--pwm_metadata {} ".format(pwm_metadata)
    cmd += "-o {}/grammars.annotated".format(WORK_DIR)
    print cmd
    os.system(cmd)

    return


main()
