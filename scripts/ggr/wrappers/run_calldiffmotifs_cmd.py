
import os

def main():
    """wrapper for call_differential_motifs.py script
    Run inside inference directory
    """
    # in case inputs are elsewhere
    SCANMOTIFS_DIR = "/datasets/inference.2019-02-05"
    
    # qval
    qval = 0.05
    
    # scanmotifs files
    foreground_main_groups = ["early", "mid", "late"]
    foreground_files = ["{}/motifs.input_x_grad.{}/ggr.scanmotifs.h5".format(SCANMOTIFS_DIR, val) for val in foreground_main_groups]
    background_files = ["{}/motifs.input_x_grad.background/ggr.scanmotifs.h5".format(SCANMOTIFS_DIR)]

    # foregrounds and background
    refine = True
    if refine:
        foregrounds = [
            "TRAJ_LABELS=0",
            "TRAJ_LABELS=7",
            "TRAJ_LABELS=8,10,11",
            "TRAJ_LABELS=9",
            "TRAJ_LABELS=12,13",
            "TRAJ_LABELS=14",
            "TRAJ_LABELS=1",
            "TRAJ_LABELS=2",
            "TRAJ_LABELS=3,4",
            "TRAJ_LABELS=5"]
    else:
        foregrounds = [
            "TRAJ_LABELS=0",
            "TRAJ_LABELS=7",
            "TRAJ_LABELS=8",
            "TRAJ_LABELS=10",
            "TRAJ_LABELS=11",
            "TRAJ_LABELS=9",
            "TRAJ_LABELS=12",
            "TRAJ_LABELS=13",
            "TRAJ_LABELS=14",
            "TRAJ_LABELS=1",
            "TRAJ_LABELS=2",
            "TRAJ_LABELS=3",
            "TRAJ_LABELS=4",
            "TRAJ_LABELS=5"]
    background = "ATAC_LABELS=0,1,2,3,4,5,6,9,10,12"

    # infer json
    infer_json = "{}/motifs.input_x_grad.background/infer.scanmotifs.json".format(SCANMOTIFS_DIR)
    
    # out
    out_dir = "motifs.adjust.diff"
    
    # cmd
    cmd = "call_differential_motifs.py "
    cmd += "--foreground_files {} ".format(" ".join(foreground_files))
    cmd += "--background_files {} ".format(" ".join(background_files))
    cmd += "--foregrounds {} ".format(" ".join(foregrounds))
    cmd += "--background {} ".format(background)
    cmd += "--inference_json {} ".format(infer_json)
    cmd += "--qval_thresh {} ".format(qval)
    cmd += "-o {}".format(out_dir)

    print cmd
    os.system(cmd)

    return

main()