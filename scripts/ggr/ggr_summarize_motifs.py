
import h5py
import argparse

import pandas as pd
import numpy as np

from tronn.util.scripts import setup_run_logs


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    parser.add_argument(
        "--data_file",
        help="pval file produced after running intersect_pwms_and_rna.py")
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    
    args = parser.parse_args()

    return args


def main():
    """condense results
    """
    # set up args
    args = parse_args()
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    # GGR ordered trajectory indices
    indices = [0,7,8,9,10,11,12,13,14,1,2,3,4,5]
    labels = ["TRAJ-{}".format(val) for val in range(1,14)]

    # go through each index to collect
    for i in range(len(indices)):
        index = indices[i]
        key = "TRAJ_LABELS-{}".format(index)
    
        with h5py.File(args.data_file, "r") as hf:
            sig = hf["pvals"][key]["sig"][:]
            rna_patterns = hf["pvals"][key]["rna_patterns"][:]
            pwm_patterns = hf["pvals"][key]["pwm_patterns"][:]
            correlations = hf["pvals"][key]["correlations"][:]
            hgnc_ids = hf["pvals"][key].attrs["hgnc_ids"]
            pwm_names = hf["pvals"][key].attrs["pwm_names"]
            
        # TF present
        tf_present = pd.DataFrame(
            correlations,
            index=hgnc_ids)
        tf_present.columns = [index]
        
        # rna pattern
        tf_data = pd.DataFrame(rna_patterns, index=hgnc_ids)
        
        # pwm present
        pwm_present = pd.DataFrame(
            np.arcsinh(np.max(pwm_patterns, axis=1)),
            index=pwm_names)
        pwm_present.columns = [index]

        # pwm pattern
        pwm_data = pd.DataFrame(pwm_patterns, index=pwm_names)
        pwm_data = pwm_data.drop_duplicates()
        
        if i == 0:
            traj_tfs = tf_present
            traj_pwms = pwm_present
            tf_patterns = tf_data
            motif_patterns = pwm_data
        else:
            traj_tfs = traj_tfs.merge(tf_present, how="outer", left_index=True, right_index=True)
            traj_pwms = traj_pwms.merge(pwm_present, how="outer", left_index=True, right_index=True)
            tf_patterns = pd.concat([tf_patterns, tf_data])
            tf_patterns = tf_patterns.drop_duplicates()
            motif_patterns = pd.concat([motif_patterns, pwm_data])
            #motif_patterns = motif_patterns.drop_duplicates()

    # remove nans/duplicates
    traj_tfs = traj_tfs.fillna(0)
    traj_pwms = traj_pwms.fillna(0).drop_duplicates()

    # reindex
    tf_patterns = tf_patterns.reindex(traj_tfs.index)
    motif_patterns = motif_patterns.groupby(motif_patterns.index).mean() # right now, just average across trajectories (though not great)
    motif_patterns = motif_patterns.reindex(traj_pwms.index)

    # filtering on specific TFs and motifs to exclude
    

    traj_tfs.to_csv("tfs_corr_summary.txt", sep="\t")
    traj_pwms.to_csv("pwms_present_summary.txt", sep="\t")
    tf_patterns.to_csv("tfs_patterns_summary.txt", sep="\t")
    motif_patterns.to_csv("pwms_patterns_summary.txt", sep="\t")
    
    return


main()
