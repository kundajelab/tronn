#!/usr/bin/env python

import os
import sys
import h5py

import numpy as np
import pandas as pd

from tronn.plot.visualization import scale_scores
from tronn.plot.visualization import plot_weights_group
from tronn.util.utils import DataKeys


idx_to_letter = {
    0: "A",
    1: "C",
    2: "G",
    3: "T"
}


def main():
    """look for interesting vignettes with variants in them
    """
    # args
    in_file = sys.argv[1]
    plot_prefix = sys.argv[2]

    # score key
    orig_impts_key = "sequence-weighted"
    match_impts_key = "sequence-weighted.active"

    # other
    LEFT_CLIP = 420
    RIGHT_CLIP = 580
    FINAL_LEFT_CLIP = 20
    FINAL_RIGHT_CLIP = 120
    
    # read in (sorted) file
    vignettes = pd.read_csv(in_file, sep="\t")

    #for i in range(vignettes.shape[0]):
    for i in range(400, 420):
        vignette = vignettes.iloc[i]
        file_name = vignette["file"]
        example_idx = vignette["file_idx"]

        # pull data and clip to match
        with h5py.File(file_name, "r") as hf:
            metadata = hf[DataKeys.SEQ_METADATA][example_idx,0]
            data = hf[DataKeys.WEIGHTED_SEQ][example_idx][:,LEFT_CLIP:RIGHT_CLIP]
            sig = hf[DataKeys.WEIGHTED_SEQ_ACTIVE][example_idx]
        assert metadata == vignette["metadata"]

        region_id = metadata.split(";")[1].split("=")[1].replace(":", "_")
        vignette_prefix = "{}.{}.{}".format(plot_prefix, region_id, example_idx)
        
        # normalize
        adjusted_scores = -scale_scores(data, sig)

        # plot full active region
        out_file = "{}.impts.pdf".format(vignette_prefix)
        plot_weights_group(adjusted_scores, out_file, sig_array=sig)

        # clip again and plot
        out_file = "{}.impts.clipped.pdf".format(vignette_prefix)
        clipped_scores = adjusted_scores[:,FINAL_LEFT_CLIP:FINAL_RIGHT_CLIP]
        clipped_sig = sig[:,FINAL_LEFT_CLIP:FINAL_RIGHT_CLIP]
        plot_weights_group(clipped_scores, out_file, sig_array=clipped_sig)
    
        # pull ATAC actual and predicted, save out and plot
        keep_indices = [0,1,2,3,4,5,6,9,10,12]
        with h5py.File(file_name, "r") as hf:
            actual = hf["ATAC_SIGNALS.NORM"][example_idx][keep_indices]
            predicted = hf["logits.norm"][example_idx][keep_indices]
        actual_v_predicted = pd.DataFrame({
            "timepoint": ["d0.0", "d0.5", "d1.0", "d1.5", "d2.0", "d2.5", "d3.0", "d4.5", "d5.0", "d6.0"],
            "ATAC": actual,
            "predicted": predicted}).set_index("timepoint")
        plot_data_file = "importances.atac.actual_v_predicted.txt"
        actual_v_predicted.to_csv(plot_data_file, header=True, index=True, sep="\t")

        script_dir = "/users/dskim89/git/ggr-project/figs/fig_2.modelling"
        plot_file = "{}.atac.actual_v_predicted.pdf".format(vignette_prefix)
        plot_cmd = "{}/plot.atac.actual_v_pred.R {} {}".format(
            script_dir, plot_data_file, plot_file)
        print plot_cmd
        os.system(plot_cmd)
    
    return


main()
