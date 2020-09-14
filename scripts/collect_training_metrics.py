

import os, sys
import json
import h5py

import numpy as np
import pandas as pd


def aggregate_positives_negatives_per_task(
        data_dir, 
        h5_files,
        label_keys=["ATAC_LABELS"]):
    """aggregate
    """
    all_results = None
    for label_key in label_keys:
        total_examples = 0
        for h5_i in range(len(h5_files)):
            print h5_files[h5_i]
            h5_file = "{}/{}".format(data_dir, h5_files[h5_i])

            # get data
            h5_results = None
            with h5py.File(h5_file, "r") as hf:

                # labels summed is positives
                data = np.sum(hf[label_key][:], axis=0)

                # total examples gives you negatives
                total_examples += hf[label_key].shape[0]

                # also get file names
                data_files = hf[label_key].attrs["filenames"]
                
            # append on
            if h5_i == 0:
                # start fresh
                num_positives = data
            else:
                # add on
                num_positives += data
        
        # set up negatives
        num_negatives = total_examples - num_positives

        # make dataframe
        results = pd.DataFrame({
            "task": data_files,
            "positives": num_positives,
            "negatives": num_negatives})
        results["label_key"] = label_key
        
        if all_results is None:
            all_results = results
        else:
            all_results = pd.concat([all_results, results])
        
    return all_results



def main():
    """get the training metrics for a dataset
    """
    data_dir = sys.argv[1]
    model_dir = sys.argv[2]
    model_prefix = sys.argv[3]
    models = range(10)

    splits = ["train", "validation", "test"]


    # for encode/roadmap
    label_keys = [
        "DNASE_LABELS",
        "TF_CHIP_LABELS"]
        
    # for GGR
    label_keys_SKIP = [
        "ATAC_LABELS",
        "TRAJ_LABELS",
        "H3K27ac_LABELS",
        "H3K4me1_LABELS",
        "H3K27me3_LABELS",
        "CTCF_LABELS",
        "POL2_LABELS",
        "TP63_LABELS",
        "KLF4_LABELS",
        "ZNF750_LABELS",
        "DYNAMIC_MARK_LABELS",
        "DYNAMIC_STATE_LABELS",
        "STABLE_MARK_LABELS",
        "STABLE_STATE_LABELS"]

    all_results = None
    
    # collect across models
    for model_i in models:
        print model_i
        
        for split in splits:
            print split
        
            # dataset json
            dataset_json = "{}/{}{}/dataset.{}_split.json".format(
                model_dir, model_prefix, model_i, split)

            # load
            with open(dataset_json, "r") as fp:
                data_dict = json.load(fp)
            data_files = data_dict["data_files"]
            results = aggregate_positives_negatives_per_task(
                data_dir, data_files, label_keys=label_keys)
            results["split"] = split
            results["model"] = "{}{}".format(model_prefix, model_i)
            
            # merge
            if all_results is None:
                all_results = results
            else:
                all_results = pd.concat([all_results, results])

    ordered_columns = ["label_key", "task", "split", "model", "positives", "negatives"]
    all_results = all_results[ordered_columns]

    # sort
    all_results = all_results.sort_values(["label_key", "task", "split", "model"])

    # calculate class imbalance
    all_results["class_imbalance"] = all_results["positives"].astype(float) / (all_results["positives"] + all_results["negatives"])
    
    # save out
    all_results.to_csv("encode_roadmap.class_imbalances.txt", sep="\t", index=False)
    
    return

main()
