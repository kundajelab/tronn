"""Contains the run function to evaluate and produce metrics graphs
"""

import os
import json
import h5py
import logging


import numpy as np
import pandas as pd

from tronn.datalayer import setup_h5_files
from tronn.datalayer import H5DataLoader

from tronn.graphs import ModelManager

from tronn.nets.nets import net_fns

from tronn.learn.cross_validation import setup_train_valid_test

from tronn.learn.evaluation import auprc
from tronn.learn.evaluation import make_recall_at_fdr
from tronn.learn.evaluation import run_and_plot_metrics
from tronn.learn.evaluation import plot_all

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def run(args):
    """command to run evaluation
    """
    logging.info("Evaluating trained model...")
    os.system("mkdir -p {}".format(args.out_dir))
    
    # set up dataloader
    dataloader = H5DataLoader(args.model_info["test_files"], fasta=args.fasta)
    test_input_fn = dataloader.build_input_fn(
        args.batch_size, label_keys=args.model_info["label_keys"])

    # set up model
    model_manager = ModelManager(
        net_fns[args.model_info["name"]],
        args.model_info["params"])

    # evaluate
    predictor = model_manager.predict(
        test_input_fn,
        args.out_dir,
        checkpoint=args.model_info["checkpoint"])

    # run eval and save to h5
    eval_h5_file = "{}/{}.eval.h5".format(args.out_dir, args.prefix)
    if not os.path.isfile(eval_h5_file):
        model_manager.infer_and_save_to_h5(predictor, eval_h5_file, 1000)
        
    # extract arrays
    with h5py.File(eval_h5_file, "r") as hf:
        labels = hf["labels"][:]
        predictions = hf["logits"][:]
        probs = hf["probs"][:]
    assert labels.shape[1] == probs.shape[1]
    
    # setup metrics functions and plot file folders
    precision_thresholds = [0.5, 0.75, 0.9, 0.95]
    metrics_functions = [roc_auc_score, auprc] + [
        make_recall_at_fdr(fdr)
        for fdr in precision_thresholds]
    metrics_functions_names = ["AUROC", "AUPRC"] + [
        "recall_at_{}_fdr".format(
            str(int(round(100.*(1. - fdr)))))
        for fdr in precision_thresholds]

    # setup plotting functions
    plot_fn_names = ["auroc", "auprc"]
    plot_param_sets = {
        "auroc": "AUROC FPR TPR",
        "auprc": "AUPRC Recall Precision"}

    # set up arrays to keep results
    metrics_array = np.zeros(
        (labels.shape[1]+1, len(metrics_functions)))
    index_list = []
    index_list.append("global")

    # run global metrics
    run_and_plot_metrics(
        labels.flatten(),
        probs.flatten(),
        metrics_functions,
        metrics_array,
        0,
        plot_fn_names,
        "{}/by_task".format(
            args.out_dir),
        "global")

    # sklearn metrics per task
    for task_idx in range(labels.shape[1]):
        task_key = "task_{}".format(task_idx)
        index_list.append(task_key)

        task_labels = labels[:,task_idx]
        task_probs = probs[:,task_idx]

        # run metrics
        run_and_plot_metrics(
            task_labels,
            task_probs,
            metrics_functions,
            metrics_array,
            task_idx + 1,
            plot_fn_names,
            "{}/by_task".format(
                args.out_dir),
            task_key)

    # convert to df, clean and save out
    metrics_df = pd.DataFrame(
        data=metrics_array,
        index=index_list,
        columns=metrics_functions_names)
    metrics_cleaned_df = metrics_df.fillna(value=0)
    out_table_file = "{0}/by_task/{1}.metrics_summary.txt".format(
        args.out_dir, args.prefix)
    metrics_cleaned_df.to_csv(out_table_file, sep='\t')

    # and plot
    plot_all("{}/by_task".format(args.out_dir),
             args.prefix, plot_param_sets)

    return None
