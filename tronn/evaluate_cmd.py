"""Contains the run function to evaluate and produce metrics graphs
"""

import os
import json
import h5py
import logging

import numpy as np
import pandas as pd

from tronn.datalayer import setup_data_laoder

from tronn.models import setup_model_manager

from tronn.nets.nets import net_fns
from tronn.learn.evaluation import auprc
from tronn.learn.evaluation import make_recall_at_fdr
from tronn.learn.evaluation import spearman_only_r
from tronn.learn.evaluation import pearson_only_r
from tronn.learn.evaluation import run_and_plot_metrics
from tronn.learn.evaluation import plot_all

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def run(args):
    """command to run evaluation
    """
    logging.info("Evaluating trained model...")
    os.system("mkdir -p {}".format(args.out_dir))

    # set up data loader
    test_data_loader = setup_data_loader(args)
    test_input_fn = test_dataloader.build_input_fn(
        args.batch_size,
        targets=args.targets,
        target_indies=args.target_indices,
        filter_targets=args.filter_targets)

    # set up model
    model_manager = setup_model_manager(args)
    
    # evaluate
    predictor = model_manager.predict(
        test_input_fn,
        args.out_dir,
        checkpoint=model_manager.model_checkpoint)

    # run eval and save to h5
    eval_h5_file = "{}/{}.eval.h5".format(args.out_dir, args.prefix)
    if not os.path.isfile(eval_h5_file):
        model_manager.infer_and_save_to_h5(
            predictor, eval_h5_file, args.num_evals)

    # extract from h5 to numpy arrays
    with h5py.File(eval_h5_file, "r") as hf:
        labels = hf["labels"][:]
        predictions = hf["logits"][:]
        probs = hf["probs"][:]
    assert labels.shape[1] == probs.shape[1]

    # set up metrics (different for classification/regression)
    if args.regression:
        metrics_functions = [
            mean_squared_error,
            r2_score,
            spearman_only_r,
            pearson_only_r]
        metrics_functions_names = [
            "MSE",
            "R2",
            "SPEARMANR",
            "PEARSONR"]
        probs = predictions # change probs to the predictions
        
        # setup plotting functions
        plot_fn_names = ["correlation"]
        plot_param_sets = {
            "correlation": "Correlation Predicted Actual"}
        
    else:
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
