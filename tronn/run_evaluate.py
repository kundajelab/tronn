"""Contains the run function to evaluate and produce metrics graphs
"""

import os
import glob
import logging
import h5py

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from tronn.graphs import TronnGraphV2
from tronn.datalayer import H5DataLoader
from tronn.nets.nets import net_fns
from tronn.learn.cross_validation import setup_cv

from tronn.learn.evaluation import auprc
from tronn.learn.evaluation import make_recall_at_fdr
from tronn.learn.evaluation import run_and_plot_metrics
from tronn.learn.evaluation import plot_all
from tronn.learn.evaluation import full_evaluate


def run(args):
    """Pipeline to evaluate a trained model
    
    Sets up model graphs and then runs the graphs for some number of batches
    to get predictions. Then builds out curves to plot and gets definitive
    metric values.
    """
    logging.info("Running evaluation...")
    os.system("mkdir -p {}".format(args.out_dir))
    
    # find data files
    data_files = sorted(glob.glob("{}/*.h5".format(args.data_dir)))
    train_files, valid_files, test_files = setup_cv(data_files, cvfold=args.cvfold)

    # set up dataloader
    dataloader = H5DataLoader(
        {"valid": valid_files, "test": test_files},
        tasks=args.tasks,
        shuffle_examples=True if args.reconstruct_regions else False)
        
    # set up neural net graph
    tronn_graph = TronnGraphV2(
        dataloader,
        net_fns[args.model["name"]],
        args.model,
        args.batch_size,
        final_activation_fn=tf.nn.sigmoid,
        loss_fn=tf.losses.sigmoid_cross_entropy,
        checkpoints=args.model_checkpoints)

    # run eval graph
    eval_h5_file = "{}/{}.eval.h5".format(args.out_dir, args.prefix)
    if not os.path.isfile(eval_h5_file):
        full_evaluate(tronn_graph, eval_h5_file)

    # extract arrays
    with h5py.File(eval_h5_file, "r") as hf:
        labels = hf["labels"][:]
        predictions = hf["logits"][:]
        probs = hf["probs"][:]
        metadata = hf["example_metadata"][:]

    # push predictions through activation to get probs
    #if args.model_type != "nn":
    #    probs = scores_to_probs(predictions)

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
        
    # if NOT single task, run all tasks on all predictions
    # labels and predictions must match in shape
    if args.single_task is None:
        assert labels.shape[1] == probs.shape[1]

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
        # TODO convert this to task or to prediction set (which requires specific task to look at)
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
        
    else:
        # if single task, compare single task to all predictions
        metrics_array = np.zeros(
            (predictions.shape[1], len(metrics_functions)))
        index_list = []
        
        task_labels = labels[:, args.single_task]
        
        for prediction_idx in range(predictions.shape[1]):
            prediction_key = "prediction_{}".format(prediction_idx)
            index_list.append(prediction_key)

            prediction_probs = probs[:, prediction_idx]

            # run metrics
            run_and_plot_metrics(
                task_labels,
                prediction_probs,
                metrics_functions,
                metrics_array,
                prediction_idx,
                plot_fn_names,
                "{}/by_prediction".format(
                    args.out_dir),
                prediction_key)

        # convert to df, clean and save out
        metrics_df = pd.DataFrame(
            data=metrics_array,
            index=index_list,
            columns=metrics_functions_names)
        metrics_cleaned_df = metrics_df.fillna(value=0)
        out_table_file = "{0}/by_prediction/{1}.metrics_summary.txt".format(
            args.out_dir, args.prefix)
        metrics_cleaned_df.to_csv(out_table_file, sep='\t')

        # and plot
        plot_all("{}/by_prediction".format(args.out_dir),
                 args.prefix, plot_param_sets)

    return None
