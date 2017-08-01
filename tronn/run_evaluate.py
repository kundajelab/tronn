"""Contains the run function to evaluate and produce metrics graphs
"""

import os
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.architectures import models
from tronn.learn.learning import predict

from tronn.run_predict import setup_model_params


def run_sklearn_metric_fn(metrics_fn, labels, probs):
    """Wrapper around sklearn metrics functions to allow code to proceed
    even if missing a class in evaluation

    Args:
      metrics_fn: a function that takes labels and probs
      labels: 1D label vector
      probs: 1D probabilities vector

    Returns:
      results: metric values
    """
    try:
        results = metrics_fn(labels, probs)
    except ValueError:
        results = None

    return results


def auprc(labels, probs):
    """Wrapper for sklearn AUPRC

    Args:
      labels: 1D vector of labels
      probs: 1D vector of probs

    Returns:
      auprc: area under precision-recall curve
    """
    pr_curve = precision_recall_curve(labels, probs)
    precision, recall = pr_curve[:2]
    
    return auc(recall, precision)


def make_recall_at_fdr(fdr):
    """Construct function to get recall at FDR
    
    Args:
      fdr: FDR value for precision cutoff

    Returns:
      recall_at_fdr: Function to generate the recall at
        fdr fraction (number of positives
        correctly called as positives out of total 
        positives, at a certain FDR value)
    """
    def recall_at_fdr(labels, probs):
        pr_curve = precision_recall_curve(labels, probs)
        precision, recall = pr_curve[:2]
        return recall[np.searchsorted(precision - fdr, 0)]
        
    return recall_at_fdr


def run_sklearn_curve_fn(curve_fn, labels, probs):
    """Wrap curve functions in case eval data missing classes
    
    Args:
      curve_fn: function that generates curve values
      labels: 1D vector of labels
      probs: 1D vector of probabilities

    Returns:
      x: x values vector
      y: y values vector
      thresh: thresholds at various levels
    """
    try:
        x, y, thresh = curve_fn(labels, probs)
    except:
        x, y, thresh = np.zeros((1)), np.zeros((1)), None

    return x, y, thresh


def save_plotting_data(labels, probs, out_file, curve="ROC"):
    """Runs ROC or PR and writes out curve data to text file
    in standardized way
    
    Args:
      labels: 1D vector of labels
      probs: 1D vector of probabilities
      out_file: output text file
      curve: what kind of curve. ROC or PR

    Returns:
      None
    """
    if curve == "ROC":
        fpr, tpr, _ = run_sklearn_curve_fn(roc_curve, labels, probs)
        plotting_df = pd.DataFrame(
            data=np.stack([fpr, tpr], axis=1),
            columns=["x", "y"])
    elif curve == "PR":
        precision, recall, _ = run_sklearn_curve_fn(
            precision_recall_curve, labels, probs)
        plotting_df = pd.DataFrame(
            data=np.stack([recall, precision], axis=1),
            columns=["x", "y"])
    else:
        print "Unknown curve type!"
        return
        
    plotting_df.to_csv(out_file, sep='\t', index=False)
    return None


def run(args):
    """Pipeline to evaluate a trained model
    
    Sets up model graphs and then runs the graphs for some number of batches
    to get predictions. Then builds out curves to plot and gets definitive
    metric values.
    """
    logging.info("Running evaluation...")
    os.system("mkdir -p {}".format(args.out_dir))
    os.system("mkdir -p {0}/{1}".format(args.out_dir, args.prefix))
    
    # find data files
    data_files = sorted(glob.glob("{}/*.h5".format(args.data_dir)))

    # set up model params
    model_params = setup_model_params(args)
    
    # set up neural network graph
    tronn_graph = TronnNeuralNetGraph(
        {"data": data_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size,
        models[args.model['name']],
        model_params,
        tf.nn.sigmoid)

    # predict
    labels, predictions, probs, metadata = predict(
        tronn_graph,
        args.model_dir,
        args.batch_size,
        num_evals=args.num_evals)

    # setup metrics functions
    precision_thresholds = [0.5, 0.75, 0.9, 0.95]
    metrics_functions = [roc_auc_score, auprc] + [
        make_recall_at_fdr(fdr)
        for fdr in precision_thresholds]
    metrics_functions_names = ["AUROC", "AUPRC"] + [
        "recall_at_{}_fdr".format(
            str(int(round(100.*(1. - fdr)))))
        for fdr in precision_thresholds]

    
    metrics_array = np.zeros(
        (labels.shape[1]+1, len(metrics_functions))
    )
    index_list = []
    
    # sklearn metrics global
    index_list.append("global")
    for metric_idx in range(len(metrics_functions)):
        try:
            metrics_array[0, metric_idx] = run_sklearn_metric_fn(
                metrics_functions[metric_idx],
                labels.flatten(), probs.flatten())
        except:
            metrics_array[0, metric_idx] = 0.

        # plotting: grab AUROC and AUPRC curves
        global_auroc_plot_file = "{0}/{1}/auroc.global.tmp.txt".format(
            args.out_dir, args.prefix)
        save_plotting_data(
            labels.flatten(), probs.flatten(),
            global_auroc_plot_file, curve="ROC")
        global_auprc_plot_file = "{0}/{1}/auprc.global.tmp.txt".format(
            args.out_dir, args.prefix)
        save_plotting_data(
            labels.flatten(), probs.flatten(),
            global_auprc_plot_file, curve="PR")
        
    # sklearn metrics per task
    for task_idx in range(labels.shape[1]):
        task_key = "task_{}".format(task_idx)
        index_list.append(task_key)

        task_labels = labels[:,task_idx]
        task_probs = probs[:,task_idx]
        
        for metric_idx in range(len(metrics_functions)):
            try:
                metrics_array[1 + task_idx, metric_idx] = run_sklearn_metric_fn(
                    metrics_functions[metric_idx],
                    task_labels, task_probs)
            except:
                metrics_array[1 + task_idx, metric_idx] = 0.

        # plotting: grab AUROC and AUPRC curves
        task_auroc_plot_file = "{0}/{1}/auroc.{2}.tmp.txt".format(
            args.out_dir, args.prefix, task_key)
        save_plotting_data(
            task_labels, task_probs,
            task_auroc_plot_file, curve="ROC")
        task_auprc_plot_file = "{0}/{1}/auprc.{2}.tmp.txt".format(
            args.out_dir, args.prefix, task_key)
        save_plotting_data(
            task_labels, task_probs,
            task_auprc_plot_file, curve="PR")

    metrics_df = pd.DataFrame(
        data=metrics_array,
        index=index_list,
        columns=metrics_functions_names)
    metrics_cleaned_df = metrics_df.fillna(value=0)

    # save out file
    out_table_file = "{0}/{1}/{1}.metrics_summary.txt".format(
        args.out_dir, args.prefix)
    metrics_cleaned_df.to_csv(out_table_file, sep='\t')

    # call R function to plot AUROC
    auroc_plot = "{0}/{1}/auroc.all.plot.png".format(
        args.out_dir, args.prefix)
    plot_auroc = ("plot_metrics_curves.R AUROC FPR TPR {0} "
                  "{1}/{2}/auroc.*.txt").format(
                      auroc_plot, args.out_dir, args.prefix)
    print plot_auroc
    os.system(plot_auroc)

    # call R function to plot AUPRC
    auprc_plot = "{0}/{1}/auprc.all.plot.png".format(
        args.out_dir, args.prefix)
    plot_auprc = ("plot_metrics_curves.R AUPRC Recall Precision {0} "
                  "{1}/{2}/auprc.*.txt").format(
                      auprc_plot, args.out_dir, args.prefix)
    print plot_auprc
    os.system(plot_auprc)

    return None
