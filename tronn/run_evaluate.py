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
from tronn.nets.nets import model_fns
from tronn.learn.cross_validation import alt_setup_cv
from tronn.learn.cross_validation import setup_cv
from tronn.learn.learning import predict

from tronn.run_predict import setup_model
from tronn.run_predict import scores_to_probs



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


def run_metrics_functions(labels, probs, metrics_functions, metrics_array, row_idx):
    """Run a series of metrics functions on the labels and probs

    Args:
      labels: 1D vector of labels
      probs: 1D vector of probabilities
      metrics_functions: dict of metric functions
      metrics_array: numpy array to store results

    Returns:
      metrics_array
    """
    for metric_idx in range(len(metrics_functions)):
        try:
            metrics_array[row_idx, metric_idx] = run_sklearn_metric_fn(
                metrics_functions[metric_idx],
                labels, probs)
        except:
            metrics_array[0, metric_idx] = 0.
            
    return metrics_array


def save_plotting_data(labels, probs, out_file, curve="auprc"):
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
    if curve == "auroc":
        fpr, tpr, _ = run_sklearn_curve_fn(roc_curve, labels, probs)
        plotting_df = pd.DataFrame(
            data=np.stack([fpr, tpr], axis=1),
            columns=["x", "y"])
    elif curve == "auprc":
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


def run_and_plot_metrics(
        labels,
        probs,
        metrics_functions,
        metrics_array,
        metrics_row_idx,
        plot_fn_names,
        plot_folder,
        prefix):
    """Wrapper to run all possible metrics types on a set of labels and probs
    """
    # run metrics for table
    run_metrics_functions(
        labels,
        probs,
        metrics_functions,
        metrics_array,
        metrics_row_idx)
    
    # run metrics for plotting
    for plot_fn_name in plot_fn_names:
        os.system("mkdir -p {0}/{1}".format(
            plot_folder, plot_fn_name))
        plot_file = "{0}/{1}/{2}.{1}.tmp.txt".format(
            plot_folder, plot_fn_name, prefix)
        save_plotting_data(
            labels, probs,
            plot_file,
            curve=plot_fn_name)
    
    return


def plot_all(plot_folder, prefix, param_sets):
    """Use R to make pretty plots, uses all files in a folder
    """
    for param_key in param_sets.keys():
        plot_file = "{0}/{1}/{2}.{1}.all.plot.png".format(plot_folder, param_key, prefix)
        plot_cmd = ("plot_metrics_curves.R {0} {1} {2}/{3}/*.txt").format(
            param_sets[param_key], plot_file, plot_folder, param_key)
        print plot_cmd
        os.system(plot_cmd)

    return


def run(args):
    """Pipeline to evaluate a trained model
    
    Sets up model graphs and then runs the graphs for some number of batches
    to get predictions. Then builds out curves to plot and gets definitive
    metric values.
    """
    logging.info("Running evaluation...")
    os.system("mkdir -p {}".format(args.out_dir))
    
    # find data files
    # NOTE right now this is technically validation set
    data_files = sorted(glob.glob("{}/*.h5".format(args.data_dir)))
    if(args.cvfile is not None):
        train_files, valid_files, test_files = alt_setup_cv(data_files, cvfold=args.cvfold, cvfile=args.cvfile)
    else:
        train_files, valid_files, test_files = setup_cv(data_files, cvfold=args.cvfold)

    # set up model params
    model_fn, model_params = setup_model(args)
    
    # set up neural network graph
    if args.reconstruct_regions:
        shuffle_data = False
    else:
        shuffle_data = True
    
    tronn_graph = TronnNeuralNetGraph(
        {"data": test_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size,
        model_fn,
        model_params,
        tf.nn.sigmoid,
        shuffle_data=shuffle_data)

    # predict
    labels, predictions, probs, metadata = predict(
        tronn_graph,
        args.model_dir,
        args.batch_size,
        model_checkpoint=args.model_checkpoint,
        num_evals=args.num_evals,
        reconstruct_regions=args.reconstruct_regions)

    # push predictions through activation to get probs
    if args.model_type != "nn":
        probs = scores_to_probs(predictions)

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
