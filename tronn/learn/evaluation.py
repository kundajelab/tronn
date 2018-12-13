"""Description: Contains functions for evaluation and summarization of metrics
"""

import os

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.contrib.metrics import streaming_pearson_correlation

from scipy.stats import spearmanr
from scipy.stats import pearsonr

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from tronn.util.tf_utils import setup_tensorflow_session
from tronn.util.tf_utils import close_tensorflow_session


def get_global_avg_metrics(labels, probabilities, num_thresholds=100):
    """Get global metric values: predictions, mean metric values
    Note that the inputs must be tensors!
    """
    predictions = tf.cast(tf.greater(probabilities, 0.5), 'float32')
    metric_map = {
        'mean_auroc': tf.metrics.auc(
            labels,
            probabilities,
            num_thresholds=num_thresholds,
            curve='ROC',
            name='mean_auroc',
            summation_method="careful_interpolation"),
        'mean_auprc': tf.metrics.auc(
            labels,
            probabilities,
            num_thresholds=num_thresholds,
            curve='PR',
            name='mean_auprc',
            summation_method="careful_interpolation"),
        'mean_accuracy': tf.metrics.accuracy(
            labels,
            predictions,
            name='mean_accuracy')}
    tf.add_to_collection(
        "auprc",
        metric_map["mean_auprc"][0])
    
    return metric_map


def _streaming_spearman_correlation(predictions, labels):
    """
    """
    batch_size = predictions.get_shape().as_list()[0]
    
    def _rank_order(tensor):
        """masks the top_k fn
        """
        # indices is sorted in descending order
        _, indices = tf.nn.top_k(tensor, k=batch_size, sorted=True)

        # one more top_k to get the rank vector
        _, rank_vals = tf.nn.top_k(indices, k=batch_size, sorted=True)
        rank_vals = tf.cast(rank_vals, tf.float32)
        
        return rank_vals

    prediction_ranks = tf.map_fn(
        _rank_order,
        tf.transpose(predictions))
    prediction_ranks = tf.transpose(prediction_ranks)
    
    label_ranks = tf.map_fn(
        _rank_order,
        tf.transpose(labels))
    label_ranks = tf.transpose(label_ranks)

    # TODO consider how to deal with update ops
    def _streaming_pearson(inputs):
        """wrapper
        """
        predictions = inputs[0]
        labels = inputs[0]
        
        return None
    
    return streaming_pearson_correlation(prediction_ranks, label_ranks)


def get_regression_metrics(labels, probabilities):
    """get global metrics
    """
    metric_map = {
        "mse": tf.metrics.mean_squared_error(labels, probabilities),
        "spearman_cor": _streaming_spearman_correlation(probabilities, labels)}
    tf.add_to_collection(
        "mse",
        metric_map["mse"][0])
    
    return metric_map


def spearman_only_r(labels, probs):
    """spearman rank correlation wrapper
    """
    results = spearmanr(labels, probs)
    
    return results[0]


def pearson_only_r(labels, probs):
    """pearson correlation wrapper
    """
    results = pearsonr(labels, probs)
    
    return results[0]


def get_confusion_matrix(labels, predictions, metric="AUPRC", fdr=0.05):
    """Given a set of predictions and not knowing which label set it may
    best predict (ie, unsupervised learning), produce the confusion matrix
    Note that the inputs are numpy style arrays
    """
    num_label_sets = labels.shape[1]
    num_prediction_sets = predictions.shape[1]
    
    confusion_matrix = np.zeros(
        (num_prediction_sets, num_label_sets))

    for prediction_set_idx in xrange(num_prediction_sets):
        prediction_set = predictions[:,prediction_set_idx]
        
        for label_set_idx in xrange(num_label_sets):
            label_set = labels[:, label_set_idx]
            
            # metric
            if metric == "auprc":
                pass
            elif metric == "auroc":
                pass
            elif metric == "recall":
                # use the recall
                pass
            else:
                raise Exception("Metric is not defined!")

            # save out
            confusion_matrix[prediction_set_idx, label_set_idx] = metric
            
    return confusion_matrix



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
    elif curve == "correlation":
        plotting_df = pd.DataFrame(
            data=np.stack([labels.flatten(), probs.flatten()], axis=1),
            columns=["x", "y"])
    else:
        # TODO save out labels and probs
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

