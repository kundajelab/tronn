"""Description: Contains functions for evaluation and summarization of metrics
"""

import numpy as np
import tensorflow as tf


def get_global_avg_metrics(labels, probabilities, tasks=[]):
    """Get global metric values: predictions, mean metric values
    Note that the inputs must be tensors!
    """
    predictions = tf.cast(tf.greater(probabilities, 0.5), 'float32')
    metric_map = {'mean_auroc': tf.metrics.auc(labels, probabilities, curve='ROC', name='mean_auroc'),
                  'mean_auprc': tf.metrics.auc(labels, probabilities, curve='PR', name='mean_auprc'),
                  'mean_accuracy': tf.metrics.accuracy(labels, predictions, name='mean_accuracy')}
    metric_value, metric_updates = tf.contrib.metrics.aggregate_metric_map(metric_map)
    update_ops = metric_updates.values()
    return metric_value, update_ops


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
