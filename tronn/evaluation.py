""" Contains functions for evaluation and summarization of metrics

"""

import math
import tensorflow as tf
import numpy as np
import h5py

from sklearn import metrics as skmetrics


def get_global_avg_metrics(labels, probabilities, tasks=[]):
    """Get global metric values: predictions, mean metric values
    """
    predictions = tf.cast(tf.greater(probabilities, 0.5), 'float32')
    metric_map = {'mean_auroc': tf.metrics.auc(labels, probabilities, curve='ROC', name='mean_auroc'),
                  'mean_auprc': tf.metrics.auc(labels, probabilities, curve='PR', name='mean_auprc'),
                  'mean_accuracy': tf.metrics.accuracy(labels, predictions, name='mean_accuracy')}
    metric_value, metric_updates = tf.contrib.metrics.aggregate_metric_map(metric_map)
    update_ops = metric_updates.values()
    return metric_value, update_ops

