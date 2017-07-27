"""Description: Contains functions for evaluation and summarization of metrics

"""

import numpy as np
import tensorflow as tf

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score



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



# TODO use TFlearn random forest. To do this, make a preprocess to kmer hdf5 file and new datalayer for that input





# TODO determine format for prediction files. definitely save them out so that can plot curves. hdf5 for prediction vectors, text for AUROC curves



def calculate_standard_metrics():
    """Summary function to calculate AUROC, AUPRC, recall at precision thresh
    """


    return None
