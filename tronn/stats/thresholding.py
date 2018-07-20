# code for performing thresholding

import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def threshold_at_recall(labels, scores, recall_thresh=1.0):
    """Get highest threshold at which there is recall_thresh level
    of recall
    """
    pr_curve = precision_recall_curve(labels, scores)
    precision, recall, thresholds = pr_curve
    threshold_index = np.searchsorted(1 - recall, 1 - recall_thresh)
    #print "precision at thresh", precision[index]
    #print "recall at thresh", recall[threshold_index]
    try:
        return thresholds[threshold_index]
    except:
        return 0 # TODO figure out what to do here...


def threshold_at_fdr(labels, scores, fdr=0.25):
    pr_curve = precision_recall_curve(labels, scores)
    precision, recall, thresholds = pr_curve
    threshold_index = np.searchsorted(precision, 1-fdr)
    try:
        return thresholds[threshold_index]
    except:
        return 0 # TODO figure out what to do here...




