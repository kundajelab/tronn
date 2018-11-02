# description parametric stats/tests

import numpy as np

from scipy.stats import hypergeom


def run_hypergeometric_test(
        positives_array, background_array):
    """calculate pvals
    """
    num_positives = positives_array.shape[0]
    num_background = background_array.shape[0]
    num_motifs = positives_array.shape[1]

    pvals = np.ones((num_motifs))
    
    for i in xrange(num_motifs):
        hits_in_positives = np.sum(positives_array[:,i] > 0)
        hits_in_background = np.sum(background_array[:,i] > 0)
        pvals[i] = hypergeom.sf(
            hits_in_positives-1,
            num_background,
            hits_in_background,
            num_positives)
    
    return pvals
