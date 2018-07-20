# description: non parametric stats/tests


import numpy as np


def select_features_by_permutation_test(
        array,
        num_shuffles=1000,
        pval_thresh=0.01):
    """shuffle the features for each example
    (shuffle the rows) to determine whether
    each feature column is much higher scoring
    relative to the null (break the column/feature 
    relationship)
    """
    # set up
    num_features = array.shape[1]
    num_permutes = int(np.ceil(num_shuffles / float(num_features)))
    results = np.zeros((num_features, num_shuffles)) # {real_M, shuffles}

    # run permutations
    start_idx = 0
    for i in xrange(num_shuffles):

        #if i % 100 == 0:
        #    print i
        
        # permute
        random_vals = np.random.random((array.shape[0]))
        idx = np.floor(np.multiply(array.shape[1], random_vals)).astype(int)
        array_permuted = np.expand_dims(
            array[np.arange(array.shape[0]), idx], axis=1) # {N, 1}

        # calculate diffs. stabilized by median since it's noisy
        #diffs = np.subtract(array, array_permuted) # {N, real_M}
        #diffs = np.subtract(np.median(array, axis=0), np.median(array_permuted, axis=0)) # {real_M}
        diffs = np.subtract(np.mean(array, axis=0), np.mean(array_permuted, axis=0)) # {real_M}
        
        # reduce
        #diffs = np.mean(diffs, axis=0) # {real_M}
        results[:,i] = diffs

    # count how often the diff is 0 or less
    results = np.mean(results <= 0, axis=1) # {real_M}
    results = (results <= pval_thresh).astype(int)
        
    return results
