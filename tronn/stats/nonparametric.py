# description: non parametric stats/tests


import numpy as np



def select_features_by_permutation_test_old(
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
    array_extra_dim = np.expand_dims(array, axis=2) # {N, real_M, 1}

    # run permutations
    start_idx = 0
    for i in xrange(num_permutes):
        
        # permute
        random_vals = np.random.random(array.shape)
        idx = np.argsort(random_vals, axis=1)
        
        array_permuted = array[np.arange(array.shape[0])[:, None], idx] # {N, M}
        array_permuted = np.expand_dims(array_permuted, axis=1) # {N, 1, perm_M}
        
        # calculate diffs
        diffs = np.subtract(array_extra_dim, array_permuted) # {N, real_M, perm_M}

        # reduce
        diffs = np.sum(diffs, axis=0) # {real_M, perm_M}

        # save out
        if i != (num_permutes - 1):
            stop_idx = start_idx + num_features
            results[:,start_idx:stop_idx] = diffs
        else:
            stop_idx = num_shuffles
            delta_idx = stop_idx - start_idx
            results[:,start_idx:stop_idx] = diffs[:,0:delta_idx]
                
        # increment
        start_idx += num_features

    # count how often the diff is 0 or less
    # results {real_M, shuffles}
    import ipdb
    ipdb.set_trace()
    
    results = np.mean(results <= 0, axis=1) # {real_M}
    results = (results <= pval_thresh).astype(int)
        
    return results


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
        diffs = np.subtract(np.median(array, axis=0), np.median(array_permuted, axis=0)) # {real_M}
        
        # reduce
        #diffs = np.mean(diffs, axis=0) # {real_M}
        results[:,i] = diffs

    # count how often the diff is 0 or less
    # results {real_M, shuffles}
    results = np.mean(results <= 0, axis=1) # {real_M}
    results = (results <= pval_thresh).astype(int)
        
    return results
