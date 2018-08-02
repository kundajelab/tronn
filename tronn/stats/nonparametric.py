# description: non parametric stats/tests


import numpy as np

from multiprocessing import Pool


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
    # this is confidence interval - sort of SE of mean
    results = np.mean(results <= 0, axis=1) # {real_M}
    results = (results <= pval_thresh).astype(int)
        
    return results


def random_flip_and_sum(array):
    """random flip the sign and return sum
    set up this way for multiprocessing pool
    """
    array = input_list
    
    random_vals = np.random.random((array.shape)) # ex {N, mutM, task, M}
    flips = -(random_vals < 0.5).astype(int) 
    noflips = (random_vals >=0.5).astype(int)
    random_flips = np.add(flips, noflips)
    
    shuffle_results = np.multiply(random_flips, array)
    shuffle_results = np.sum(shuffle_results, axis=0) # ex {mutM, task, M}

    return shuffle_results



def run_delta_permutation_test(
        array,
        num_shuffles=1000,
        pval_thresh=0.0001, # 0.01
        twotailed=True):
    """when given an array, randomly
    flip the sign and recalculate the sum (across last axis)
    consider significant if actual sum is above the 
    pval_thresh
    """
    #num_hits = np.sum(array!=0, axis=0) # {mutM, task, M} # question is, how do i know
    true_sums = np.sum(array, axis=0) # ex {mutM, task, M}
    
    results_shape = list(array.shape) # ex {N, mutM, task, M}
    results_shape[0] = num_shuffles # ex {num_shuffles, mutM, task, M}
    results = np.zeros(results_shape) # ex {num_shuffles, mutM, task, M}
    
    for i in xrange(num_shuffles):
        if i % 100 == 0:
            print i
            
        random_vals = np.random.random((array.shape)) # ex {N, mutM, task, M}
        flips = -(random_vals < 0.5).astype(int) 
        noflips = (random_vals >=0.5).astype(int)
        random_flips = np.add(flips, noflips)
        
        shuffle_results = np.multiply(random_flips, array)
        shuffle_results = np.sum(shuffle_results, axis=0) # ex {mutM, task, M}

        results[i] = shuffle_results
        
    # move axis back here
    results = np.moveaxis(results, 0, -1) # ex {mutM, task, M, shuf}

    # get the pval thresh percentile values
    thresholds = np.percentile(results, 100.*(1-pval_thresh), axis=-1) # {mutM, task, M}
    
    # apply the threshold
    positive_pass = np.greater(true_sums, thresholds)
    negative_pass = np.less(true_sums, -thresholds)
    passed_threshold = np.add(positive_pass, negative_pass)

    return passed_threshold
