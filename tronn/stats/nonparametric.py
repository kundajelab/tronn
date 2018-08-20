# description: non parametric stats/tests


import numpy as np

from multiprocessing import Pool
from scipy.stats import rankdata


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


def get_pvals_from_delta_shuffles(
        shuffle_data,
        true_data,
        twotailed=True):
    """calculate pvals. this applies to the case where you use the 
    deltas - which means that the mean is near zero, so the values at the extremes
    both apply
    """
    # set up data
    true_data = np.expand_dims(true_data, axis=0)
    true_w_shuffles = np.concatenate((true_data, shuffle_data), axis=0)

    if twotailed:
        # calculate first in the negative direction
        ranks = np.apply_along_axis(rankdata, 0, true_w_shuffles)
        neg_dir_pvals = 2 * ranks[0] / float(true_w_shuffles.shape[0])

        # and calculate in the positive direction
        ranks = true_w_shuffles.shape[0] - ranks + 1
        pos_dir_pvals = 2 * ranks[0] / float(true_w_shuffles.shape[0])

        # and take the min of both
        pvals = np.minimum(neg_dir_pvals, pos_dir_pvals)
        
    else:
        ranks = np.apply_along_axis(rankdata, 0, true_w_shuffles)
        pvals = ranks[0] / float(true_w_shuffles.shape[0])

    return pvals


def threshold_by_qvalues(
        pvals,
        qval_thresh=0.50,
        num_bins=50):
    """adjust for FDR
    """
    # flatten the results
    pvals_flattened = np.sort(pvals.flatten())
    num_pvals_per_bin, pval_bins = np.histogram(
        pvals_flattened, bins=num_bins)

    # get baseline
    baseline = np.mean(num_pvals_per_bin[-2:])
    
    # then get the cumulative sum of pvalue dist
    p_cumsum = np.cumsum(num_pvals_per_bin)

    # then get a cumulative sum of the uniform dist
    baseline_uniform = baseline * np.ones(num_pvals_per_bin.shape)
    baseline_cumsum = np.cumsum(baseline_uniform)
    
    # divide the uniform cumsum by the cumsum of pvalues
    qvals = baseline_cumsum / p_cumsum

    # find the point where the FDR crosses the qval_thresh
    pval_cutoff_idx = np.searchsorted(qvals, qval_thresh)
    pval_cutoff = pval_bins[pval_cutoff_idx]

    # give back thresholds
    thresholds = pvals <= pval_cutoff
    
    return thresholds


def run_delta_permutation_test(
        array,
        num_shuffles=1000,
        pval_thresh=0.05, # 0.01
        twotailed=True,
        qvalue=True,
        qval_thresh=0.50):
    """when given an array, randomly
    flip the sign and recalculate the sum (across last axis)
    consider significant if actual sum is above the 
    pval_thresh
    
    NOTE: since this is a delta test, the NULL is that the difference
    is zero.

    important: better to set a loose FDR, and then utilize the 
    fact that there are multiple tasks to then give you more information
    """
    true_sums = np.sum(array, axis=0) # ex {mutM, task, M}
    results_shape = list(array.shape) # ex {N, mutM, task, M}
    results_shape[0] = num_shuffles # ex {num_shuffles, mutM, task, M}
    results = np.zeros(results_shape) # ex {num_shuffles, mutM, task, M}

    # do shuffles
    for i in xrange(num_shuffles):
        if i % 100 == 0:
            print i
            
        random_vals = np.random.random((array.shape)) # ex {N, mutM, task, M}
        flips = -(random_vals <= 0.5).astype(int) 
        noflips = (random_vals > 0.5).astype(int)
        random_flips = np.add(flips, noflips)
        
        shuffle_results = np.multiply(random_flips, array)
        shuffle_results = np.sum(shuffle_results, axis=0) # ex {mutM, task, M}

        results[i] = shuffle_results

    # calculate pvals
    pvals = get_pvals_from_delta_shuffles(
        results,
        true_sums,
        twotailed=twotailed) # {mutM, task, M}
    
    # get cutoffs
    if qvalue:
        passed_threshold = threshold_by_qvalues(pvals, qval_thresh=qval_thresh)
    else:
        passed_threshold = pvals < pval_thresh

    # and then filter by task
    # rationale: if there is a 50% FDR rate, then for
    # any given hypothesis, need at least 2 hits across tasks
    # such that at least 1 of them was likely real
    num_tasks = passed_threshold.shape[1]
    sig_task_threshold = min(num_tasks - 1, (1. /(1 - qval_thresh)))
    passed_threshold_cross_task = np.sum(
        passed_threshold, axis=1, keepdims=True) >  sig_task_threshold
    passed_threshold = passed_threshold_cross_task
    #passed_threshold = np.multiply(
    #    passed_threshold_cross_task,
    #    passed_threshold)

    return pvals, passed_threshold
