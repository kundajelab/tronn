# Description: just helper functions for
# consistent cross validation across functions

import h5py

import numpy as np


def setup_kfolds(
        data_dict,
        ordered_chrom_keys,
        num_examples_per_file,
        k):
    """given k, split the files as equally as possible across k
    """
    # set up kfolds dict
    kfolds = {}
    examples_per_fold = {}
    for k_idx in xrange(k):
        kfolds[k_idx] = [[], [], []]
        examples_per_fold[k_idx] = 0
    
    # now initialize with first files
    for k_idx in xrange(k):
        chrom_key = ordered_chrom_keys[k_idx]
        kfolds[k_idx][0] += data_dict[chrom_key][0]
        kfolds[k_idx][1] += data_dict[chrom_key][1]
        kfolds[k_idx][2] += data_dict[chrom_key][2]
        examples_per_fold[k_idx] += num_examples_per_file[k_idx]
        
    # given k buckets, go from biggest to smallest,
    # always filling in the bucket that has the smallest num of examples
    for i in xrange(k, len(data_dict.keys())):
        fill_k = 0
        least_examples_k = examples_per_fold[0]
        
        # check which has fewest examples
        for k_idx in xrange(1, k):
            fold_examples = examples_per_fold[k_idx]
            if fold_examples < least_examples_k:
                fill_k = k_idx
                least_examples_k = fold_examples

        # append to that one
        chrom_key = ordered_chrom_keys[i]
        kfolds[fill_k][0] += data_dict[chrom_key][0]
        kfolds[fill_k][1] += data_dict[chrom_key][1]
        kfolds[fill_k][2] += data_dict[chrom_key][2]
        examples_per_fold[fill_k] += num_examples_per_file[i]

    # debug tool print reduced set
    if False:
        kfold_print = {}
        total_examples = 0
        for key in kfolds.keys():
            chroms = list(set([filename.split(".")[-4] for filename in kfolds[key][0]]))
            kfold_print[key] = (chroms, examples_per_fold[key])
            total_examples += examples_per_fold[key]
        print kfold_print
        print total_examples / float(k)
        
    return kfolds, examples_per_fold


def setup_train_valid_test(
        chrom_file_dict,
        ordered_chrom_keys,
        num_examples_per_chrom,
        k,
        valid_folds=[],
        test_folds=[],
        regression=False):
    """set up folds
    """
    kfolds, examples_per_fold = setup_kfolds(
        chrom_file_dict,
        ordered_chrom_keys,
        num_examples_per_chrom,
        k)
    
    # fill in folds as needed
    if len(test_folds) == 0:
        test_folds = [k - 1]

    if len(valid_folds) == 0:
        valid_folds = [test_folds[0] - 1]
        assert valid_folds[0] not in test_folds

    train_folds = [i for i in xrange(k) if i not in test_folds + valid_folds]

    # training: get positives and training negatives
    train_files = []
    for i in train_folds:
        train_files += kfolds[i][0]
        if not regression:
            train_files += kfolds[i][1]

    # validation: get positives and training negatives
    valid_files = []
    for i in valid_folds:
        valid_files += kfolds[i][0]
        if not regression:
            valid_files += kfolds[i][1]

    # test: get positives and genomewide negatives
    test_files = []
    for i in test_folds:
        test_files += kfolds[i][0]
        if not regression:
            if len(kfolds[i][2]) > 0:
                test_files += kfolds[i][2]
            else:
                test_files += kfolds[i][1]
            
    return train_files, valid_files, test_files
