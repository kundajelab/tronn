# Description: just helper functions for
# consistent cross validation across functions

import h5py

import numpy as np

# figure out how to build better cross validation sets

from tronn.datalayer import H5DataLoader


# todo - build leave one out
# todo - build k fold
# todo - build fractional
# todo - build manual

def _get_chrom_size_ordered(data_dict, positives_only=False):
    """returns an ordered list of chrom keys 
    based on size of files
    """
    chrom_keys = []
    num_examples_per_chrom = []
    for chrom in sorted(data_dict.keys()):
        chrom_keys.append(chrom)
        num_positives = H5DataLoader.get_num_examples(data_dict[chrom][0])
        training_negatives = H5DataLoader.get_num_examples(data_dict[chrom][1])
        #global_negatives = H5DataLoader.get_num_examples(data_dict[chrom][2])
        num_examples_per_chrom.append(num_positives + training_negatives)
        
    # order in descending example total
    num_examples_per_chrom = np.array(num_examples_per_chrom)
    indices = np.argsort(num_examples_per_chrom)[::-1]
    ordered_chrom_keys = np.array(chrom_keys)[indices]
    ordered_num_examples = num_examples_per_chrom[indices]

    return ordered_chrom_keys, ordered_num_examples


def setup_kfold_cv(data_dict, k):
    """given k, split the files as equally as possible across k
    """
    # set up kfolds dict
    kfolds = {}
    examples_per_fold = {}
    for k_idx in xrange(k):
        kfolds[k_idx] = [[], [], []]
        examples_per_fold[k_idx] = 0
        
    # reorder according to biggest file first
    ordered_chrom_keys, num_examples_per_file = _get_chrom_size_ordered(data_dict)
    
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
        
    return kfolds, examples_per_fold


def setup_train_valid_test(h5_files, k, valid_folds=[], test_folds=[]):
    """set up folds
    """
    kfolds, examples_per_fold = setup_kfold_cv(h5_files, k)
    
    # fill in folds as needed
    if len(test_folds) == 0:
        test_folds = [k - 1]

    if len(valid_folds) == 0:
        valid_folds = [test_folds[-1] - 1]
        assert valid_folds[0] not in test_folds

    train_folds = [i for i in xrange(k) if i not in test_folds + valid_folds]

    # training: get positives and training negatives
    train_files = []
    for i in train_folds:
        train_files += kfolds[i][0]
        train_files += kfolds[i][1]

    # validation: get positives and training negatives
    valid_files = []
    for i in valid_folds:
        valid_files += kfolds[i][0]
        valid_files += kfolds[i][1]

    # test: get positives and genomewide negatives
    test_files = []
    for i in test_folds:
        test_files += kfolds[i][0]
        if len(kfolds[i][2]) > 0:
            test_files += kfolds[i][2]
        else:
            test_files += kfolds[i][1]
            
    return train_files, valid_files, test_files




def setup_cv(data_files, cvfold=0):
    """Helper function to choose good CV folds
    Just hard-coded 3 cv folds
    """
    # never train with chrY
    data_files = sorted([data_file for data_file in data_files if "chrY" not in data_file])

    # 1 for validation, 2 for test, never chrX
    cv_start_indices = [19, 1, 7, 4, 16]
    cv_start_idx = cv_start_indices[cvfold]

    # set up splits
    train_files = list(data_files)
    del train_files[cv_start_idx:cv_start_idx+3]
    valid_files = [data_files[cv_start_idx]]
    test_files = data_files[cv_start_idx+1:cv_start_idx+3]

    # logging
    print "Train:", [filename.split(".")[-2] for filename in train_files]
    print "Validation:", [filename.split(".")[-2] for filename in valid_files]
    print "Test:", [filename.split(".")[-2] for filename in test_files]
    
    return train_files, valid_files, test_files

