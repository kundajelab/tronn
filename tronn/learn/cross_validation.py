# Description: just helper functions for
# consistent cross validation across functions


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
