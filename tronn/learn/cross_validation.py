# Description: just helper functions for
# consistent cross validation across functions



def setup_cv(data_files, cvfold=0):
    """Helper function to choose good CV folds
    Just hard-coded 3 cv folds
    """
    # never train with chrY
    data_files.remove("chrY")
    data_files = sorted(data_files)

    # 1 for validation, 2 for test, never chrX or Y (or tiny ones)
    cv_start_indices = [1, 7, 16]
    cv_start_idx = cv_start_indices[cvfold]
    
    train_files = list(data_files)
    del train_files[cv_start_idx:cv_start_idx+3]
    valid_files = [data_files[cv_start_idx]]
    test_files = data_files[cv_start_idx+1:cv_start_idx+3]
    
    print "Train:", train_files
    print "Validation:", valid_files
    print "Test:", test_files

    return train_files, valid_files, test_files
