# Description: just helper functions for
# consistent cross validation across functions

import os
import numpy as np

def setup_cv(data_files, cvfold=0):
    """Helper function to choose good CV folds
    Just hard-coded 3 cv folds
    """
    # 1 for validation, 2 for test, never chrX or Y (or tiny ones)
    cv_start_indices = [1, 7, 16]
    cv_start_idx = cv_start_indices[cvfold]
    
    train_files = list(data_files)
    del train_files[cv_start_idx:cv_start_idx+3]
    valid_files = [data_files[cv_start_idx]]
    test_files = data_files[cv_start_idx+1:cv_start_idx+3]

    return train_files, valid_files, test_files

def alt_setup_cv(data_files, cvfold=0, cvfile=None, criteria='dataBased'):
	"""Helper function to choose chromosomes for training, validation and testing
    	The table format assumption: 
		chrName: chromosome name
		dataSize: number of examples for the chromosome in the intervals file
		dataBased: indicator of train/valid/test and foldNumber - indicators are 1-based not 0-based - valid_indicators are -ve - test_indicators are +ve 
		chrSize: chromosome size, 
		chrBased: indicator of train/valid/test and foldNumber based on chr size - indicators are 1-based not 0-based - valid_indicators are -ve - test_indicators are +ve 
    	"""
	print(cvfile)
	assert os.path.exists(cvfile), "Need a table for cross validation: crossValidationTable.txt"
	
	result = np.genfromtxt(cvfile, names=True, dtype=None)
	
	train_files = []
	valid_files = []
	test_files = []
	for i in range(len(result)):
		currentChr = "." + result[i]['chrName'] + "."
		indx = [idx for idx, dfl in enumerate(data_files) if currentChr in dfl][0]
		
		# valid_indicators are -ve - test_indicators are +ve 
		if(result[i][criteria] == (cvfold+1)): #indicators are 1-based not 0-based 
			#test_indxs.append(indx)
			test_files.append(data_files[indx])  
			continue 

		# valid_indicators are -ve - test_indicators are +ve 
		if(np.abs(result[i][criteria]) == (cvfold+1)): #indicators are 1-based not 0-based 
			#valid_indxs.append(indx)
			valid_files.append(data_files[indx])
			continue

		#train_indxs.append(indx)
		train_files.append(data_files[indx]) 	
	
    	return train_files, valid_files, test_files
