# description: some helpful utils for working with h5 files

import h5py

import numpy as np
import pandas as pd


def h5_dataset_to_text_file(h5_file, key, text_file, col_keep_indices, colnames):
    """Grab a dataset out of h5 (2D max) and save out to a text file
    """
    with h5py.File(h5_file, "r") as hf:
        dataset = hf[key][:][:,np.array(col_keep_indices)]
        
        # set up dataframe and save out
        dataset_df = pd.DataFrame(dataset, index=hf["example_metadata"][:][:,0], columns=colnames)
        dataset_df.to_csv(text_file, sep='\t')

    return None


