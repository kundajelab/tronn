#!/usr/bin/env python3


import sys
import phenograph
import numpy as np
import pandas as pd



def main():

    out_file = sys.argv[2]
    
    # load in data file
    data_file = sys.argv[1]
    data_df = pd.read_table(data_file)
    data = data_df.transpose().as_matrix()

    # run phenograph
    communities, graph, Q = phenograph.cluster(data)

    # and save results out to various matrices
    # ordered matrix to plot with R, and then smaller group files for easy downstream processing
    sort_indices = np.argsort(communities)

    data_sorted = data[sort_indices,:]
    communities_sorted = communities[sort_indices]
    columns_sorted = data_df.columns[sort_indices]
    
    out_df = pd.DataFrame(data=data_sorted, index=columns_sorted)
    out_df['community'] = communities_sorted
    out_df.to_csv(out_file, sep='\t')
    
    
    import pdb
    pdb.set_trace()
    
    return None

main()

