"""Contains the run function for weighted kmers
"""

import os
import h5py
import logging

import phenograph

from tronn.util.parallelize import setup_multiprocessing_queue
from tronn.util.parallelize import run_in_parallel


from tronn.interpretation.wkm import kmerize # TODO(dk) change kmerize to use feature key
from tronn.interpretation.wkm import reduce_kmer_mat # TODO(dk) rename as get_topk_kmers
from tronn.interpretation.wkm import make_motif_sets

# TODO(dk) import parallelize functions


def wkm_pipeline(importance_file, feature_key, prefix, kmer_lens=[7]):
    """Pipeline to run wkm to make motifs on one importance file
    
    Args:
      importance_file: hdf5 file where the feature dataset is set up
        as (example, 4, seq_length)
      feature_key: the key string to the feature dataset

    """
    with h5py.File(importance_file, 'r') as hf:
        assert len(hf[feature_key].shape) == 3, "Data file is not formatted properly"
        assert hf[feature_key].shape[1] == 4, "Data file is not formatted properly"

    # kmerize
    wkm_full, wkm_onehot_full = kmerize(importance_file, feature_key, kmer_lens)

    # reduce
    wkm_best = reduce_kmer_mat(wkm_full, kmer_len, cutoff=120) # TODO(dk) clean up this function!!
    wkm_best_transposed = wkm_best.transpose().as_matrix()
    logging.info("Number of kmers kept: {}".format(wkm_best.shape[1]))
    
    # run phenograph here
    wkm_cluster_file = "{}.phenograph.clusters.txt".format(prefix)
    communities, graph, Q = phenograph.cluster(wkm_best_transposed)

    # save results
    sort_indices = np.argsort(communities)
    data_sorted = wkm_best_transposed[sort_indices,:]
    communities_sorted = communities[sort_indices]
    columns_sorted = wkm_reduced.columns[sort_indices]

    out_df = pd.DataFrame(data=data_sorted, index=columns_sorted)
    out_df['community'] = communities_sorted
    out_df.to_csv(wkm_cluster_file, sep='\t')
    
    # make motifs at this point
    # and save out as a PWM file
    make_motif_sets(out_df, onehot_wkm_full, 'task_{}'.format(task_num))

    # TODO(dk) separate out a motif scanner

    return None




def run(args):
    """Pipeline to run wkm to make motifs
    """
    logging.info("Running wkm...")

    wkm_queue = setup_multiprocessing_queue()
    

    for importance_file in args.importance_files:

        prefix = os.path.basename(importance_file).split('.h5')[0]
        
        os.system('mkdir -p {0}/{1}'.format(args.out_dir, prefix))

        feature_key = 'importances_task{}'.format(prefix.split('task')[-1]) # TODO(dk) fix this
        
        fn_args = [importance_file, feature_key,
                   "{0}/{1}/{1}".format(args.out_dir, prefix)]

        wkm_queue.put([wkm_pipeline, fn_args])
        
        # TODO make sure to set up prefix with correct out dir
        
        logging.info("working on {}".format(importance_file))

    run_in_parallel(wkm_queue)

    return None
