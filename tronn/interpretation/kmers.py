"""Kmer utils
"""

import os
import h5py
import math
import glob
import numpy as np

import tensorflow as tf

from tronn.util.parallelize import setup_multiprocessing_queue
from tronn.util.parallelize import run_in_parallel

from tronn.util.tf_utils import setup_tensorflow_session
from tronn.util.tf_utils import close_tensorflow_session

from tronn.outlayer import H5Handler
from tronn.outlayer import H5InputHandler

def kmer_array_to_hash(kmer_array, num_bases=5):
    """Get unique identifying position of kmer
    """
    kmer_idx = 0
    for i in range(kmer_array.shape[1]):

        if kmer_array[0,i] > 0:
            num = 1
        elif kmer_array[1,i] > 0:
            num = 2
        elif kmer_array[2,i] > 0:
            num = 3
        elif kmer_array[3,i] > 0:
            num = 4
        else:
            num = 0
            
        kmer_idx += num * (num_bases**i)
        
    return kmer_idx


def kmer_hash_to_string(idx, kmer_len=7, num_bases=5):
    """From unique index val, get kmer string back
    """
    num_to_base = {0: "N", 1:"A", 2:"C", 3:"G", 4:"T"}

    idx_tmp = idx
    reverse_kmer_string = []
    for pos in reversed(range(kmer_len)):
        num = int(idx_tmp / num_bases**pos)
        try:
            reverse_kmer_string.append(num_to_base[num])
        except:
            import pdb # to debug
            pdb.set_trace()
        idx_tmp -= num * (num_bases**pos)
    kmer_string = reversed(reverse_kmer_string)
    
    return ''.join(kmer_string)


def kmer_hash_to_array(idx, kmer_len=7, num_bases=5):
    """From unique index val, get kmer string back
    """
    num_to_base = {0: "N", 1:"A", 2:"C", 3:"G", 4:"T"}

    kmer_array = np.zeros((num_bases-1, kmer_len))
    
    idx_tmp = idx
    for pos in reversed(range(kmer_len)):
        num = int(idx_tmp / num_bases**pos)
        if num != 0:
            kmer_array[num-1,pos] = 1
        idx_tmp -= num * (num_bases**pos)
    
    return kmer_array


def kmer_array_to_string(kmer_array, num_bases=5):
    """Get unique identifying position of kmer
    """
    kmer_idx = 0
    base_list = []
    # TODO check reversal
    for i in range(kmer_array.shape[1]):

        if np.sum(kmer_array[:,i]) == 0:
            bp = 'N'
        else:
            idx = np.argmax(kmer_array[:,i])
        
            if idx == 0:
                bp = 'A'
            elif idx == 1:
                bp = 'C'
            elif idx == 2:
                bp = 'G'
            elif idx == 3:
                bp = 'T'

        base_list.append(bp)
        
    return ''.join(base_list)


def kmerize_gpu(tronn_graph, out_h5_file, total_examples, batch_size=128):
    """Utilize tensorflow to featurize as gapped kmers
    """
    with tf.Graph().as_default() as g:

        # build graph
        # need to keep: example_metadata, features, labels
        kmer_features = tronn_graph.build_graph()

        
        tensor_dict = {
            "example_metadata": tronn_graph.metadata,
            "features": kmer_features,
            "labels": tronn_graph.labels
        }
        
        # setup session
        sess, coord, threads = setup_tensorflow_session()

        # setup output file
        with h5py.File(out_h5_file, "w") as out:

            h5_handler = H5Handler(out, tensor_dict, total_examples, resizable=False, batch_size=batch_size)

            batch_num = int(math.ceil(total_examples / float(batch_size)))
            print batch_num
            
            for batch_idx in xrange(batch_num):

                if batch_idx % 10 == 0:
                    print batch_idx

                # run session
                outputs = sess.run(tensor_dict)
            
                # store batch
                h5_handler.store_batch(outputs)

                # push
                if batch_idx < (batch_num-1):
                    h5_handler.push_batch()
                else:
                    h5_handler.flush(
                        defined_batch_end=total_examples)
                    
        close_tensorflow_session(coord, threads)
        
    return


def kmerize(
        sequence_h5_file,
        out_h5_file,
        kmer_len=6,
        num_bases=5):
    """Kmerize sequence file
    """
    with h5py.File(sequence_h5_file, "r") as hf:

        total_examples = hf["example_metadata"].shape[0]
        print "total:", total_examples
        total_columns = num_bases**kmer_len
        # set up h5 input handler
        in_h5_handler = H5InputHandler(hf, flatten=True)
        
        with h5py.File(out_h5_file, "w") as out:
            # set up h5_handler
            out_h5_handler = H5Handler(
                out, hf, total_examples, resizable=False, is_tensor_input=False, skip=["features"])
        
            out_h5_handler.add_dataset("features", (total_examples, total_columns))

            # and now get batches and kmerize and store
            for example_idx in xrange(total_examples):

                if example_idx % 1000 == 0:
                    print example_idx

                example_array = in_h5_handler.get_example_array()

                # kmerize
                #sequence = np.squeeze(example_array["features"]).transpose(1, 0)
                sequence = example_array["features"]
                kmer_counts = np.zeros((1, total_columns))
                for i in range(sequence.shape[1] - kmer_len):
                    kmer = sequence[:,i:(i+kmer_len)]
                    kmer_idx = kmer_array_to_hash(kmer)
                    kmer_counts[0, kmer_idx] += 1
                # don't keep counts of NNNN kmers
                kmer_counts[0, 0] = 0
                example_array["features"] = kmer_counts
                
                # and store
                out_h5_handler.store_example(example_array)

    return



def kmerize2(
        h5_file,
        kmer_lens=[6, 8, 10],
        num_bases=5,
        is_importances=False,
        task_num=0,
        save=False,
        save_file='tmp.h5'):
    """Convert weighted sequence into weighted kmers

    Args:
      importances_h5: file with importances h5 file

    Returns:
      matrix of kmer scores
    """

    if is_importances:
        feature_key = 'importances_task{}'.format(task_num)
    else:
        feature_key = 'features'
    
    # determine total cols of matrix (number of kmer features)
    total_cols = 0
    for kmer_len in kmer_lens:
        total_cols += num_bases**kmer_len
    print 'total kmer features:', total_cols
        
    # determine number of sequences and generate matrix
    with h5py.File(h5_file, 'r') as hf:
        if is_importances:
            num_examples = np.sum(hf['labels'][:,task_num] > 0)
            example_indices = np.where(hf['labels'][:,task_num] > 0)
        else:
            num_examples = hf['regions'].shape[0]
            example_indices = [np.array(range(num_examples))]
        num_tasks = hf['labels'].shape[1]
    print 'total examples:', num_examples
            

    if is_importances:
        wkm_mat = np.zeros((num_examples, total_cols))
        onehot_wkm_mat = np.zeros((num_examples, total_cols, 4, max(kmer_lens)))

    with h5py.File(save_file, 'w') as out:

        # set up datasets
        importance_mat = out.create_dataset(feature_key,
                                            [num_examples, total_cols])
        labels_mat = out.create_dataset('labels',
                                        [num_examples, num_tasks])
        regions_mat = out.create_dataset('regions',
                                         [num_examples, 1],
                                         dtype='S100')
        
        
        # now start reading into file
        with h5py.File(h5_file, 'r') as hf:
            
            # for each kmer, figure out start position
            for kmer_len_idx in range(len(kmer_lens)):
                print 'kmer len', kmer_lens[kmer_len_idx]
                kmer_len = kmer_lens[kmer_len_idx]
                min_pos = kmer_len - 2
                if kmer_len_idx == 0:
                    start_idx = 0
                else:
                    start_idx = kmer_len_idx * (num_bases**kmer_lens[kmer_len_idx-1])

                # run through all the examples
                current_idx = 0
                for example_idx in example_indices[0]:
                    
                    if current_idx % 500 == 0:
                        print current_idx

                    # go through the sequence

                    if not is_importances:
                        wkm_vec = np.zeros((total_cols))
                        sequence = np.squeeze(hf[feature_key][example_idx,:,:,:]).transpose(1,0)
                    else:
                        sequence = hf[feature_key][example_idx,:,:]
                    
                    for i in range(sequence.shape[1] - kmer_len):
                        weighted_kmer = sequence[:,i:(i+kmer_len)]
                        kmer = (weighted_kmer > 0).astype(int)
                        
                        if np.sum(kmer) < min_pos:
                            continue

                        kmer_idx = kmer_to_idx(kmer)

                        if is_importances:
                            wkm_score = np.sum(weighted_kmer) # bp adjusted score?
                            wkm_mat[current_idx, start_idx+kmer_idx] += wkm_score
                            onehot_wkm_mat[current_idx, start_idx+kmer_idx,:,0:kmer_len] += weighted_kmer
                        else:
                            wkm_vec[start_idx+kmer_idx] += 1

                    # and then save if necessary
                    if save:
                        importance_mat[current_idx,:] = wkm_vec
                        labels_mat[current_idx,:] = hf['labels'][example_idx,:]
                        regions_mat[current_idx] = hf['regions'][example_idx]
                    
                    current_idx += 1

    if not save:
        os.system('rm {}'.format(save_file))
                    
    if is_importances:
        onehot_wkm_avg = np.mean(onehot_wkm_mat, axis=0)
        return wkm_mat, onehot_wkm_avg
    else:
        return None

    
def kmerize_parallel(in_dir, out_dir, parallel=24):
    """Utilize func_worker to run on multiple files
    """
    
    kmerize_queue = setup_multiprocessing_queue()

    # find h5 files
    h5_files = glob.glob('{}/*.h5'.format(in_dir))
    print "Found {} h5 files".format(len(h5_files))

    for h5_file in h5_files:
        out_file = '{0}/{1}.kmers.h5'.format(
            out_dir,
            os.path.basename(h5_file).split('.h5')[0])
        kmerize_args = [h5_file, out_file]

        if not os.path.isfile(out_file):
            kmerize_queue.put([kmerize, kmerize_args])

    run_in_parallel(kmerize_queue, parallel=parallel, wait=True)

    return None

