# Description: contains code for seqlets

import h5py
import math

#import phenograph

import numpy as np

from collections import Counter

from tronn.outlayer import H5Handler

from tronn.interpretation.kmers import kmer_array_to_hash
from tronn.interpretation.kmers import kmer_hash_to_string


def extract_seqlets(thresholded_importances_h5_file, tasks, out_h5_file, seq_len=1000, kmer_len=7, stride=1):
    """Given a thresholded importances file, will extract seqlets out to h5 file
    """
    # make resizable array
    min_pos = int(math.ceil(kmer_len / 2.))
    print min_pos
    
    with h5py.File(out_h5_file, "w") as out:
        
        with h5py.File(thresholded_importances_h5_file, 'r') as hf:

            # get importances keys, ordered
            importance_keys = ["importances_task{}".format(task) for task in tasks]

            # get other param
            total_task_num = len(importance_keys)
            
            # get things to skip in creating h5 file
            # and also check seq len
            skip = []
            for key in hf.keys():
                if "importance" in key:
                    skip.append(key)
                    seq_len = hf[key].shape[2]

            example_num = hf["feature_metadata"].shape[0]
            seqlet_max_num = example_num * (seq_len - kmer_len)
            
            h5_handler = H5Handler(
                out,
                hf,
                seqlet_max_num,
                skip=skip,
                is_tensor_input=False)

            # TODO(dk) create dataset for seqlet x task, and seqlet storage
            dataset_shape = (seqlet_max_num, total_task_num)
            h5_handler.add_dataset("seqlet_x_task", dataset_shape, maxshape=dataset_shape)

            dataset_shape = (seqlet_max_num, 4, kmer_len)
            h5_handler.add_dataset("seqlets", dataset_shape, maxshape=dataset_shape)

            # dataset to store kmer id (to easily merge kmers in the end same cluster)
            dataset_shape = (seqlet_max_num, 1)
            h5_handler.add_dataset("kmer_id", dataset_shape, maxshape=dataset_shape)

            current_idx = 0
            tmp_importances_array = np.zeros((total_task_num, 4, seq_len))
            tmp_kmer_array = np.zeros((total_task_num, 4, kmer_len))
            for example_idx in xrange(example_num):
                # go through importances files, get examples from current idx and store in tmp array

                for i in xrange(total_task_num):
                    tmp_importances_array[i,:,:] = hf[importance_keys[i]][example_idx,:,:]
            
                # and then go through and extract kmers of length, by stride, and ignore those that are zero
                # or more than half zeros
                for pos_idx in xrange(seq_len - kmer_len):
                    seqlet = tmp_importances_array[:,:,pos_idx:(pos_idx+kmer_len)]
                    onehot_kmer = (np.sum(seqlet, axis=0) > 0).astype(int)

                    if np.sum(onehot_kmer) < min_pos:
                        continue

                    seqlet_scores = np.sum(seqlet, axis=(1, 2))

                    # make a arrays dict to store with h5_handler
                    array_dict = {}
                    for key in hf.keys():
                        if "feature_metadata" in key:
                            array_dict[key] = hf[key][example_idx,0]
                        elif not "importance" in key:
                            array_dict[key] = hf[key][example_idx,:]

                    array_dict["seqlet_x_task"] = seqlet_scores
                    array_dict["seqlets"] = np.mean(seqlet, axis=0)
                    
                    # also convert kmer to ID representation and store
                    array_dict["kmer_id"] = kmer_array_to_hash(onehot_kmer)

                    h5_handler.store_example(array_dict)
                    current_idx += 1

                    if current_idx % 1000 == 0:
                        print "examples", example_idx
                        print "seqlets", current_idx
    
            h5_handler.flush()
            h5_handler.chomp_datasets()

    return



def reduce_seqlets(seqlet_h5_file, out_h5_file, min_kmer_fract=0.05, top_seqlet_total=100000):
    """Tricks to reduce the seqlet space
    First get kmers with abundance and cluster those

    And ignore low abundance things

    Get a variability score and then keep the top seqlets that are that variable
    """
    # First determine most abundant kmers and cluster those and reduce
    # TODO dont forget to trim
    batch_size = 512
    with h5py.File(seqlet_h5_file, "r") as hf:

        # determine min kmer abundance to chop
        region_counts = Counter(hf["feature_metadata"][:,0])
        total_regions = len(region_counts)
        min_kmer_abundance = min_kmer_fract * total_regions
        print min_kmer_abundance

        # set up kmer hashes
        kmer_hashes = hf["kmer_id"][:,0]
        kmer_hash_counts = Counter(kmer_hashes)
        kmer_hash_to_count = dict(kmer_hash_counts)
        #most_common = kmer_hash_counts.most_common(125)

        # then figure out high variance values
        cv = np.std(hf["seqlet_x_task"], axis=1) / np.mean(hf["seqlet_x_task"], axis=1)
        top_indices = np.argpartition(cv, -top_seqlet_total)[-top_seqlet_total:]
        sorted_top_indices = top_indices[np.argsort(cv[top_indices])]
        cv_cutoff = cv[sorted_top_indices[0]]
        print cv_cutoff

        # and now run through batches and save out into reduced file
        with h5py.File(out_h5_file, 'w') as out:
            h5_handler = H5Handler(
                out,
                hf,
                top_seqlet_total,
                is_tensor_input=False)

            total_batches = int(math.ceil(hf["seqlet_x_task"].shape[0] / float(batch_size)))
            print total_batches
            current_idx = 0
            batch_end = current_idx + batch_size
            
            for batch in xrange(total_batches):
                if batch % 10 == 0:
                    print batch
                
                if batch_end > hf["seqlet_x_task"].shape[0]:
                    batch_end = hf["seqlet_x_task"].shape[0]

                tmp_examples = hf["seqlet_x_task"][current_idx:batch_end,:]

                for example_idx in xrange(current_idx, batch_end):

                    # check conditions
                    if kmer_hash_to_count[hf["kmer_id"][example_idx][0]] < min_kmer_abundance:
                        continue

                    if cv[current_idx] < cv_cutoff:
                        continue

                    # if passes, save out
                    array_dict = {}
                    for key in hf.keys():
                        if "feature_metadata" in key:
                            array_dict[key] = hf[key][example_idx,0]
                        else:
                            array_dict[key] = hf[key][example_idx,:]

                    h5_handler.store_example(array_dict)

                current_idx = batch_end
                batch_end = current_idx + batch_size
        
            h5_handler.flush()
            h5_handler.chomp_datasets()
    
    return


def cluster_seqlets(seqlets_h5_file):
    """Use phenograph to quickly cluster
    """

    with h5py.File(seqlets_h5_file, "r") as hf:

        #communities, graph, Q = phenograph.cluster(hf["seqlets_x_task"])

        import ipdb
        ipdb.set_trace()
        

    return



def reduce_by_kmer():

    return
