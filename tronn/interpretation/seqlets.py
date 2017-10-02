# Description: contains code for seqlets

import h5py

from tronn.outlayer import H5Handler


def extract_seqlets(thresholded_importances_h5_files, out_h5_file, kmer_len=7, stride=1):
    """Given a thresholded importances file, will extract seqlets out to h5 file
    """
    # make resizable array
    total_task_num = len(thresholded_importances_h5_files)
    min_pos = int(kmer_len / 2.)
    
    with h5py.File(out_h5_file, "r") as hf:
        
        importance_handles = [
            h5py.File(importance_file)
            for importance_file in thresholded_importances_h5_files]

        # get things to skip in creating h5 file
        # and also check seq len
        skip = []
        for key in importance_handles[0].keys():
            if "importance" in key:
                skip.append(key)
                seq_len = importance_handles[0][key].shape[2]

        example_num = importance_handles[0]["feature_metadata"].shape[0]
                
        h5_handler = H5Handler(
            hf,
            importance_handles[0],
            example_num,
            skip=skip)

        # TODO(dk) create dataset for seqlet x task, and seqlet storage
        dataset_shape = (example_num * (1000 - kmer_len), total_task_num)
        h5_handler.h5_handle.create_dataset(
            "seqlet_x_task",
            dataset_shape,
            maxshape=dataset_shape)

        dataset_shape = (example_num * (seq_len - kmer_len), 4, kmer_len)
        h5_handler.h5_handle.create_dataset(
            "seqlets",
            dataset_shape,
            maxshape=dataset_shape)

        # dataset to store kmer id (to easily merge kmers in the end same cluster)
        dataset_shape = (example_num * (seq_len - kmer_len), 1)
        h5_handler.h5_handle.create_dataset(
            "kmer_id",
            dataset_shape,
            maxshape=dataset_shape)
        
        current_idx = 0
        tmp_kmer_array = np.zeros((total_task_num, 4, kmer_len))
        for example_idx in xrange(example_num):
            # go through importances files, get examples from current idx and store in tmp array

            for task_idx in xrange(total_task_num):
                tmp_kmer_array[task_idx,:,:] = importance_handles[task_idx][example_idx,:,:]
            
            # and then go through and extract kmers of length, by stride, and ignore those that are zero
            # or more than half zeros
            for pos_idx in xrange(seq_len - kmer_len):
                seqlet = tmp_kmer_array[:,:,pos_idx:(pos_idx+kmer_len)]
                onehot_kmer = (seqlet > 0).astype(int)

                if np.sum(onehot_kmer) < min_pos:
                    continue

                seqlet_scores = np.sum(seqlet, axis=[1, 2])

                # make a arrays dict to store with h5_handler
                array_dict = {}
                for key in importance_handles[0].keys():
                    # TODO(dk) this is probably not exactly right
                    if not "importance" in key:
                        array_dict[key] = importance_handles[0][key][example_idx,:]

                array_dict["seqlet_x_task"] = seqlet_scores
                array_dict["seqlets"] = np.mean(seqlet, axis=0)

                # TODO(dk) also convert kmer to ID representation and store
                

                hf_handle.store_example(array_dict)
                current_idx += 1
    
        h5_handle.flush()

        # close everything
        for handle in importance_handles:
            handle.close()

    return
