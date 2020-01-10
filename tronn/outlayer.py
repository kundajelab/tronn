"""Description: output layer - allows passing through examples 
or merging regions as desired
"""

import os
import glob
import h5py

import numpy as np
import pandas as pd

class H5Handler(object):

    def __init__(
            self,
            h5_handle,
            tensor_dict,
            sample_size,
            group="",
            batch_size=512,
            chunk_batch_size=32, # note not used currently
            resizable=True,
            is_tensor_input=True,
            skip=[],
            direct_transfer=["label_metadata"],
            saving_single_examples=True, # this should be changed if saving over batch
            compression_opts=4):
        """Keep h5 handle and other relevant storing mechanisms
        """
        self.h5_handle = h5_handle
        self.group = group
        self.example_keys = []
        self.input_batch_size = 1 # preset as 1
        self.saving_single_examples = saving_single_examples
        
        # set up h5 datasets
        for key in tensor_dict.keys():
            h5_key = "{}/{}".format(self.group, key)
            if key in skip:
                continue
            if key in direct_transfer:
                self.h5_handle.create_dataset(key, data=tensor_dict[key])
                continue
            if is_tensor_input:
                dataset_shape = [sample_size] + [int(i) for i in tensor_dict[key].get_shape()[1:]]
            else:
                if saving_single_examples:
                    # batch is not in first dim
                    dataset_shape = [sample_size] + [int(i) for i in tensor_dict[key].shape]
                else:
                    # batch IS in first dim
                    dataset_shape = [sample_size] + [int(i) for i in tensor_dict[key].shape[1:]]
                    self.input_batch_size = int(tensor_dict[key].shape[0]) # save new input batch size
            maxshape = dataset_shape if resizable else None
            
            if "example_metadata" in key:
                self.h5_handle.create_dataset(
                    h5_key, dataset_shape, maxshape=maxshape, dtype="S100",
                    compression="gzip", compression_opts=compression_opts, shuffle=True)
            elif "string" in key:
                self.h5_handle.create_dataset(
                    h5_key, dataset_shape, maxshape=maxshape, dtype="S1000",
                    compression="gzip", compression_opts=compression_opts, shuffle=True)
            else:
                self.h5_handle.create_dataset(
                    h5_key, dataset_shape, maxshape=maxshape,
                    compression="gzip", compression_opts=compression_opts, shuffle=True)
            self.example_keys.append(key)
        
        # batch size must be a multiple of the input batch size
        assert batch_size % self.input_batch_size == 0, "batch size: {}; input batch size: {}, from {}".format(batch_size, self.input_batch_size, key) 

        # other needed args            
        self.resizable = resizable
        self.batch_size = batch_size
        self.batch_start = 0
        self.batch_end = self.batch_start + batch_size
        self.setup_tmp_arrays()

        
    def setup_tmp_arrays(self):
        """Setup numpy arrays as tmp storage before batch storage into h5
        """
        tmp_arrays = {}
        for key in self.example_keys:
            tmp_arrays[key] = [] # just lists, stack at end
        self.tmp_arrays = tmp_arrays
        self.tmp_arrays_idx = 0

        return

    
    def add_dataset(self, key, shape, maxshape=None):
        """Add dataset and update numpy array
        """
        h5_key = "{}/{}".format(self.group, key)
        self.h5_handle.create_dataset(h5_key, shape, maxshape=maxshape)
        self.example_keys.append(key)
        
        tmp_shape = [self.batch_size] + [int(i) for i in shape[1:]]
        self.tmp_arrays[key] = np.zeros(tmp_shape)
        
        return

    
    def store_example(self, example_arrays):
        """Store an example into the tmp numpy arrays, push batch out if done with batch
        """
        # i think might just need to change here to push batch instead of indiv
        tmp_i_start = self.tmp_arrays_idx
        tmp_i_stop = self.tmp_arrays_idx + self.input_batch_size
        
        for key in self.example_keys:
            self.tmp_arrays[key].append(example_arrays[key])
        self.tmp_arrays_idx += self.input_batch_size
        
        # now if at end of batch, push out and reset tmp
        if self.tmp_arrays_idx == self.batch_size:
            self.push_batch()

        return

    
    def push_batch(self):
        """Go from the tmp array to the h5 file
        """
        for key in self.example_keys:
            h5_key = "{}/{}".format(self.group, key)
            # stack or concatenate
            if self.saving_single_examples:
                tmp_array = np.stack(self.tmp_arrays[key], axis=0)
            else:
                tmp_array = np.concatenate(self.tmp_arrays[key], axis=0)
            # adjust dtype as needed
            if key == "example_metadata":
                tmp_array = tmp_array.astype("S100")
            elif "string" in key:
                tmp_array = tmp_array.astype("S1000")
            # and write in
            self.h5_handle[h5_key][self.batch_start:self.batch_end] = tmp_array
                
        # set new point in batch
        self.batch_start = self.batch_end
        self.batch_end += self.batch_size
        self.setup_tmp_arrays()
        self.tmp_arrays_idx = 0
        
        return


    def flush(self, defined_batch_end=None):
        """Check to see how many are real examples and push the last batch gracefully in
        """
        # determine actual batch end
        test_key = "example_metadata"
        batch_end = len(self.tmp_arrays[test_key]) * self.input_batch_size
        
        # in this set up, easy to just use push batch again
        self.push_batch()
        
        # adjust batch end
        self.batch_end = self.batch_end - self.batch_size + batch_end
        
        return

    
    def chomp_datasets(self):
        """Once done adding things, if can resize then resize datasets
        """
        assert self.resizable == True

        for key in self.example_keys:
            h5_key = "{}/{}".format(self.group, key)
            dataset_final_shape = [self.batch_end] + [int(i) for i in self.h5_handle[h5_key].shape[1:]]
            self.h5_handle[h5_key].resize(dataset_final_shape)
            
        return


    
def strided_merge_generator(array, metadata, stride=50, bin_size=200):
    """given an array with stride, aggregate appropriately
    assume contiguous?
    """
    # check length - assumes (N, seq_len, ...)
    seq_len = array.shape[1]

    # adjust metadata here
    active = pd.DataFrame(data=metadata)[0].str.split(";").str[1]
    region = pd.DataFrame(data=metadata)[0].str.split(";").str[0]
    active = active.str.split("=").str[1]
    chrom = active.str.split(":").str[0].values
    coords = active.str.split(":").str[1]
    pos_start = coords.str.split("-").str[0].values.astype(int)
    pos_stop = coords.str.split("-").str[1].values.astype(int)
    
    # clip off edges that don't fit striding well
    clip_len = (seq_len / 2) % stride
    clip_start = clip_len
    clip_end = seq_len - clip_len
    array = array[:,clip_start:clip_end] # (N, seq_len, ...)

    # also clip metadata, after checking new array shape
    clip_len = bin_size/2 - array.shape[1]/2
    pos_start += clip_len
    pos_stop -= clip_len
    
    # figure out how to reshape
    array_shape = list(array.shape) # (N, seqlen, ...)
    array_shape.insert(1, -1) # (N, -1, seqlen, ...)
    array_shape[2] = stride # (N, -1, stride, ...)
    array = np.reshape(array, array_shape) # (N, num_strides, stride, ...)

    # figure out how many strides will be valid
    #num_valid = array.shape[0] - 2*(array.shape[1] -1)
    #new_shape = [num_valid*stride] + list(array.shape[3:])
    
    # now merge and save out
    start_idx = array.shape[1] - 1
    end_idx = array.shape[0] - array.shape[1] + 1
    for example_idx in range(start_idx, end_idx):

        # metadata
        example_id = region[example_idx]
        example_chrom = np.repeat(chrom[example_idx], stride)
        example_pos_start = pos_stop[example_idx] - stride +  np.arange(stride)
        example_pos_stop = example_pos_start + 1
        example_metadata = pd.DataFrame({
            "chrom": example_chrom,
            "start": example_pos_start,
            "stop": example_pos_stop})
        
        # get strided sum
        total_overlap = 0
        for stride_idx in range(array.shape[1]):
            slice_example_idx = example_idx + stride_idx # <- this is what to check
            if region[slice_example_idx] != example_id:
                continue
            slice_stride_idx = array.shape[1] - stride_idx -1
            array_slice = array[slice_example_idx, slice_stride_idx]
            if stride_idx == 0:
                current_sum = array_slice
            else:
                current_sum += array_slice
            total_overlap += 1

        # get mean
        current_mean = current_sum / float(total_overlap)
        
        yield current_mean, example_metadata


def strided_merge_to_bedgraph(data, metadata, out_prefix, stride=50):
    """write out strided merge to bedgraph files, separating task predictions
    """
    # get tasks
    num_tasks = data.shape[2]

    # make files for each, with file pointers?
    generator = strided_merge_generator(data, metadata, stride=stride)
    for data_merged, metadata_matched in generator:
        for task_idx in range(num_tasks):
            
            # merge in data and metadata
            results = metadata_matched.copy()
            results["val"] = data_merged[:,task_idx]
            
            # filter out zeros
            results = results[results["val"] != 0]

            # save out info if anything remaining
            if results.shape[0] != 0:
                out_file = "{}.taskidx-{}.bedgraph".format(out_prefix, task_idx)
                with open(out_file, "a") as fp:
                    results.to_csv(fp, sep="\t", header=False, index=False)
                    
    return None


def h5_to_bigwig(
        h5_file,
        out_prefix,
        chromsizes,
        features_key="sequence-weighted.active",
        metadata_key="example_metadata"):
    """
    """
    # pull data
    with h5py.File(h5_file, "r") as hf:
        num_tasks = hf[features_key].shape[2]
        metadata = hf[metadata_key][:,0] # (N)
        
    # clean up first
    for task_idx in range(num_tasks):
        out_file = "{}.taskidx-{}.bedgraph".format(out_prefix, task_idx)
        if os.path.isfile(out_file):
            os.system("rm {}".format(out_file))

    # batching
    start_idx = 0
    end_idx = start_idx + 1
    current_id = metadata[start_idx].split(";")[0]
    num_examples = metadata.shape[0]
    while end_idx < num_examples:

        # check if same ID or not
        if (metadata[end_idx].split(";")[0] == current_id) and (end_idx < num_examples-1):
            end_idx += 1
            continue
        
        # get batch
        if end_idx < num_examples:
            with h5py.File(h5_file, "r") as hf:
                data_batch = hf[features_key][start_idx:end_idx] # (N, task, seqlen, 4)
            metadata_batch = metadata[start_idx:end_idx]
        else:
            with h5py.File(h5_file, "r") as hf:
                data_batch = hf[features_key][start_idx:num_examples] # (N, task, seqlen, 4)
            metadata_batch = metadata[start_idx:num_examples]

        # other adjust
        data_batch = np.sum(data_batch, axis=-1) # (N, task, seqlen)
        data_batch = np.swapaxes(data_batch, -1, -2) # (N, seqlen, task)
            
        # strided merge
        strided_merge_to_bedgraph(data_batch, metadata_batch, out_prefix, stride=50)
        
        # and get new
        start_idx = end_idx
        end_idx = start_idx + 1
        current_id = metadata[start_idx].split(";")[0]

    # and then convert to bigwig
    bedgraph_files = sorted(glob.glob("{}*bedgraph".format(out_prefix)))
    for bedgraph_file in bedgraph_files:
        bigwig_file = "{}.bigwig".format(bedgraph_file.split(".bedgraph")[0])
        bedgraph_to_bigwig_cmd = "bedGraphToBigWig {} {} {}".format(
            bedgraph_file,
            chromsizes,
            bigwig_file)
        os.system(bedgraph_to_bigwig_cmd)
    
    return None
