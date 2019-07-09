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
            resizable=True,
            is_tensor_input=True,
            skip=[],
            direct_transfer=["label_metadata"]):
        """Keep h5 handle and other relevant storing mechanisms
        """
        self.h5_handle = h5_handle
        self.tensor_dict = tensor_dict
        self.sample_size = sample_size
        self.group = group
        self.is_tensor_input = is_tensor_input
        self.skip = skip
        self.direct_transfer = direct_transfer
        self.example_keys = []
        for key in tensor_dict.keys():
            h5_key = "{}/{}".format(self.group, key)
            if key in self.skip:
                continue
            if key in self.direct_transfer:
                self.h5_handle.create_dataset(key, data=tensor_dict[key])
                continue
            if is_tensor_input:
                dataset_shape = [sample_size] + [int(i) for i in tensor_dict[key].get_shape()[1:]]
            else:
                dataset_shape = [sample_size] + [int(i) for i in tensor_dict[key].shape]
            maxshape = dataset_shape if resizable else None
            if "example_metadata" in key:
                self.h5_handle.create_dataset(h5_key, dataset_shape, maxshape=maxshape, dtype="S100")
            elif "string" in key:
                self.h5_handle.create_dataset(h5_key, dataset_shape, maxshape=maxshape, dtype="S1000")
            else:
                self.h5_handle.create_dataset(h5_key, dataset_shape, maxshape=maxshape)
            self.example_keys.append(key)
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
            h5_key = "{}/{}".format(self.group, key)
            
            dataset_shape = [self.batch_size] + [int(i) for i in self.h5_handle[h5_key].shape[1:]]

            if key == "example_metadata":
                tmp_arrays[key] = np.empty(dataset_shape, dtype="S100")
                tmp_arrays[key].fill("features=chr1:0-1000")
            elif "string" in key:
                tmp_arrays[key] = np.empty(dataset_shape, dtype="S1000")
                tmp_arrays[key].fill("NNNN")
            else:
                tmp_arrays[key] = np.zeros(dataset_shape)
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
        for key in self.example_keys:
            #try:
            self.tmp_arrays[key][self.tmp_arrays_idx] = example_arrays[key]
            #except:
            #    import ipdb
            #    ipdb.set_trace()
        self.tmp_arrays_idx += 1
        
        # now if at end of batch, push out and reset tmp
        if self.tmp_arrays_idx == self.batch_size:
            self.push_batch()

        return

    
    def store_batch(self, batch):
        """Coming from batch input
        """
        self.tmp_arrays = batch
        self.push_batch()
        
        return

    
    def push_batch(self):
        """Go from the tmp array to the h5 file
        """
        for key in self.example_keys:
            h5_key = "{}/{}".format(self.group, key)
            #try:
            self.h5_handle[h5_key][self.batch_start:self.batch_end] = self.tmp_arrays[key]
            #except:
            #    import ipdb
            #    ipdb.set_trace()
            
        # set new point in batch
        self.batch_start = self.batch_end
        self.batch_end += self.batch_size
        self.setup_tmp_arrays()
        self.tmp_arrays_idx = 0
        
        return


    def flush(self, defined_batch_end=None):
        """Check to see how many are real examples and push the last batch gracefully in
        """
        if defined_batch_end is not None:
            batch_end = defined_batch_end
        else:
            for batch_end in xrange(self.tmp_arrays["example_metadata"].shape[0]):
                if self.tmp_arrays["example_metadata"][batch_end][0].rstrip("\0") == "features=chr1:0-1000":
                    break
        self.batch_end = self.batch_start + batch_end

        # check if smaller than batch size
        #test_key = self.example_keys[0]
        #if self.h5_handle[test_key][self.batch_start:self.batch_end].shape[0] < batch_end:
        #    batch_end = self.h5_handle[test_key][self.batch_start:self.batch_end].shape[0]
        #    self.batch_end = self.batch_start + batch_end # TODO something up with this
        
        # save out
        for key in self.example_keys:
            h5_key = "{}/{}".format(self.group, key)
            #try:
            self.h5_handle[h5_key][self.batch_start:self.batch_end] = self.tmp_arrays[key][0:batch_end]
            #except:
            #    import ipdb
            #    ipdb.set_trace()

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
    start_idx = array.shape[1]
    end_idx = array.shape[0] - array.shape[1] + 1
    for example_idx in range(start_idx, end_idx):

        # metadata
        example_id = region[example_idx] # TODO this is wrong - need original feature
        example_chrom = np.repeat(chrom[example_idx], stride)
        example_pos_start = pos_stop[example_idx] - stride +  np.arange(stride)
        example_pos_stop = example_pos_start + 1
        example_metadata = pd.DataFrame({
            "chrom": example_chrom,
            "start": example_pos_start,
            "stop": example_pos_stop})
        
        # get strided sum
        # TODO for each one, need to check if actually contigous
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
            
        yield current_sum, example_metadata


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
    with h5py.File(h5_file, "r") as hf:
        data = hf[features_key][:] # (N, task, seqlen, 4)
        data = np.sum(data, axis=-1) # (N, task, seqlen)
        data = np.swapaxes(data, -1, -2) # (N, seqlen, task)
        metadata = hf[metadata_key][:,0] # (N)
        
    # clean up first
    num_tasks = data.shape[2]
    for task_idx in range(num_tasks):
        out_file = "{}.taskidx-{}.bedgraph".format(out_prefix, task_idx)
        if os.path.isfile(out_file):
            os.system("rm {}".format(out_file))
        
    # TO CONSIDER: also do the above in BATCHES. to reduce how much is being loaded into memory at one time.
    strided_merge_to_bedgraph(data, metadata, out_prefix, stride=50)

    # and then convert to bigwig
    bedgraph_files = sorted(glob.glob("{}*bedgraph".format(out_prefix)))
    print bedgraph_files
    for bedgraph_file in bedgraph_files:
        bigwig_file = "{}.bigwig".format(bedgraph_file.split(".bedgraph")[0])
        bedgraph_to_bigwig_cmd = "bedGraphToBigWig {} {} {}".format(
            bedgraph_file,
            chromsizes,
            bigwig_file)
        os.system(bedgraph_to_bigwig_cmd)
    
    return None
