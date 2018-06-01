"""Description: output layer - allows passing through examples 
or merging regions as desired
"""

import numpy as np

import tensorflow as tf


def split_region_old(region):
    """Split region into a list of [chr, start, stop]
    """
    region = region.split(";")[-1].split("=")[1] # to account for new region names
    chrom = region.split(":")[0]
    start = int(region.split(":")[1].split("-")[0])
    stop = int(region.split(":")[1].split("-")[1].split("(")[0].rstrip('\0'))
    return [chrom, start, stop]


class RegionObjectOLD(object):

    def __init__(self, shape, merge_type):
        """Basically a region object must be some form of numpy array
        and adding to it
        """
        self.array = np.zeros(shape)
        self.merge_type = merge_type

        
    def merge(self, new_array, offset=0):
        """merge in array
        """
        # merge by offset
        if self.merge_type == "offset":
            
            basepair_dimension_idx = 1
            # note that sequence len should be dim 1, {1, seq_len, 4}
            # deal with overhangs and clip
            left_clip_idx = abs(min(0, offset))
            right_clip_idx = new_array.shape[basepair_dimension_idx] - abs(
                min(0, self.array.shape[basepair_dimension_idx] - (new_array.shape[basepair_dimension_idx] + offset)))
            new_array_clipped = new_array[:, left_clip_idx:right_clip_idx,:]

            if offset < 0:
                adjusted_offset = 0
            else:
                adjusted_offset = offset

            left_zero_padding = np.zeros(
                (new_array.shape[0], adjusted_offset, 4))
            right_zero_padding = np.zeros(
                (new_array.shape[0],
                 self.array.shape[basepair_dimension_idx] - adjusted_offset - new_array_clipped.shape[basepair_dimension_idx],
                 4))

            self.array += np.concatenate(
                (left_zero_padding, new_array_clipped, right_zero_padding),
                axis=basepair_dimension_idx)
        # merge by sum
        elif self.merge_type == "sum":
            self.array += new_array
        # merge by max
        elif self.merge_type == "max":
            self.array = np.maximum(self.array, new_array)

        
    def get_array(self):
        """return array
        """
        return self.array

    def get_merge_type(self):
        """return the merge type
        """
        return self.merge_type


class RegionTrackerOLD(object):
    """Builds a flexible region tracker that updates a dictionary
    """

    def __init__(self, region, region_data):

        self.region = region
        self.chrom, self.start, self.stop = split_region(region)
        self.region_data = {}
        
        for key in region_data.keys():
            if region_data[key][1] == "offset":
                self.region_data[key] = RegionObject(
                    (region_data[key][0].shape[0], self.stop - self.start, 4), "offset")
            else:
                self.region_data[key] = RegionObject(
                    region_data[key][0].shape,
                    region_data[key][1])

                
    def is_same_region(self, region):
        return self.region == region

    
    def merge(self, region, region_data):
        chrom, start, stop = split_region(region)
        offset = start - self.start
        for key in self.region_data.keys():
            self.region_data[key].merge(
                region_data[key][0], offset=offset)

            
    def get_region(self, offset_fixed_width=None):
        """Return a region element
        """
        region_data = {}
        for key in self.region_data.keys():
            if self.region_data[key].get_merge_type() == "offset" and offset_fixed_width is not None:
                # if desire fixed length output, pad sequence
                variable_size_array = self.region_data[key].get_array()
                variable_size_array_len = variable_size_array.shape[1]
                if variable_size_array_len < offset_fixed_width:
                    zero_array = np.zeros((4, offset_fixed_width - variable_size_array_len))
                    padded_array = np.concatenate((variable_size_array, zero_array), axis=1)
                else:
                    trim_len = (variable_size_array_len - offset_fixed_width) / 2.
                    padded_array = variable_size_array[:, trim_len:offset_fixed_width+trim_len]
                region_data[key] = padded_array
            else:
                region_data[key] = self.region_data[key].get_array()
        return self.region, region_data


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
        #del self.h5_handle[group]
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
            elif "features.string" in key:
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
                tmp_arrays[key].fill("false=chrY:0-0")
            elif "features.string" in key:
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
            try:
                self.tmp_arrays[key][self.tmp_arrays_idx] = example_arrays[key]
            except:
                import ipdb
                ipdb.set_trace()
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
            try:
                self.h5_handle[h5_key][self.batch_start:self.batch_end] = self.tmp_arrays[key]
            except:
                import ipdb
                ipdb.set_trace()
            
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
                if self.tmp_arrays["example_metadata"][batch_end][0].rstrip("\0") == "false=chrY:0-0":
                    break
        self.batch_end = self.batch_start + batch_end
        
        for key in self.example_keys:
            h5_key = "{}/{}".format(self.group, key)
            try:
                self.h5_handle[h5_key][self.batch_start:self.batch_end] = self.tmp_arrays[key][0:batch_end]
            except:
                import ipdb
                ipdb.set_trace()

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




class OutLayer(object):
    """Handles the outputs of the graph gracefully"""
    
    def __init__(
            self,
            sess,
            graph_tensor_outputs,
            #batch_size,
            reconstruct_regions=False,
            ignore_outputs=[]):
        self.sess = sess
        self.graph_tensor_outputs = graph_tensor_outputs
        self.reconstruct_regions = reconstruct_regions
        self.ignore_outputs = ignore_outputs

        # remove the ignore outputs
        for key in ignore_outputs:
            if key in self.graph_tensor_outputs.keys():
                del self.graph_tensor_outputs[key]
        
        # run first batch
        self._run_sess()

        
    def _run_sess(self):
        # run sess
        self.outputs = self.sess.run(self.graph_tensor_outputs)
        
        # set up batch size
        test_key = self.outputs.keys()[0]
        self.batch_size = self.outputs[test_key].shape[0]
        self.batch_idx = 0

        
    def __iter__(self):
        return self

    
    def next(self, batch_size=1):
        """get the next example out
        """
        out_arrays = {}
        # consider putting coord should stop in here?
        if self.batch_idx >= self.batch_size:
            try:
                self._run_sess()
            except tf.errors.OutOfRangeError:
                raise StopIteration
        # collect batch outputs and return
        for key in self.graph_tensor_outputs.keys():
            # check if string
            result = self.outputs[key][self.batch_idx]
            if isinstance(result[0], basestring):
                out_arrays[key] = result[0]
            else:
                out_arrays[key] = result
        self.batch_idx += 1

        return out_arrays
