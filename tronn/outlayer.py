"""Description: output layer - allows passing through examples 
or merging regions as desired
"""

import numpy as np


def split_region(region):
    """Split region into a list of [chr, start, stop]
    """
    chrom = region.split(":")[0]
    start = int(region.split(":")[1].split("-")[0])
    stop = int(region.split(":")[1].split("-")[1].split("(")[0].rstrip('\0'))
    return [chrom, start, stop]


class RegionObject(object):

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
            # note that sequence len should be dim 1
            # deal with overhangs and clip
            left_clip_idx = abs(min(0, offset))
            right_clip_idx = new_array.shape[1] - abs(
                min(0, self.array.shape[1] - (new_array.shape[1] + offset)))
            new_array_clipped = new_array[:,left_clip_idx:right_clip_idx]

            if offset < 0:
                adjusted_offset = 0
            else:
                adjusted_offset = offset
                
            left_zero_padding = np.zeros(
                (4, adjusted_offset))
            right_zero_padding = np.zeros(
                (4, self.array.shape[1] - adjusted_offset - new_array_clipped.shape[1]))
            self.array += np.concatenate(
                (left_zero_padding, new_array_clipped, right_zero_padding),
                axis=1)
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


class RegionTracker(object):
    """Builds a flexible region tracker that updates a dictionary
    """

    def __init__(self, region, region_data):

        self.region = region
        self.chrom, self.start, self.stop = split_region(region)
        self.region_data = {}
        
        for key in region_data.keys():
            if region_data[key][1] == "offset":
                self.region_data[key] = RegionObject(
                    (4, self.stop - self.start), "offset")
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


class ExampleGenerator(object):

    def __init__(
            self,
            sess,
            tensor_dict,
            batch_size,
            reconstruct_regions=False,
            keep_negatives=True,
            filter_by_prediction=False,
            filter_tasks=[]):
        """Set up params for example generation
        """
        self.sess = sess
        self.tensor_dict = tensor_dict
        self.batch_size = batch_size
        self.batch_pointer = 0
        self.valid_examples = 0
        self.reconstruct_regions = reconstruct_regions
        self.keep_negatives = keep_negatives

        if self.reconstruct_regions: #?? whats going on here
            self.name_idx = -1
        else:
            self.name_idx = 0

        self.filter_by_prediction = filter_by_prediction
        self.filter_tasks = np.array(filter_tasks)
        
        filter_tasks_mask_tmp = np.zeros((1, int(tensor_dict["labels"].get_shape()[1])))
        filter_tasks_mask_tmp[0, filter_tasks] = 1
        self.filter_tasks_mask = filter_tasks_mask_tmp


            
        # initialize region tracker with first region in batch
        self.batch_region_arrays = self.sess.run(self.tensor_dict)
        region, region_arrays = self.get_filtered_example()
        self.region_tracker = RegionTracker(region, region_arrays)
        self.region_tracker.merge(region, region_arrays)

        
    def build_example_dict_arrays(self):
        """Build dictionary of arrays for 1 example
        """
        # if necessary, get a new batch
        if self.batch_pointer == self.batch_size:
            self.batch_pointer = 0
            self.batch_region_arrays = self.sess.run(self.tensor_dict)

        # extract data and construct into region array dict
        region_arrays = {}
        for key in self.batch_region_arrays.keys():
            if key == "feature_metadata":
                region = self.batch_region_arrays[key][self.batch_pointer,0]
                region_name = region.split(";")[self.name_idx].split("=")[1].split("::")[0]
            elif "importance" in key:
                region_arrays[key] = (
                    self.batch_region_arrays[key][self.batch_pointer,:,:],
                    "offset")
                    #np.squeeze(
                    #    self.batch_region_arrays[key][self.batch_pointer,:,:,:]).transpose(1,0),
                    #"offset")
            else:
                region_arrays[key] = (
                    self.batch_region_arrays[key][self.batch_pointer,:],
                    "max")
        self.batch_pointer += 1

        return region_name, region_arrays

    
    def get_filtered_example(self, accuracy_cutoff=0.4):
        """Get the built dictionary and filter
        """
        while True:

            region_name, region_arrays = self.build_example_dict_arrays()

            # check negatives
            if not self.keep_negatives and region_arrays["negative"][0] == 1.:
                continue

            # check correctly predicted
            # compare label vector to probs vector (NOT XOR gate) and then mask by tasks that we care about.
            if self.filter_by_prediction and region_arrays["subset_accuracy"][0] < accuracy_cutoff:
                continue

            # if all conditions met, break and output filtered example
            break

        return region_name, region_arrays
    
                
    def run(self):

        if self.reconstruct_regions:
            # go until you can yield an example
            while True:
                
                # go through an example
                region, region_arrays = self.get_filtered_example()

                # merge if same
                if self.region_tracker.is_same_region(region):
                    self.region_tracker.merge(region, region_arrays)
                else:
                    out_region, out_region_arrays = self.region_tracker.get_region()
                    # and reset with new info
                    self.region_tracker = RegionTracker(region, region_arrays)
                    self.region_tracker.merge(region, region_arrays)
                    break
            self.valid_examples += 1
            if self.valid_examples % 100 == 0:
                print self.valid_examples
            
            return out_region, out_region_arrays
        
        else:

            # go through an example
            region, region_arrays = self.get_filtered_example()
                
            # push out old example and load in next one
            out_region, out_region_arrays = self.region_tracker.get_region()
            self.region_tracker = RegionTracker(region, region_arrays)
            self.region_tracker.merge(region, region_arrays)

            self.valid_examples += 1
            if self.valid_examples % 1000 == 0:
                print self.valid_examples
            return out_region, out_region_arrays


class H5Handler(object):

    def __init__(self, h5_handle, tensor_dict, sample_size, batch_size=512, resizable=True):
        """Keep h5 handle and other relevant storing mechanisms
        """
        self.h5_handle = h5_handle
        self.tensor_dict = tensor_dict
        self.sample_size = sample_size
        for key in tensor_dict.keys():
            dataset_shape = [sample_size] + [int(i) for i in tensor_dict[key].get_shape()[1:]]
            maxshape = dataset_shape if resizable else None
            if "feature_metadata" in key:
                self.h5_handle.create_dataset(key, dataset_shape, maxshape=maxshape, dtype="S100")
            else:
                self.h5_handle.create_dataset(key, dataset_shape, maxshape=maxshape)
        self.resizable = resizable
        self.batch_size = batch_size
        self.batch_start = 0
        self.batch_end = self.batch_start + batch_size
        self.setup_tmp_arrays()

        
    def setup_tmp_arrays(self):
        """Setup numpy arrays as tmp storage before batch storage into h5
        """
        tmp_arrays = {}
        for key in self.tensor_dict.keys():
            dataset_shape = [self.batch_size] + [int(i) for i in self.tensor_dict[key].get_shape()[1:]]
            if "feature_metadata" in key:
                tmp_arrays[key] = np.array(["chrY:0-0" for i in xrange(self.batch_size)], dtype="S100")
            else:
                tmp_arrays[key] = np.zeros(dataset_shape)
        self.tmp_arrays = tmp_arrays
        self.tmp_arrays_idx = 0
        
        return

        
    def store_example(self, example_arrays):
        """Store an example into the tmp numpy arrays, push batch out if done with batch
        """
        for key in example_arrays.keys():
            if "feature_metadata" in key:
                self.tmp_arrays[key][self.tmp_arrays_idx] = example_arrays[key]
            elif "importance" in key:
                self.tmp_arrays[key][self.tmp_arrays_idx,:,:] = example_arrays[key]
            else:
                self.tmp_arrays[key][self.tmp_arrays_idx,:] = example_arrays[key]
        self.tmp_arrays_idx += 1

        # now if at end of batch, push out and reset tmp
        if self.tmp_arrays_idx == self.batch_size:
            self.push_batch()

        return
        
        
    def push_batch(self):
        """Go from the tmp array to the h5 file
        """
        for key in self.tmp_arrays.keys():
            if "feature_metadata" in key:
                self.h5_handle[key][self.batch_start:self.batch_end] = self.tmp_arrays[key].reshape((self.batch_size, 1))
            elif "importance" in key:
                self.h5_handle[key][self.batch_start:self.batch_end,:,:] = self.tmp_arrays[key]

            else:
                self.h5_handle[key][self.batch_start:self.batch_end,:] = self.tmp_arrays[key]

        # set new point in batch
        self.batch_start = self.batch_end
        self.batch_end += self.batch_size
        self.setup_tmp_arrays()
        self.tmp_arrays_idx = 0

        return
