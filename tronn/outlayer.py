"""Description: output layer - allows passing through examples 
or merging regions as desired
"""

import numpy as np


def split_region(region):
    """Split region into a list of [chr, start, stop]
    """
    region = region.split(";")[-1].split("=")[1] # to account for new region names
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
        self.all_examples = 0
        self.valid_examples = 0
        self.reconstruct_regions = reconstruct_regions
        self.keep_negatives = keep_negatives
        self.pos_neg_imbalance = 0 # update only for positives, and allow negatives until it balances

        if self.reconstruct_regions: #?? whats going on here
            self.name_idx = -1
        else:
            # to make a unique name, keep the whole thing?
            self.name_idx = 0

        self.filter_by_prediction = filter_by_prediction
        self.filter_tasks = np.array(filter_tasks)
        
        filter_tasks_mask_tmp = np.zeros((1, int(tensor_dict["labels"].get_shape()[1])))
        filter_tasks_mask_tmp[0, filter_tasks] = 1
        self.filter_tasks_mask = filter_tasks_mask_tmp
            
        # initialize region tracker with first region in batch
        self.batch_region_arrays = self.sess.run(self.tensor_dict)
        region, region_arrays = self.build_example_dict_arrays()
        self.region_tracker = RegionTracker(region, region_arrays)
        self.region_tracker.merge(region, region_arrays)

        
    def build_example_dict_arrays(self, accuracy_cutoff=0.7):
        """Build dictionary of arrays for 1 example
        """

        while True:
            # debug check
            if self.all_examples % 1000 == 0:
                print "all examples: {}".format(self.all_examples)
            
            # if necessary, get a new batch
            if self.batch_pointer == self.batch_size:
                self.batch_pointer = 0
                self.batch_region_arrays = self.sess.run(self.tensor_dict)

            # filtering should happen here
            if (not self.keep_negatives) and (self.batch_region_arrays["negative"][self.batch_pointer] == 1):
                self.batch_pointer += 1
                self.all_examples += 1
                continue

            # if it's negative no imbalance, pass
            if (self.pos_neg_imbalance <= 0) and (self.batch_region_arrays["negative"][self.batch_pointer] == 1):
                self.batch_pointer += 1
                self.all_examples += 1
                continue

            if self.filter_by_prediction and self.batch_region_arrays["subset_accuracy"][self.batch_pointer] < accuracy_cutoff:
                #print self.batch_region_arrays["subset_accuracy"][self.batch_pointer]
                #print self.batch_region_arrays["labels"][self.batch_pointer, 0:10]
                #print self.batch_region_arrays["probs"][self.batch_pointer, 0:10]
                self.batch_pointer += 1
                self.all_examples += 1
                continue

            # if it passed conditions and was positive, increase the positive mark. if negative, decrease it (min 0)
            if self.batch_region_arrays["negative"][self.batch_pointer] == 1:
                self.pos_neg_imbalance = max(0, self.pos_neg_imbalance - 1)
            else:
                self.pos_neg_imbalance += 1
            
            break # if all filtering conditions were met
            
        # extract data and construct into region array dict
        region_arrays = {}
        for key in self.batch_region_arrays.keys():
            if key == "example_metadata":
                region = self.batch_region_arrays[key][self.batch_pointer,0]
                if self.name_idx == -1:
                    region_name = region.split(";")[self.name_idx].split("=")[1].split("::")[0]
                else:
                    # the whole region name is the unique id.
                    region_name = region
            elif "region_importance" in key: # TODO - change this later, if region merging
                region_arrays[key] = (
                    self.batch_region_arrays[key][self.batch_pointer,:,:],
                    "offset")
            else:
                region_arrays[key] = (
                    self.batch_region_arrays[key][self.batch_pointer,:],
                    "sum") # CHANGE THIS LATER
        self.batch_pointer += 1
        self.all_examples += 1
        
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
            # compare label vector to probs vector (NOT + XOR gate) and then mask by tasks that we care about.
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
                region, region_arrays = self.build_example_dict_arrays()

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
                print "valid examples: {}".format(self.valid_examples)
            
            return out_region, out_region_arrays
        
        else:

            # go through an example
            region, region_arrays = self.build_example_dict_arrays()
                
            # push out old example and load in next one
            out_region, out_region_arrays = self.region_tracker.get_region()
            self.region_tracker = RegionTracker(region, region_arrays)
            self.region_tracker.merge(region, region_arrays)

            self.valid_examples += 1
            if self.valid_examples % 1000 == 0:
                print "valid examples: {}".format(self.valid_examples)
            return out_region, out_region_arrays


class H5Handler(object):

    def __init__(
            self,
            h5_handle,
            tensor_dict,
            sample_size,
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
        self.is_tensor_input = is_tensor_input
        self.skip = skip
        self.direct_transfer = direct_transfer
        self.example_keys = []
        for key in tensor_dict.keys():
            if key in self.skip:
                continue
            if key in self.direct_transfer:
                self.h5_handle.create_dataset(key, data=tensor_dict[key])
                continue
            if is_tensor_input:
                dataset_shape = [sample_size] + [int(i) for i in tensor_dict[key].get_shape()[1:]]
            else:
                dataset_shape = [sample_size] + [int(i) for i in tensor_dict[key].shape[1:]]
            maxshape = dataset_shape if resizable else None
            if "example_metadata" in key:
                self.h5_handle.create_dataset(key, dataset_shape, maxshape=maxshape, dtype="S100")
            else:
                self.h5_handle.create_dataset(key, dataset_shape, maxshape=maxshape)
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
            
            dataset_shape = [self.batch_size] + [int(i) for i in self.h5_handle[key].shape[1:]]
                
            if "example_metadata" in key:
                tmp_arrays[key] = np.array(["false=chrY:0-0" for i in xrange(self.batch_size)], dtype="S100") # .reshape(self.batch_size, 1)
            else:
                tmp_arrays[key] = np.zeros(dataset_shape)
        self.tmp_arrays = tmp_arrays
        self.tmp_arrays_idx = 0

        return

    
    def add_dataset(self, key, shape, maxshape=None):
        """Add dataset and update numpy array
        """
        self.h5_handle.create_dataset(key, shape, maxshape=maxshape)
        self.example_keys.append(key)
        
        tmp_shape = [self.batch_size] + [int(i) for i in shape[1:]]
        self.tmp_arrays[key] = np.zeros(tmp_shape)
        
        return

    
    def store_example(self, example_arrays):
        """Store an example into the tmp numpy arrays, push batch out if done with batch
        """
        for key in self.example_keys:
            self.tmp_arrays[key][self.tmp_arrays_idx] = example_arrays[key]
                
            #if "example_metadata" in key:
            #    self.tmp_arrays[key][self.tmp_arrays_idx] = example_arrays[key]
            #elif "importance" in key:
            #    self.tmp_arrays[key][self.tmp_arrays_idx,:,:] = example_arrays[key]
            #elif "seqlets" in key:
            #    self.tmp_arrays[key][self.tmp_arrays_idx,:,:] = example_arrays[key]
            #else:
            #    self.tmp_arrays[key][self.tmp_arrays_idx,:] = example_arrays[key]
        self.tmp_arrays_idx += 1
        
        # now if at end of batch, push out and reset tmp
        if self.tmp_arrays_idx == self.batch_size:
            self.push_batch()

        return

    
    def store_batch(self, batch):
        """Coming from batch input
        """
        self.tmp_arrays = batch
        self.tmp_arrays["example_metadata"] = batch["example_metadata"].astype("S100")

        # todo maybe just call push batch?
        
        return

    
    def push_batch(self):
        """Go from the tmp array to the h5 file
        """
        for key in self.example_keys:
            if len(self.tmp_arrays[key].shape) == 1:
                self.h5_handle[key][self.batch_start:self.batch_end] = self.tmp_arrays[key].reshape(
                    self.tmp_arrays[key].shape[0], 1)
            else:
                self.h5_handle[key][self.batch_start:self.batch_end] = self.tmp_arrays[key]
            
                
                
                #if "example_metadata" in key:
                #    self.h5_handle[key][self.batch_start:self.batch_end] = self.tmp_arrays[key] #.reshape((self.batch_size, 1))
                #elif "importance" in key:
                #    self.h5_handle[key][self.batch_start:self.batch_end,:,:] = self.tmp_arrays[key]
                #else:
                #    self.h5_handle[key][self.batch_start:self.batch_end,:] = self.tmp_arrays[key]
            
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
                if self.tmp_arrays["example_metadata"][batch_end].rstrip("\0") == "false=chrY:0-0":
                    break
        self.batch_end = self.batch_start + batch_end
        
        for key in self.example_keys:
            #self.h5_handle[key][self.batch_start:self.batch_end] = self.tmp_arrays[key][0:batch_end]
            if len(self.tmp_arrays[key].shape) == 1:
                self.h5_handle[key][self.batch_start:self.batch_end] = self.tmp_arrays[key][0:batch_end].reshape(
                    batch_end, 1)
            else:
                self.h5_handle[key][self.batch_start:self.batch_end] = self.tmp_arrays[key][0:batch_end]

        return

    
    def chomp_datasets(self):
        """Once done adding things, if can resize then resize datasets
        """
        assert self.resizable == True

        for key in self.example_keys:
            dataset_final_shape = [self.batch_end] + [int(i) for i in self.h5_handle[key].shape[1:]]
            self.h5_handle[key].resize(dataset_final_shape)
            
        return


class H5InputHandler(object):

    def __init__(self, h5_handle, batch_size=512, flatten=False, skip=["label_metadata"]):
        self.h5_handle = h5_handle
        self.example_idx = 0
        self.batch_end = batch_size
        self.batch_size = batch_size
        self.batch_idx = 0
        self.total_examples = self.h5_handle["example_metadata"].shape[0]
        self.skip = skip
        self.flatten = flatten
        self.pull_batch()
        

    def pull_batch(self):
        """Pull out a batch of examples
        """
        if self.example_idx + self.batch_size > self.total_examples:
            self.batch_end = self.total_examples
        
        batch_arrays = {}
        for key in self.h5_handle.keys():
            if key in self.skip:
                continue

            if self.flatten and key == "features":
                batch_arrays[key] = np.squeeze(self.h5_handle[key][self.example_idx:self.batch_end]).transpose(0, 2, 1)
            else:
                batch_arrays[key] = self.h5_handle[key][self.example_idx:self.batch_end]
            
            #if "example_metadata" in key:
            #    batch_arrays[key] = self.h5_handle[key][self.example_idx:self.batch_end]
            #elif "feature" in key:
            #    batch_arrays[key] = self.h5_handle[key][self.example_idx:self.batch_end,:,:,:]
            #else:
            #    batch_arrays[key] = self.h5_handle[key][self.example_idx:self.batch_end,:]
            
        self.batch_arrays = batch_arrays
        self.batch_idx = 0
        self.example_idx = self.batch_end
        self.batch_end += self.batch_size
                
        return


    def get_example_array(self):
        """Pull out one example
        """
        example_array = {}
        for key in self.batch_arrays.keys():
            example_array[key] = self.batch_arrays[key][self.batch_idx]
            
            #if "example_metadata" in key:
            #    example_array[key] = self.batch_arrays[key][self.batch_idx,0]
            #elif "feature" in key:
            #    example_array[key] = self.batch_arrays[key][self.batch_idx,:,:,:]
            #else:
            #    example_array[key] = self.batch_arrays[key][self.batch_idx,:]
        self.batch_idx += 1
                
        if self.batch_idx == self.batch_size:
            self.pull_batch()

        return example_array
