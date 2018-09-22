"""Description: output layer - allows passing through examples 
or merging regions as desired
"""

import numpy as np


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
                tmp_arrays[key].fill("features=chr1:0-1000")
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
                if self.tmp_arrays["example_metadata"][batch_end][0].rstrip("\0") == "features=chrY:0-1000":
                    break
        self.batch_end = self.batch_start + batch_end

        # check if smaller than batch size
        test_key = self.example_keys[0]
        if self.h5_handle[test_key][self.batch_start:self.batch_end].shape[0] < batch_end:
            batch_end = self.h5_handle[test_key][self.batch_start:self.batch_end].shape[0]
            self.batch_end = self.batch_start + batch_end
        
        # save out
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
