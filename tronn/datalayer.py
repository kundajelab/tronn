"""Contains functions for I/O to tensorflow graphs
"""

import os
import glob
import h5py
import math
import logging

import abc

import numpy as np
import tensorflow as tf

from tronn.nets.filter_nets import filter_by_labels
from tronn.nets.filter_nets import filter_singleton_labels

from tronn.nets.sequence_nets import generate_dinucleotide_shuffles
from tronn.nets.sequence_nets import generate_scaled_inputs
from tronn.nets.sequence_nets import pad_examples


def setup_h5_files(data_dir):
    """quick helper function to go into h5 directory and organize h5 files
    """
    chroms = ["chr{}".format(i) for i in xrange(1,22)] + ["chrX", "chrY"]

    # save into a chromosome dict
    data_dict = {}
    for chrom in chroms:
        positives_files = sorted(glob.glob("{}/*{}[.]*.h5".format(data_dir, chrom)))
        positives_files = [filename for filename in positives_files if "negative" not in filename]
        training_negatives_files = sorted(glob.glob("{}/*training-negatives*{}[.]*.h5".format(data_dir, chrom)))
        global_negatives_files = sorted(glob.glob("{}/*genomewide-negatives*{}[.]*.h5".format(data_dir, chrom)))
        data_dict[chrom] = (
            positives_files,
            training_negatives_files,
            global_negatives_files)
        
    return data_dict


class DataLoader(object):
    """build the base level dataloader"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """set up dataloader
        """
        pass

    
    @abc.abstractmethod
    def build_raw_dataflow(self, batch_size, task_indices=[]):
        """build a raw dataflow from the files to tensors
        """
        pass

    
    def build_filtered_dataflow(
            self,
            batch_size,
            label_keys=["labels"],
            label_tasks=[],
            filter_tasks=[],
            singleton_filter_tasks=[],
            num_dinuc_shuffles=0,
            num_scaled_inputs=0,
            keep_keys=[],
            **kwargs):
        """build dataflow with additional preprocessing
        """
        # set up transform stack
        # TODO eventually set up different preprocessing stacks
        transform_stack = []
        if len(filter_tasks) != 0:
            # go through the list to subset more finegrained
            # in a hierarchical way
            for i in xrange(len(filter_tasks)):
                transform_stack.append((
                    filter_by_labels,
                    {"labels_key": "labels",
                     "filter_tasks": filter_tasks[i],
                     "name": "label_filter_{}".format(i)}))
                print filter_tasks[i]
        if len(singleton_filter_tasks) != 0:
            transform_stack.append((
                filter_singleton_labels,
                {"labels_key": "labels",
                 "filter_tasks": singleton_filter_tasks,
                 "name": "singleton_label_filter"}))
        if num_dinuc_shuffles > 0:
            transform_stack.append((
                generate_dinucleotide_shuffles,
                {"num_shuffles": num_dinuc_shuffles}))
        if num_scaled_inputs > 0:
            transform_stack.append((
                generate_scaled_inputs,
                {"num_scaled_inputs": num_scaled_inputs}))

        # build all together
        with tf.variable_scope("dataloader"):
        #with tf.variable_scope(""):
            # build raw dataflow
            inputs = self.build_raw_dataflow(
                batch_size,
                task_indices=label_tasks,
                label_keys=label_keys,
                **kwargs)
            
            # build transform stack
            master_params = {}
            master_params.update({"batch_size": batch_size})
            for transform_fn, params in transform_stack:
                master_params.update(params)
                print transform_fn
                inputs, _ = transform_fn(inputs, master_params)
            
            # adjust outputs as needed
            if len(keep_keys) > 0:
                new_inputs = {}
                for key in keep_keys:
                    new_inputs[key] = inputs[key]
                inputs = new_inputs
                
        return inputs

    
    def build_input_fn(self, batch_size, **kwargs):
        """build the dataflow function. will be called later in graph
        """
        def dataflow_fn():
            """dataflow function. must have no args.
            """
            inputs = self.build_filtered_dataflow(batch_size, **kwargs)
            return inputs, None

        return dataflow_fn


    def build_variant_input_fn(self, batch_size, features_prefix="features", **kwargs):
        """build a dataflow function that interleaves the variants
        """
        assert batch_size % 2 == 0
        batch_size = int(batch_size / 2)
        def dataflow_fn():
            """dataflow function. must have no args.
            """
            inputs = self.build_filtered_dataflow(batch_size, **kwargs)
            
            # interleave the variants
            ref_features = inputs["{}.ref".format(features_prefix)]
            alt_features = inputs["{}.alt".format(features_prefix)]
            tmp_features = tf.stack([ref_features, alt_features], axis=1)
            tmp_features = tf.reshape(
                tmp_features,
                [-1] + ref_features.get_shape().as_list()[1:])
            inputs["features"] = tmp_features
            
            # note that everything else needs to be padded
            inputs, _ = pad_examples(
                inputs,
                {"ignore": ["features"], "batch_size": batch_size})

            return inputs, None
        
        return dataflow_fn
    
    
    

class H5DataLoader(DataLoader):
    """build a dataloader from h5"""

    def __init__(self, h5_files, **kwargs):
        """initialize with data files
        """
        super(H5DataLoader, self).__init__(**kwargs)
        self.h5_files = h5_files
        self.num_examples = self.get_num_examples(h5_files)
        self.num_examples_per_file = self.get_num_examples_per_file(h5_files)

        
    @staticmethod
    def get_num_examples(h5_files, test_key="example_metadata"):
        """get total num examples in the dataset
        """
        num_examples = 0
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as hf:
                num_examples += hf[test_key].shape[0]
        return num_examples

    
    @staticmethod
    def get_num_examples_per_file(h5_files, test_key="example_metadata"):
        """get num examples per h5 file
        """
        num_examples_per_file = []
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as hf:
                num_examples_per_file.append(hf[test_key].shape[0])
        return num_examples_per_file
    
    
    def get_dataset_metrics(self):
        """Get class imbalances, num of outputs, etc
        """
        for data_file in data_files:
            print data_file
            print ""
        
        return

    
    @staticmethod
    def h5_to_slices(
            h5_handle,
            start_idx,
            batch_size,
            task_indices=[],
            keys=None,
            features_key="features",
            label_keys=["labels"],
            skip_keys=["label_metadata"]):
        """Get slices from the (open) h5 file back and pad with 
        null sequences as needed to fill the batch.

        Args:
        h5_handle: h5py handle on an opened file
        start_idx: start position in file
        batch_size: size of batch
        task_indices: list of task indices that you want, requires label set
        keys: what keys you want to pull out of data
        features_key: key of features
        labels_key: key of labels
        skip_keys: what keys to ignore
        
        Returns:
        slices of data
        """
        # calculate end idx
        end_idx = start_idx + batch_size

        # get keys
        if keys is None:
            keys = sorted([key for key in h5_handle.keys() if key not in skip_keys])
        else:
            keys = [key for key in keys if key not in skip_keys]
            
        # if end is within the file, extract out slice. otherwise, pad and pass out full batch
        slices = {}
        if end_idx < h5_handle[keys[0]].shape[0]:
            for key in keys:
                if "metadata" in key:
                    slices[key] = h5_handle[key][start_idx:end_idx].reshape((batch_size, 1)) # TODO don't reshape?
                else:
                    slices[key] = h5_handle[key][start_idx:end_idx][:].astype(np.float32)
        else:
            end_idx = h5_handle[keys[0]].shape[0]
            batch_padding_num = batch_size - (end_idx - start_idx)
            for key in keys:
                if "metadata" in key:
                    slice_tmp = h5_handle[key][start_idx:end_idx].reshape((end_idx-start_idx, 1))
                    slice_pad = np.array(
                        ["false=chrY:0-0" for i in xrange(batch_padding_num)]).reshape(
                            (batch_padding_num, 1))
                else:
                    slice_tmp = h5_handle[key][start_idx:end_idx][:].astype(np.float32)
                    slice_pad_shape = [batch_padding_num] + list(slice_tmp.shape[1:])
                    slice_pad = np.zeros(slice_pad_shape, dtype=np.float32)
                    
                slices[key] = np.concatenate([slice_tmp, slice_pad], axis=0)

        # adjust labels - concatenate desired labels and then get task indices
        labels = []
        for key in label_keys:
            labels.append(slices[key])
            slices["labels"] = np.concatenate(labels, axis=1)
            
        slices["labels"] = slices["labels"][:,task_indices]
                
        return slices

    
    @staticmethod
    def h5_to_tensors(
            h5_file,
            batch_size,
            task_indices=[],
            keys=None,
            skip_keys=["label_metadata"],
            features_key="features", # TODO do we need this?
            label_keys=["labels"],
            metadata_key="example_metadata",
            shuffle=True,
            num_epochs=1):
        """Extract batches from a single hdf5 file.
        
        Args:
        hdf5_file: hdf5 file with input data 
        batch_size: desired batch size to return to tensor
        tasks: use if a subset of tasks desired
        features_key: the key for the input dataset examples ('features' or 'importances')
        shuffle: shuffle the batches
        fake_task_num: how many EXTRA outputs to generate to fill in for multi-output
        
        Returns:
        features_tensor: a tensor (batch_size, feature_dims) for examples
        labels_tensor: a tensor (batch_size, num_tasks) for labels
        metadata_tensor: a tensor (batch_size, 1) for a metadata string (often region name)
        """
        # get hdf5 file params
        h5_handle = h5py.File(h5_file, "r")

        # keys
        if keys is None:
            keys = sorted(h5_handle.keys())
        keys = [key for key in keys if key not in skip_keys]
        
        # set up batch id producer
        max_batches = int(math.ceil(h5_handle[keys[0]].shape[0]/float(batch_size)))
        batch_id_producer = tf.train.range_input_producer(
            max_batches, shuffle=shuffle, seed=0, num_epochs=num_epochs)
        batch_id = batch_id_producer.dequeue()
        logging.debug("loading {0} with max batches {1}".format(
            os.path.basename(h5_file), max_batches))
        
        # determine the Tout
        tensor_dtypes = []
        for key in keys:
            if isinstance(h5_handle[key][0], basestring):
                tensor_dtypes.append(tf.string)
            else:
                tensor_dtypes.append(tf.float32)

        # labels get set up in the file so need to append to get them out
        keys.append("labels")
        tensor_dtypes.append(tf.float32)
        
        # function to get examples based on batch_id (for py_func)
        def batch_id_to_examples(batch_id):
            """Given a batch ID, get the features, labels, and metadata
            This is an important wrapper to be able to use TensorFlow's pyfunc.
            """
            batch_start = batch_id*batch_size
            slice_array = H5DataLoader.h5_to_slices(
                h5_handle,
                batch_start,
                batch_size,
                task_indices=task_indices,
                features_key=features_key,
                label_keys=label_keys)
            slice_list = []
            for key in keys:
                slice_list.append(slice_array[key])
            return slice_list
            
        # py_func
        inputs = tf.py_func(
            func=batch_id_to_examples,
            inp=[batch_id],
            Tout=tensor_dtypes,
            stateful=False, name='py_func_batch_id_to_examples')

        # set shapes
        for i in xrange(len(inputs)):
            if "labels" == keys[i]:
                inputs[i].set_shape([batch_size, len(task_indices)])
            elif "metadata" in keys[i]:
                inputs[i].set_shape([batch_size, 1])
            else:
                inputs[i].set_shape([batch_size] + list(h5_handle[keys[i]].shape[1:]))

        # make dict
        inputs = dict(zip(keys, inputs))
                
        return inputs


    @staticmethod
    def _generate_global_batch_to_file_dict(
            total_batches, total_batches_per_file):
        """generate a dictionary that maps the global batch idx 
        to the file index and local batch index
        """
        global_batch_to_file_idx = {}
        local_batch = 0
        current_file_idx = 0
        for global_batch in xrange(total_batches):
            if local_batch > total_batches_per_file[current_file_idx]:
                # go to next file idx
                local_batch = 0
                if (current_file_idx + 1) == len(h5_handles): # loop around as needed
                    current_file_idx = 0
                else:
                    current_file_idx += 1
            global_batch_to_file_idx[global_batch] = (current_file_idx, local_batch)
            local_batch += 1

        return global_batch_to_file_idx
    

    @staticmethod
    def h5_to_ordered_tensors(
            h5_files, 
            batch_size,
            task_indices=[],
            keys=None,
            skip_keys=["label_metadata"],
            features_key="features",
            label_keys=["labels"],
            metadata_key="example_metadata",
            num_epochs=1):
        """Extract batches from hdf5 file list. This is used to get ordered examples out 
        to re-merge back into regions.
        
        Args:
        hdf5_files: list of hdf5 files with input data 
        batch_size: desired batch size to return to tensor
        tasks: use if a subset of tasks desired
        features_key: the key for the input dataset examples ('features' or 'importances')
        
        Returns:
        features_tensor: a tensor (batch_size, feature_dims) for examples
        labels_tensor: a tensor (batch_size, num_tasks) for labels
        metadata_tensor: a tensor (batch_size, 1) for a metadata string (often region name)
        """
        # get h5 file list params
        logging.info("loading {}".format(" ".join(h5_files)))
        h5_handles = [h5py.File(h5_file, "r") for h5_file in h5_files]
        num_examples_per_file = H5DataLoader.get_num_examples_per_file(h5_files)
        total_batches_per_file = [int(math.ceil(num_examples) / float(batch_size))
                                  for num_examples in num_examples_per_file ]

        # keys
        if keys is None:
            keys = sorted(h5_handles[0].keys())
        keys = [key for key in keys if key not in skip_keys]
        
        # determine the Tout
        tensor_dtypes = []
        for key in keys:
            if isinstance(h5_handles[0][key][0], basestring):
                tensor_dtypes.append(tf.string)
            else:
                tensor_dtypes.append(tf.float32)

        # labels get set up in the file so need to append to get them out
        keys.append("labels")
        tensor_dtypes.append(tf.float32)
        
        # set up batch id producer
        max_batches = sum(total_batches_per_file)
        batch_id_queue = tf.train.range_input_producer(
            max_batches, shuffle=False, seed=0, num_epochs=num_epochs)
        batch_id = batch_id_queue.dequeue()
        logging.info("max batches: {}".format(max_batches))
        
        # generate a batch_to_file dictionary so it's easy to get the file
        global_batch_to_file_idx = H5DataLoader._generate_global_batch_to_file_dict(
            max_batches, total_batches_per_file)

        # Extract examples based on batch_id
        def batch_id_to_examples(batch_id):
            """Given a batch ID, get the features, labels, and metadata
            """
            # set up h5_handle and batch_start
            file_idx, local_batch = global_batch_to_file_idx[batch_id]
	    h5_handle = h5_handles[file_idx]
            batch_start = local_batch*batch_size
            slice_array = H5DataLoader.h5_to_slices(
                h5_handle,
                batch_start,
                batch_size,
                task_indices=task_indices,
                features_key=features_key,
                label_keys=label_keys)
            slice_list = []
            for key in keys:
                slice_list.append(slice_array[key])
            return slice_list

        # py_func
        inputs = tf.py_func(
            func=batch_id_to_examples,
            inp=[batch_id],
            Tout=tensor_dtypes,
            stateful=False, name='py_func_batch_id_to_examples')

        # set shapes
        for i in xrange(len(inputs)):
            if "labels" == keys[i]:
                inputs[i].set_shape([batch_size, len(task_indices)])
            elif "metadata" in keys[i]:
                inputs[i].set_shape([batch_size, 1])
            else:
                inputs[i].set_shape([batch_size] + list(h5_handles[0][keys[i]].shape[1:]))

        # make dict
        inputs = dict(zip(keys, inputs))

        return inputs


    def build_raw_dataflow(
            self,
            batch_size,
            task_indices=[],
            features_key="features",
            label_keys=["labels"],
            metadata_key="example_metadata",
            shuffle=True,
            shuffle_seed=1337):
        """build dataflow from files to tensors
        """
        logging.info(
            "loading data for task indices {0} from {1} hdf5_files: {2} examples".format(
                task_indices, len(self.h5_files), self.num_examples))
        
        # adjust task indices as needed
        if len(task_indices) == 0:
            num_labels = 0
            with h5py.File(self.h5_files[0], "r") as hf:
                for key in label_keys:
                    num_labels += hf[key].shape[1]
            task_indices = range(num_labels)

        # TODO if wanted to adjust ratio of positives to negatives, would need a different
        # h5_to_tensors method to mix at different ratios
        if shuffle:
            # use a thread for each hdf5 file to put together in parallel
            example_slices_list = [
                H5DataLoader.h5_to_tensors(
                    h5_file, batch_size, task_indices, label_keys=label_keys, shuffle=True)
                for h5_file in self.h5_files]
            min_after_dequeue = 10000
            capacity = min_after_dequeue + (len(example_slices_list)+10) * batch_size
            inputs = tf.train.shuffle_batch_join(
                example_slices_list,
                batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                seed=shuffle_seed,
                enqueue_many=True,
                name='batcher')
        else:
            # since ordered, keep hdf5 files in a list. unfortunately will not take
            # advantage of parallelism as well
    	    example_slices_list = H5DataLoader.h5_to_ordered_tensors(
                self.h5_files, batch_size, task_indices, label_keys=label_keys)
    	    inputs = tf.train.batch(
                example_slices_list,
                batch_size,
                capacity=100000,
                enqueue_many=True,
                name='batcher')
        
        return inputs


    def get_positive_weights_per_task(self):
        """Calculates positive weights to be used in weighted cross entropy
        """
        for filename_idx in range(len(self.data_files)):
            with h5py.File(self.data_files[filename_idx], 'r') as hf:
                file_pos = np.sum(hf['labels'], axis=0)
                file_tot = np.repeat(hf['labels'].shape[0], hf['labels'].shape[1])
                if filename_idx == 0:
                    total_pos = file_pos
                    total_examples = file_tot
                else:
                    total_pos += file_pos
                    total_examples += file_tot

        return np.divide(total_examples - total_pos, total_pos)
    

    def get_task_and_class_weights(self):
        """Calculates task and class weights to be used in positives focused loss
        """
        for filename_idx in range(len(self.data_files)):
            with h5py.File(self.data_files[filename_idx], 'r') as hf:
                file_pos = np.sum(hf['labels'], axis=0)
                file_tot = np.repeat(hf['labels'].shape[0], hf['labels'].shape[1])
                if filename_idx == 0:
                    total_pos = file_pos
                    total_examples = file_tot
                else:
                    total_pos += file_pos
                    total_examples += file_tot

        return np.divide(total_pos, total_examples - total_pos), np.divide(total_examples - total_pos, total_pos)

    

class ArrayDataLoader(DataLoader):
    """build a dataloader from numpy arrays"""

    def __init__(self, feed_dict, array_names, array_types, **kwargs):
    #def __init__(self, feed_dict, **kwargs):
        """keep names and shapes
        """
        self.feed_dict = feed_dict
        self.array_names = array_names
        self.array_types = array_types

        
    def build_raw_dataflow(self, batch_size, task_indices=[]):
        """load data
        """
        inputs = {}
        for i in xrange(len(self.array_names)):
            array_name = self.array_names[i]
            assert self.feed_dict[array_name] is not None
                
            inputs[array_name] = tf.placeholder(
                self.array_types[i],
                shape=[batch_size]+list(self.feed_dict[array_name].shape[1:]),
                name=array_name)

        # TODO this might be useful
        #new_inputs = tf.train.batch(
        #    inputs,
        #    batch_size,
        #    capacity=batch_size*3,
        #    enqueue_many=True,
        #    name='batcher')
            
        return inputs
            
    
class VariantDataLoader(DataLoader):
    """build a dataloader that starts from a vcf file"""

    def __init__(self, vcf_file, fasta_file):
        self.vcf_file = vcf_file
        self.fasta_file = fasta_file

    def load_raw_data(self, batch_size):

        return None
    

class BedDataLoader(DataLoader):
    """build a dataloader starting from a bed file"""

    def __init__(self, bed_file, fasta_file):
        self.bed_file = bed_file
        self.fasta_file = fasta_file


    def load_raw_data(self, batch_size):
        """load raw data
        """
        
        return None
