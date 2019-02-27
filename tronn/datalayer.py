"""Contains functions for I/O to tensorflow
"""

import os
import six
import abc
import gzip
import glob
import h5py
import json
import math
import random
import logging
import threading

import numpy as np
import pandas as pd
import tensorflow as tf

from tronn.learn.cross_validation import setup_train_valid_test
from tronn.preprocess.fasta import GenomicIntervalConverter
from tronn.nets.util_nets import rebatch
from tronn.util.utils import DataKeys


class DataLoader(object):
    """build the base level dataloader"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, data_dir):
        """set up dataloader. minimally should require the data directory,
        and body of this function collects the files.
        """
        raise NotImplementedError("required method, implement in child class!")

    
    @abc.abstractmethod
    def build_generator(self):
        """build a generator function from the files to intervals.
        must return the generator, a dict of tf dtypes, and a dict of shapes
        generator is expected to return one example at a time.
        """
        raise NotImplementedError("required method, implement in child class!")

    
    @staticmethod
    def build_target_select_function(target_indices, target_key=DataKeys.LABELS):
        """build a function to gather selected indices from dataset
        
        Args:
          target indices: list of indices to index the final label set
          target_key: use in case the label tensor is not named "labels"

        Returns:
          map_fn: function to slice the labels
        """
        def map_fn(features, labels):
            features[target_key] = tf.gather(
                features[target_key], target_indices, axis=-1)
            return features, labels
        
        return map_fn

    
    @staticmethod
    def build_target_filter_function(target_set):
        """build a function for filtering examples based on whether the
        target is positive (in classification, positive label)

        Args:
          target_key: the key to the target tensor
          target_indices: list of indices of targets of interest
          filter_type: how to reduce ("any" or "all") for passing filter

        Returns:
          filter_fn: function to feed to tf.Dataset.filter
        """
        keys_and_indices = target_set[0]
        params = target_set[1]
        reduce_type = params.get("reduce_type", "any")
        logging.debug("building filter fn for {}".format(target_set))
            
        def filter_fn(features, labels):
            filter_targets = []
            for key, indices in keys_and_indices:
                if len(indices) != 0:
                    filter_targets.append(tf.gather(features[key], indices, axis=-1))
                else:
                    filter_targets.append(features[key])
            filter_targets = tf.concat(filter_targets, axis=-1)
            filter_targets = tf.greater(filter_targets, 0)
            if reduce_type == "any":
                passes_filter = tf.reduce_any(filter_targets)
            elif reduce_type == "all":
                passes_filter = tf.reduce_all(filter_targets)
            elif reduce_type == "min":
                min_count = float(params.get("min", 1))
                positive_num = tf.reduce_sum(tf.cast(filter_targets, tf.float32))
                passes_filter = tf.greater_equal(positive_num, min_count)
            elif reduce_type == "max":
                max_count = float(params.get("max", filter_targets.get_shape().aslist()[-1]))
                positive_num = tf.reduce_sum(tf.cast(filter_targets, tf.float32))
                passes_filter = tf.less_equal(positive_num, max_count)
            else:
                raise NotImplementedError("requested filter type not implemented!")
            return passes_filter
        
        return filter_fn
    
            
    def build_dataset_dataflow(
            self,
            batch_size,
            shuffle=True,
            filter_targets=[],
            singleton_filter_targets=[],
            target_indices=[],
            encode_onehot_features=True,
            num_threads=16,
            num_to_prefetch=5,
            deterministic=False,
            shuffle_min=10000,
            lock=threading.Lock(),
            **kwargs):
        """build dataflow from generator(s) to tf Dataset

        Args:
          batch_size: batch size for batch going into the graph
          shuffle: whether to shuffle examples
          filter_targets: dict of keys with indices for filtering
          singleton_filter_targets: list of keys for removing singletons
          num_threads: number of threads to use for data loading
          encode_onehot_features: if tensor with key "features" is a string
            of values that are not yet one-hot, then encode the features.
            NOTE: this replaces "features" with the one hot encoded tensor.
          num_to_prefetch: how many examples/batches to prefetch

        Returns:
          tf.data.Dataset instance.
        """
        # build a different generator object for each data file
        generators = [
            self.build_generator(
                batch_size=batch_size,
                shuffle=shuffle,
                lock=lock,
                **kwargs)[0](data_file)
            for data_file in self.data_files]
            
        # get dtypes and shapes
        _, dtypes_dict, shapes_dict = self.build_generator(
            batch_size=batch_size, shuffle=shuffle, lock=lock, **kwargs)

        # convert to dataset
        def from_generator(i):
            dataset = tf.data.Dataset.from_generator(
                lambda i: generators[i],
                (dtypes_dict, tf.int32),
                output_shapes=(shapes_dict, ()),
                args=(i,))
            return dataset
        
        # set up interleaved datasets
        dataset = tf.data.Dataset.from_tensor_slices(list(range(len(self.data_files)))).apply(
            tf.contrib.data.parallel_interleave(
                lambda file_idx: from_generator(file_idx),
                #lambda filename: from_generator(filename),
                cycle_length=num_threads,
                #block_length=batch_size, # all of these seem to slow it down
                #cycle_length=len(self.data_files),
                #prefetch_input_elements=batch_size,
                buffer_output_elements=1, 
                sloppy=deterministic))
        
        # filter on specific keys and indices
        if len(filter_targets) != 0:
            for target_set in filter_targets:
                dataset = dataset.filter(
                    DataLoader.build_target_filter_function(target_set))
                
        # gather labels as needed
        if len(target_indices) > 0:
            dataset = dataset.map(
                DataLoader.build_target_select_function(
                    target_indices))
        
        # shuffle
        if shuffle:
            dataset = dataset.shuffle(shuffle_min)

        # batch
        if False:
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(batch_size))
        else:
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(
                    map_func=DataLoader.encode_onehot_sequence_single,
                    batch_size=batch_size,
                    drop_remainder=True))

        # prefetch
        dataset = dataset.prefetch(num_to_prefetch)

        return dataset


    def _generator_to_tensors(
            self,
            h5_file,
            batch_size,
            shuffle=True,
            target_indices=[],
            lock=threading.Lock(),
            **kwargs):
        """take one generator and get tensors through py_func
        """
        # set up generator and ordered keys, dtypes
        generator, dtypes_dict, shapes_dict = self.build_generator(
            batch_size=batch_size, shuffle=shuffle, lock=lock, **kwargs)
        ordered_keys = list(dtypes_dict.keys())
        ordered_dtypes = [dtypes_dict[key] for key in ordered_keys]
        h5_to_generator = generator(h5_file, yield_single_examples=False)

        # wrapper function to get generator into py func
        def h5_to_tensors(batch_id):
            slice_array, _ = next(h5_to_generator)
            results = []
            for key in ordered_keys:
                results.append(slice_array[key])
            return results
        
        # py func
        inputs = tf.py_func(
            func=h5_to_tensors,
            inp=[0], #[batch_id]
            Tout=ordered_dtypes,
            stateful=True,
            name='py_func_batch_id_to_examples')
        
        # set shape
        for i in range(len(ordered_keys)):
            inputs[i].set_shape(
                [batch_size]+list(shapes_dict[ordered_keys[i]]))
                
        # back to dict
        inputs = dict(list(zip(ordered_keys, inputs)))

        return inputs

    
    @staticmethod
    def _queue_filter(inputs, filter_fn, name, batch_size):
        """takes in inputs and filter function
        and filters, puts back into a queue
        """
        # wrap the filter fn because only using features (not labels)
        def filter_fn_wrapper(inputs):
            return filter_fn(inputs, inputs)

        # get condition mask
        passes_filter = tf.map_fn(
            filter_fn_wrapper,
            inputs,
            dtype=tf.bool)

        # gather in place
        keep_indices = tf.reshape(tf.where(passes_filter), [-1])
        for key in list(inputs.keys()):
            inputs[key] = tf.gather(inputs[key], keep_indices)

        # rebatch
        inputs, _ = rebatch(
            inputs,
            {"name": name,
             "batch_size": batch_size,
             "num_queue_threads": 4})
        
        return inputs
    
    
    def build_queue_dataflow(
            self,
            batch_size,
            shuffle=True,
            filter_targets=[],
            singleton_filter_targets=[],
            target_indices=[],
            encode_onehot_features=True,
            num_threads=16,
            num_to_prefetch=5,
            deterministic=False,
            shuffle_min=10000,
            lock=threading.Lock(),
            **kwargs):
        """wrap the generator with py_func
        """
        # set up py_func on top of generator for each file
        example_slices_list = [
            self._generator_to_tensors(
                data_file,
                batch_size,
                shuffle=shuffle,
                target_indices=target_indices,
                lock=lock,
                **kwargs)
            for data_file in self.data_files]
        
        # determine min to shuffle, capacity
        if len(example_slices_list) > 1:
            min_after_dequeue = shuffle_min
        else:
            min_after_dequeue = batch_size * 3
        capacity = min_after_dequeue + (len(example_slices_list)+10) * batch_size

        # shuffle batch join
        if shuffle:
            inputs = tf.train.shuffle_batch_join(
                example_slices_list,
                batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                seed=42,
                enqueue_many=True,
                name='batcher')
        else:
            inputs = tf.train.batch_join(
                example_slices_list,
                batch_size,
                capacity=capacity,
                enqueue_many=True,
                name='batcher')
            
        # filter on specific keys and indices
        if len(filter_targets) != 0:
            filter_idx = 0
            for filter_set in filter_targets:
                inputs = DataLoader._queue_filter(
                    inputs,
                    DataLoader.build_target_filter_function(filter_set),
                    "targets_filter_{}".format(filter_idx),
                    batch_size)
                filter_idx += 1

        # gather labels as needed
        if len(target_indices) > 0:
            inputs[DataKeys.LABELS] = tf.gather(
                inputs[DataKeys.LABELS], axis=1)

        # onehot encode the batch
        inputs[DataKeys.FEATURES] = tf.map_fn(
            DataLoader.encode_onehot_sequence,
            inputs[DataKeys.FEATURES],
            dtype=tf.float32)
        
        return inputs, None
    
    
    def build_input_fn(self, batch_size, use_queues=True, **kwargs):
        """build the function that will be called later after 
        graph is instantiated. Note that there are two options, 
        can either build using tf Dataset or use queues.
        Queues are currently faster, though filtering is not
        yet implemented on the queues.

        Args:
          batch_size: batch size for tensors going into graph
        
        Returns:
          input_fn: input function that will be called after
            graph creation
        """
        def input_fn():
            """data flow fn. must have no args.
            """
            if use_queues:
                return self.build_queue_dataflow(batch_size, **kwargs)
            else:
                return self.build_dataset_dataflow(batch_size, **kwargs)

        
        return input_fn
        

    @staticmethod
    def encode_onehot_sequence(sequence):
        """adjust to onehot, given indices tensor

        Args:
          sequence: single sequence, with basepairs encoded as
           integers 0-4, in the order ACGTN
        
        Returns:
          sequence: one hot encoded sequence, set up as NHWC
            where H is 1, W is length of sequence, and C is basepair
        """
        sequence = tf.one_hot(sequence, 5, axis=-1) # {seq_len, 5}
        sequence = tf.expand_dims(sequence, axis=0) # {1, seq_len, 5}
        sequence = tf.gather(sequence, [0, 1, 2, 3], axis=2) # {1, seq_len, 4]
        
        return sequence

    
    @staticmethod
    def encode_onehot_sequence_single(features, labels):
        """convenience function for converting interval to one hot
        sequence in the tf Dataset framework
        
        Args:
          features: dict of tensors
          labels: not used, placeholder for tf.data.Dataset framework

        Returns:
          features: the tensor in "features" is now one hot
          labels: no changes made
        """
        features[DataKeys.FEATURES] = DataLoader.encode_onehot_sequence(
            features[DataKeys.FEATURES])
        
        return features, labels

    
    @staticmethod
    def encode_onehot_sequence_batch(features, labels):
        """convenience function for batch converting intervals 
        to sequence in the tf Dataset framework

        Args:
          features: dict of tensors
          labels: not used, placeholder for tf.data.Dataset framework

        Returns:
          features: the tensor in "features" is now one hot
          labels: no changes made
        """
        features[DataKeys.FEATURES] = tf.map_fn(
            DataLoader.encode_onehot_sequence,
            features[DataKeys.FEATURES],
            dtype=tf.float32)
        
        return features, labels

    

class H5DataLoader(DataLoader):
    """build a dataloader to handle h5 files"""

    def __init__(
            self,
            data_dir=None,
            data_files=[],
            fasta=None,
            dataset_json={},
            **kwargs):
        """initialize with data files
        
        Args:
          data_dir: directory of h5 files
          data_files: a list of h5 filenames
          fasta: fasta file for getting sequence from BED intervals on the fly
        """
        # save dir and files
        self.data_dir = data_dir
        self.h5_files = data_files
        
        # resolve files and dir
        self.h5_files = self._resolve_dir_and_files(
            self.data_dir, data_files)
            
        # set up fasta
        self.fasta = fasta

        # calculate basic stats
        self.num_examples = self.get_num_examples(self.h5_files)
        self.num_examples_per_file = self.get_num_examples_per_file(self.h5_files)

        # save h5 files as data files
        self.data_files = self.h5_files
        
        # by the end of all this, needs to have data files
        assert len(self.data_files) > 0


    @staticmethod
    def _resolve_dir_and_files(data_dir, h5_files):
        """make sure that the desired data dir and full path to
        h5 files are consistent
        
        Args:
          data_dir: required, the data dir where desired files are
          h5_files: either a list of files or None, paths will be 
            adjusted to match data_dir

        Returns:
          h5_files: consistent list of data files
        """
        if data_dir is not None:
            if len(h5_files) > 0:
                h5_files = [
                    "{}/{}".format(data_dir, os.path.basename(h5_file))
                    for h5_file in h5_files]
            else:
                h5_files = glob.glob("{}/*h5".format(data_dir))
                h5_files = [h5_file for h5_file in h5_files
                            if "manifold" not in h5_file]
        else:
            assert len(h5_files) > 0

        return h5_files


    def get_chromosomes(self):
        """figure out which chromosomes are in the dataset
        """
        chroms = []
        for data_file in self.data_files:
            with h5py.File(data_file, "r") as hf:
                file_chroms = list(hf["/"].attrs["chromosomes"])
            chroms = list(set(chroms + file_chroms))
        chroms = sorted(chroms)

        return chroms

    
    @staticmethod
    def organize_h5_files_by_chrom(
            h5_files, genome="human"):
        """helper function to go into h5 directory and organize h5 files
        
        Args:
          h5_files: list of h5_files, must contain the chrom name in the filename

        Returns:
          chrom_file_dict: dict of files. key is chrom name, val is tuple of
            lists - (positives, training negatives, genomewide negatives)
        """
        if genome == "human":
            chroms = ["chr{}".format(i) for i in range(1,23)] + ["chrX", "chrY"]
        else:
            raise Exception("unrecognized genome!")

        # save into a chromosome dict
        # TODO(dk) eventually adjust this to check h5 attributes instead of filename, safer that way
        chrom_file_dict = {}
        for chrom in chroms:
            chrom_files = [
                chrom_file for chrom_file in h5_files
                if "{}.".format(chrom) in chrom_file]
            positives_files = sorted(
                [filename for filename in chrom_files if "negative" not in filename])
            training_negatives_files = sorted(
                [filename for filename in chrom_files if "training-negatives" in filename])
            global_negatives_files = sorted(
                [filename for filename in chrom_files if "genomewide-negatives" in filename])
            chrom_file_dict[chrom] = (
                positives_files,
                training_negatives_files,
                global_negatives_files)
            
        return chrom_file_dict


    @staticmethod
    def get_size_ordered_chrom_list(
            chrom_file_dict,
            include_training_negatives=True,
            include_global_negatives=False):
        """returns an ordered list of chrom keys based on size of files

        Args:
          chrom_file_dict: dictionary of chrom files. 
            key is chrom name, val is list of files
       
        Returns:
          ordered_chrom_keys: list of chrom names
        """
        chrom_keys = []
        num_examples_per_chrom = []
        for chrom in sorted(chrom_file_dict.keys()):
            chrom_keys.append(chrom)
            num_chrom_examples = H5DataLoader.get_num_examples(chrom_file_dict[chrom][0])
            if include_training_negatives:
                num_chrom_examples += H5DataLoader.get_num_examples(chrom_file_dict[chrom][1])
            if include_global_negatives:
                num_chrom_examples += H5DataLoader.get_num_examples(chrom_file_dict[chrom][2])
            num_examples_per_chrom.append(num_chrom_examples)

        # order in descending example total
        num_examples_per_chrom = np.array(num_examples_per_chrom)
        indices = np.argsort(num_examples_per_chrom)[::-1]
        ordered_chrom_keys = np.array(chrom_keys)[indices]
        ordered_num_examples = num_examples_per_chrom[indices]

        return ordered_chrom_keys, ordered_num_examples

    
    @staticmethod
    def get_num_examples(h5_files, test_key=DataKeys.SEQ_METADATA):
        """get total num examples across the h5 files in the list
        """
        num_examples = 0
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as hf:
                num_examples += hf[test_key].shape[0]
        return num_examples

    
    @staticmethod
    def get_num_examples_per_file(h5_files, test_key=DataKeys.SEQ_METADATA):
        """get num examples per h5 file
        """
        num_examples_per_file = []
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as hf:
                num_examples_per_file.append(hf[test_key].shape[0])
        return num_examples_per_file

    
    def get_num_targets(
            self,
            targets=[([(DataKeys.LABELS, [])], {"reduce_type": "none"})],
            target_indices=[],
            test_file_idx=0):
        """get number of labels
        """
        if len(target_indices) == 0:
            num_targets = 0
            # need to manually figure out what the joint target set looks like
            for keys_and_indices, params in targets:
                reduce_type = params.get("reduce_type", "none")
                if reduce_type == "all":
                    num_targets += 1
                elif reduce_type == "any":
                    num_targets += 1
                elif reduce_type == "min":
                    num_targets += 1
                else:
                    # need to grab the set
                    with h5py.File(self.h5_files[test_file_idx], "r") as hf:
                        for key, indices in keys_and_indices:
                            if len(indices) != 0:
                                num_targets += len(indices)
                            else:
                                num_targets += hf[key].shape[1]
        else:
            # the target num is the length of the indices
            num_targets = len(target_indices)

        logging.info(
            "Found {} targets across target set(s) {}, finally indexed with {}.".format(
                num_targets,
                ";".join([target[0][0][0] for target in targets]),
                target_indices))
        
        return num_targets

    
    @staticmethod
    def h5_to_slices(
            h5_handle,
            start_idx,
            batch_size,
            keys_to_load=None,
            targets=[],
            target_indices=[]):
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
        # TODO set up predefined numpy arrays?
        #assert tmp_arrays is not None
        
        # calculate end idx
        end_idx = start_idx + batch_size
            
        # if end is within the file, extract out slice. otherwise, pad and pass out full batch
        slices = {}
        if end_idx <= h5_handle[keys_to_load[0]].shape[0]:
            for key in keys_to_load:
                if h5_handle[key][0].dtype.char == "S":
                    # reshape if len(dims) is 1
                    # TODO - don't do this anymore?
                    if len(h5_handle[key][0].shape) == 0:
                        slices[key] = h5_handle[key][start_idx:end_idx].reshape((batch_size, 1))
                    else:
                        slices[key] = h5_handle[key][start_idx:end_idx]
                else:
                    slices[key] = h5_handle[key][start_idx:end_idx][:].astype(np.float32)
        else:
            # TODO - don't pad anymore?
            end_idx = h5_handle[keys_to_load[0]].shape[0]
            batch_padding_num = batch_size - (end_idx - start_idx)
            for key in keys_to_load:
                if "metadata" in key:
                    slice_tmp = h5_handle[key][start_idx:end_idx].reshape((end_idx-start_idx, 1))
                    slice_pad = np.array(
                        ["features=chr1:0-1000" for i in range(batch_padding_num)]).reshape(
                            (batch_padding_num, 1))
                elif h5_handle[key][0].dtype.char == "S":
                    if len(h5_handle[key][0].shape) == 0:
                        # reshape if len(dims) is 1
                        slice_tmp = h5_handle[key][start_idx:end_idx].reshape((end_idx-start_idx, 1))
                        slice_pad = np.array(
                            ["N" for i in range(batch_padding_num)]).reshape(
                                (batch_padding_num, 1))
                    else:
                        slice_tmp = h5_handle[key][start_idx:end_idx]
                        slice_pad = np.array(
                            ["N" for i in range(batch_padding_num)])
                else:
                    slice_tmp = h5_handle[key][start_idx:end_idx][:].astype(np.float32)
                    slice_pad_shape = [batch_padding_num] + list(slice_tmp.shape[1:])
                    slice_pad = np.zeros(slice_pad_shape, dtype=np.float32)
                    
                slices[key] = np.concatenate([slice_tmp, slice_pad], axis=0)

        # concatenate labels
        if len(targets) != 0:
            labels = []

            for keys_and_indices, params in targets:
                labels_subset = []

                # pull all the keys and indices into a subset
                for key, indices in keys_and_indices:
                    if len(indices) == 0:
                        labels_subset.append(slices[key])
                    else:
                        labels_subset.append(slices[key][:, indices])
                labels_subset = np.concatenate(labels_subset, axis=1)

                # then reduce as needed
                reduce_type = params.get("reduce_type", "none")
                if reduce_type == "all":
                    labels_subset = np.all(labels_subset, axis=1, keepdims=True)
                elif reduce_type == "any":
                    labels_subset = np.any(labels_subset, axis=1, keepdims=True)
                elif reduce_type == "none":
                    labels_subset = labels_subset
                else:
                    raise ValueError("reduce type not recognized!")
                    
                # then append to labels
                labels.append(labels_subset)

            # concatenate them all
            labels = np.concatenate(labels, axis=1)

            # and finally, adjust for the target_indices
            if len(target_indices) != 0:
                slices[DataKeys.LABELS] = labels[:, target_indices]
            else:
                slices[DataKeys.LABELS] = labels

        return slices


    def _get_clean_keys(
            self,
            keys,
            skip_keys=[],
            test_file_idx=0,
            example_key=DataKeys.SEQ_METADATA):
        """remove keys that are possibly metadata
        do this by checking the num rows, anything with the 
        wrong num rows is not "clean"
        """
        test_h5 = self.h5_files[test_file_idx]
        with h5py.File(test_h5, "r") as h5_handle:
            if len(keys) == 0:
                keys = list(h5_handle.keys())
            num_examples = h5_handle[example_key].shape[0]
            
            for key in keys:
                # check if actually a dataset (vs hdf5 group)
                if not isinstance(h5_handle[key], h5py.Dataset):
                    skip_keys.append(key)
                    continue
                # check if scalar
                if h5_handle[key].shape == 0:
                    skip_keys.append(key)
                    continue
                # check if different shape
                if h5_handle[key].shape[0] != num_examples:
                    skip_keys.append(key)
                    continue
            # filter the keys
            clean_keys = [
                key for key in sorted(h5_handle.keys())
                if key not in skip_keys]

        return clean_keys, skip_keys

    
    def _get_dtypes_and_shapes(
            self,
            keys,
            test_file_idx=0):
        """get the tf dtypes and shapes to put into tf.dataset
        """
        test_h5 = self.h5_files[test_file_idx]
        with h5py.File(test_h5, "r") as h5_handle:
            dtypes_dict = {}
            shapes_dict = {}
            for key in keys:
                if isinstance(h5_handle[key][0], str):
                    dtypes_dict[key] = tf.string
                    shapes_dict[key] = [1]
                elif h5_handle[key][0].dtype.char == "S":
                    dtypes_dict[key] = tf.string
                    shapes_dict[key] = h5_handle[key].shape[1:]
                else:
                    dtypes_dict[key] = tf.float32
                    shapes_dict[key] = h5_handle[key].shape[1:]

        return dtypes_dict, shapes_dict

    
    def _add_targets_dtype_and_shape(
            self,
            keys,
            dtypes_dict,
            shapes_dict,
            targets,
            target_indices=[],
            test_file_idx=0):
        """labels get pulled together from various h5 dataset,
        set up dtype and shape
        """
        keys.append(DataKeys.LABELS)
        dtypes_dict[DataKeys.LABELS] = tf.float32
        shapes_dict[DataKeys.LABELS] = [self.get_num_targets(
            targets=targets, target_indices=target_indices)]
        
        return keys, dtypes_dict, shapes_dict


    def _add_onehot_features_dtype_and_shape(
            self,
            keys,
            dtypes_dict,
            shapes_dict,
            seq_len=1000):
        """if adding features on the fly (inside the generator), 
        add in dtype and shape
        """
        if DataKeys.FEATURES not in keys:
            keys.append(DataKeys.FEATURES)
            dtypes_dict[DataKeys.FEATURES] = tf.uint8
            shapes_dict[DataKeys.FEATURES] = [seq_len]
        else:
            # replace
            dtypes_dict[DataKeys.FEATURES] = tf.uint8
            shapes_dict[DataKeys.FEATURES] = [seq_len]
        
        return keys, dtypes_dict, shapes_dict
    

    def build_generator(
            self,
            batch_size=256,
            task_indices=[],
            keys=[],
            skip_keys=[],
            targets=[([(DataKeys.LABELS, [])], {"reduce_type": "none"})],
            target_indices=[],
            examples_subset=[],
            seq_len=1000,
            lock=threading.Lock(),
            shuffle=True):
        """make a generator, pulls batches into numpy array from h5
        only goes through data ONCE

        NOTE: this is the key function that custom dataloaders must build
        it must return a generator, a tf dtypes dictionary, and a shape
        dictionary.
        """
        assert self.fasta is not None

        # get clean keys
        clean_keys_to_load, skip_keys = self._get_clean_keys(keys, skip_keys)

        # set up dtype and shape dicts
        dtypes_dict, shapes_dict = self._get_dtypes_and_shapes(clean_keys_to_load)
        clean_keys, dtypes_dict, shapes_dict = self._add_targets_dtype_and_shape(
            list(clean_keys_to_load), dtypes_dict, shapes_dict, targets, target_indices=task_indices)
        clean_keys, dtypes_dict, shapes_dict = self._add_onehot_features_dtype_and_shape(
            clean_keys, dtypes_dict, shapes_dict, seq_len=seq_len)

        # pull out the fasta to throw into the generator
        #fasta = self.fasta

        # make the generator
        # explicitly designed this way to work with tf Dataset
        class Generator(object):

            def __init__(self, fasta, batch_size):
                self.fasta = fasta
                self.batch_size = batch_size
                
            
            def __call__(self, h5_file, yield_single_examples=True):
                """run the generator"""

                batch_size = self.batch_size
                fasta = self.fasta
                
                if len(examples_subset) != 0:
                    batch_size = 1

                # set up interval to sequence converter
                converter = GenomicIntervalConverter(lock, fasta, batch_size)
                
                # open h5 file
                with h5py.File(h5_file, "r") as h5_handle:
                    test_key = list(h5_handle.keys())[0]

                    # if using examples, then get the indices and change batch size to 1
                    if len(examples_subset) != 0:
                        max_batches = len(examples_subset)
                        batch_ids = np.where(
                            np.isin(
                                h5_handle[DataKeys.SEQ_METADATA],
                                examples_subset))[0].tolist()
                        yield_single_examples = False
                    else:
                        # set up batch id total and get batches
                        max_batches = int(math.ceil(h5_handle[test_key].shape[0]/float(batch_size)))
                        batch_ids = list(range(max_batches))

                    # shuffle batches as needed
                    if shuffle:
                        random.Random(42).shuffle(batch_ids)
                            
                    # logging
                    logging.debug("loading {0} with batch size {1} to get batches {2}".format(
                        os.path.basename(h5_file), batch_size, max_batches))
                    
                    # and go through batches
                    try:
                        assert len(clean_keys_to_load) != 0
                        
                        for batch_id in batch_ids:
                            batch_start = batch_id*batch_size
                            slice_array = H5DataLoader.h5_to_slices(
                                h5_handle,
                                batch_start,
                                batch_size,
                                keys_to_load=clean_keys_to_load,
                                targets=targets,
                                target_indices=target_indices)

                            # onehot encode on the fly
                            # TODO keep the string sequence
                            slice_array[DataKeys.FEATURES] = converter.convert(
                                slice_array["example_metadata"])
                            
                            # yield
                            if yield_single_examples: # NOTE: this is the most limiting step
                                for i in range(batch_size):
                                    yield ({
                                        key: value[i]
                                        for key, value in six.iteritems(slice_array)
                                    }, 1.)
                            else:
                                yield (slice_array, 1.)
                            
                    except ValueError as value_error:
                        logging.debug(value_error)
                        logging.info("Stopping {}".format(h5_file))
                        raise StopIteration

                    finally:
                        converter.close()
                        print("finished {}".format(h5_file))
                            
        # instantiate
        generator = Generator(self.fasta, batch_size)
        
        return generator, dtypes_dict, shapes_dict


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

    
    def setup_cross_validation_dataloaders(
            self,
            kfolds=10,
            valid_folds=[0],
            test_folds=[1],
            regression=False):
        """takes this dataloader and generates three new ones:
        1) train
        2) validation
        3) test
        Returns these as a tuple of dataloaders.
        """
        # get files by chromosome
        chrom_file_dict = H5DataLoader.organize_h5_files_by_chrom(
            self.h5_files, genome="human")

        # order by number of examples
        ordered_chrom_names, num_examples_per_chrom = H5DataLoader.get_size_ordered_chrom_list(
            chrom_file_dict, include_training_negatives=not regression)

        # set up splits
        splits = setup_train_valid_test(
            chrom_file_dict,
            ordered_chrom_names,
            num_examples_per_chrom,
            kfolds,
            valid_folds=valid_folds,
            test_folds=test_folds,
            regression=regression)

        # pull splits into new dataloaders
        split_dataloaders = [
            H5DataLoader(self.data_dir, data_files=split, fasta=self.fasta)
            for split in splits]

        return split_dataloaders


    def _check_chromosomes(self, chromosomes):
        """check and make sure the list only contains desired
        chromosomes
        """
        consistent = True
        for data_file in self.data_files:
            with h5py.File(data_file, "r") as hf:
                file_chroms = list(hf["/"].attrs["chromosomes"])
            intersect_total = len(set(file_chroms).intersection(set(chromosomes)))
            if intersect_total < len(file_chroms):
                consistent = False
                break

        return consistent

    
    def filter_for_chromosomes(self, chromosomes):
        """given a list of chromosomes, reduce data files to those
        that are in the chromosome list
        """
        filtered_files = []
        for data_file in self.data_files:
            with h5py.File(data_file, "r") as hf:
                file_chroms = list(hf["/"].attrs["chromosomes"])
            intersect_total = len(set(file_chroms).intersection(set(chromosomes)))
            if intersect_total == len(file_chroms):
                filtered_files.append(data_file)
        new_dataloader = H5DataLoader(
            self.data_dir, data_files=filtered_files, fasta=self.fasta)
                
        return new_dataloader
    

    def setup_positives_only_dataloader(self):
        """only work with positives
        """
        # TODO(dk) - use an attributes tag
        positives_files = [
            h5_file for h5_file in self.data_files
            if "negative" not in h5_file]
        new_dataloader = H5DataLoader(
            self.data_dir, data_files=positives_files, fasta=self.fasta)
        
        return new_dataloader


    def remove_genomewide_negatives(self):
        """dataloader without genomewide negatives
        """
        # TODO(dk) use an attributes tag
        new_files = [
            h5_file for h5_file in self.data_files
            if "genomewide-negatives" not in h5_file]
        new_dataloader = H5DataLoader(
            self.data_dir, data_files=new_files, fasta=self.fasta)
        
        return new_dataloader


    def remove_training_negatives(self):
        """dataloader without genomewide negatives
        """
        # TODO(dk) use an attributes tag
        new_files = [
            h5_file for h5_file in self.data_files
            if "training-negatives" not in h5_file]
        new_dataloader = H5DataLoader(
            self.data_dir, data_files=new_files, fasta=self.fasta)
        
        return new_dataloader
    
    
    def get_classification_metrics(
            self,
            file_prefix,
            targets=[(DataKeys.LABELS, [])]):
        """Get class imbalances, num of outputs, etc
        """
        # for each label key set
        for key, indices in label_keys:
            # get num tasks
            with h5py.File(self.h5_files[0], "r") as hf:
                if len(indices) == 0:
                    num_tasks = hf[key].shape[1]
                else:
                    num_tasks = len(indices)
                task_files = hf[key].attrs["filenames"]

            # go through the files and sum up
            total_examples = 0
            positives = np.zeros((num_tasks))
            for i in range(len(self.h5_files)):
                with h5py.File(self.h5_files[i], "r") as hf:
                    total_examples += hf[key].shape[0]
                    if len(indices) == 0:
                        positives += np.sum(hf[key][:] > 0, axis=0)
                    else:
                        positives += np.sum(hf[key][:] > 0, axis=0)

            # and save out to a file
            dataset_metrics = pd.DataFrame(
                {"positives": positives.astype(int),
                 "positives/total": positives/total_examples,
                 "positives/negatives": positives/(total_examples - positives),
                 "negatives": (total_examples - positives).astype(int),
                 "total": total_examples},
                index=task_files)
            dataset_metrics.to_csv(
                "{}.clf.metrics.{}.txt".format(file_prefix, key),
                sep="\t")
        
        return None


    def get_regression_metrics(
            self,
            file_prefix,
            targets=[(DataKeys.LABELS, [])]):
        """get number of examples, mean, std
        note that this takes a sample (ie, 1 file) to see distribution
        """
        # TODO get filenames in
        # for each label key set
        for key in label_keys:
            # get num tasks
            with h5py.File(self.h5_files[0], "r") as hf:
                num_tasks = hf[key].shape[1]
                means = np.mean(hf[key][:], axis=0)
                stdevs = np.std(hf[key][:], axis=0)

            # go through the files and sum up
            total_examples = 0
            for i in range(len(self.h5_files)):
                with h5py.File(self.h5_files[i], "r") as hf:
                    total_examples += hf[key].shape[0]

            # and save out to a file
            dataset_metrics = pd.DataFrame(
                {"mean": means,
                 "stdev": stdevs,
                 "total": total_examples})
            dataset_metrics.to_csv(
                "{}.regr.metrics.{}.txt".format(file_prefix, key),
                sep="\t")
        
        return None

        
    def describe(self):
        """output a dictionary describing the basics to be able to
        reinstantiate a dataset
        """
        dataset = {
            "data_dir": self.data_dir,
            "data_files": sorted([
                os.path.basename(data_file)
                for data_file in self.data_files]),
            "fasta": self.fasta}
            
        return dataset


    def load_dataset(self, key):
        """extract dataset across files and return as one numpy array, plus any attributes?
        """
        data = []
        for data_file in self.data_files:
            with h5py.File(data_file, "r") as hf:
                file_data = hf[key][:]
                data.append(file_data)

        data = np.concatenate(data, axis=0)
                
        return data


    def load_datasets(self, keys):
        """extract dataset across files and return as dict of arrays
        """
        data = {}
        for key in keys:
            data[key] = self.load_dataset(key)
        
        return data

    
    
class ArrayDataLoader(DataLoader):
    """build a dataloader from numpy arrays"""

    def __init__(self, feed_dict, array_names, array_types, **kwargs):
        """keep names and shapes
        """
        self.feed_dict = feed_dict
        self.array_names = array_names
        self.array_types = array_types


        
    def build_generator(
            self,
            batch_size=256,
            task_indices=[],
            keys=[],
            skip_keys=[],
            targets=[([(DataKeys.LABELS, [])], {"reduce_type": "none"})],
            target_indices=[],
            examples_subset=[],
            seq_len=1000,
            lock=threading.Lock(),
            shuffle=True):
        """build generator
        """
        
        return None


    
class VariantDataLoader(DataLoader):
    """build a dataloader that starts from a vcf file"""

    def __init__(self, vcf_file, ref_fasta, alt_fasta):
        self.vcf_file = vcf_file
        self.data_files = [vcf_file]
        self.ref_fasta = ref_fasta
        self.alt_fasta = alt_fasta

    def setup_positives_only_dataloader(self):
        """don't adjust positives
        """
        return self

    def get_chromosomes(self):
        """get chroms
        """
        return ["NA"]
    
    @staticmethod
    def setup_strided_positions(
            chrom,
            pos,
            snp_id,
            snp_info,
            num_positions,
            active_sequence_length=120,
            full_sequence_length=1000):
        """
        """
        # start from the middle and adjust left and right
        seq_metadata = []
        ids = []
        snp_metadata = []
        active_extend_len = int(active_sequence_length / 2.)
        full_extend_len = int(full_sequence_length / 2.)
        
        # calculate stride and determine offsets
        stride = int(active_sequence_length / float(num_positions))
        center_point = int(full_sequence_length / 2.)
        #offset = center_point - (num_positions / 2) * stride
        strided_positions = np.arange(0, stride*num_positions, stride)

        # example positions are distributed around the center point
        example_positions = (strided_positions - np.median(strided_positions) + center_point).astype(int)

        # genomic positions need to add the position and remove center point
        genomic_positions = example_positions - center_point + pos
        
        for position_idx in range(len(example_positions)):
            genomic_position = genomic_positions[position_idx]
            # VCF is 1 based and bedtools is 0 based for start!
            region = "{}:{}-{}".format(chrom, pos-1, pos)
            active = "{}:{}-{}".format(
                chrom, genomic_position-active_extend_len, genomic_position+active_extend_len)
            features = "{}:{}-{}".format(
                chrom, genomic_position-full_extend_len, genomic_position+full_extend_len)
            metadata = "region={};active={};features={}".format(
                region, active, features)
            seq_metadata.append(metadata)
            ids.append(snp_id)
            snp_metadata.append(snp_info)

        # adjust dims/make array
        seq_metadata = np.expand_dims(np.array(seq_metadata), axis=-1)
        ids = np.array(ids)
        snp_metadata = np.array(snp_metadata)
        
        return seq_metadata, example_positions, ids, snp_metadata
    
        
    def build_generator(
            self,
            batch_size=16,
            task_indices=[],
            keys=[],
            skip_keys=[],
            targets=[([(DataKeys.LABELS, [])], {"reduce_type": "none"})],
            target_indices=[],
            examples_subset=[],
            seq_len=1000,
            strided_reps=1,
            lock=threading.Lock(),
            shuffle=True):
        """build generator function
        """
        # tensors: example_metadata, features
        dtypes_dict = {
            DataKeys.SEQ_METADATA: tf.string,
            DataKeys.VARIANT_IDX: tf.int64,
            DataKeys.VARIANT_ID: tf.string,
            DataKeys.VARIANT_INFO: tf.string,
            DataKeys.FEATURES: tf.uint8}
        shapes_dict = {
            DataKeys.SEQ_METADATA: [1],
            DataKeys.VARIANT_IDX: [],
            DataKeys.VARIANT_ID: [],
            DataKeys.VARIANT_INFO: [],
            DataKeys.FEATURES: [seq_len]}
        
        class Generator(object):

            def __init__(self, ref_fasta, alt_fasta, batch_size):
                self.ref_fasta = ref_fasta
                self.alt_fasta = alt_fasta
                self.batch_size = batch_size
            
            def __call__(self, vcf_file, yield_single_examples=True):
                """call generator
                """
                # build converters
                ref_converter = GenomicIntervalConverter(lock, self.ref_fasta, 1)
                alt_converter = GenomicIntervalConverter(lock, self.alt_fasta, 1)

                # determine padding amount at end of file
                num_variants = 0
                with open(vcf_file, "r") as fp:
                    for line in fp:
                        if line.startswith("#"):
                            continue
                        num_variants += 1
                padding_num = batch_size - (num_variants % batch_size)
                logging.info("will pad variants with {}".format(padding_num))
                total_generator_calls = num_variants + padding_num
                    
                # and then go through each
                generator_calls = 0
                fake_data = False
                with open(vcf_file, "r") as fp:
                    while generator_calls < total_generator_calls:
                        line = fp.readline()
                        
                        if line.startswith("#"):
                            continue

                        if len(line) == 0:
                            # empty, end of file - use previous line
                            fake_data = True
                            line = prev_line
                            
                        fields = line.strip().split("\t")
                        chrom = "chr{}".format(fields[0])
                        pos = int(fields[1])
                        snp_id = fields[2]
                        snp_info = fields[7]

                        # set up strided reps
                        seq_metadata, positions, ids, snp_metadata = VariantDataLoader.setup_strided_positions(
                            chrom, pos, snp_id, snp_info, strided_reps,
                            full_sequence_length=seq_len) # {strided_rep*N}

                        # adjust if fake data
                        if fake_data:
                            seq_metadata = ["features=chr1:0-1000"
                                            for i in range(seq_metadata.shape[0])]
                            seq_metadata = np.expand_dims(np.array(seq_metadata), axis=1)
                        
                        # get sequence
                        ref_features = ref_converter.convert(seq_metadata)
                        alt_features = alt_converter.convert(seq_metadata)
                        
                        # combine them
                        features = np.stack([ref_features, alt_features], axis=1) # {strided_rep*N, 2}

                        # put into array
                        slice_array = {
                            DataKeys.SEQ_METADATA: seq_metadata,
                            DataKeys.VARIANT_IDX: positions,
                            DataKeys.VARIANT_ID: ids,
                            DataKeys.VARIANT_INFO: snp_metadata,
                            DataKeys.FEATURES: features}

                        # and adjust to interleave - {strided_rep*N*2}
                        for key in slice_array.keys():
                            if key != DataKeys.FEATURES:
                                # adjust for interleaving
                                slice_array[key] = np.stack([slice_array[key]]*2, axis=1)
                            # interleave
                            slice_array[key] = np.reshape(
                                slice_array[key],
                                [-1] + list(slice_array[key].shape[2:]))
                            
                        # yield
                        generator_calls += 1
                        prev_line = line
                        yield (slice_array, 1.)
                
        # instantiate
        generator = Generator(self.ref_fasta, self.alt_fasta, batch_size)
        
        return generator, dtypes_dict, shapes_dict
        
    
class BedDataLoader(DataLoader):
    """build a dataloader starting from a bed file"""

    def __init__(self, bed_file, fasta_file):
        self.bed_file = bed_file
        self.fasta_file = fasta_file
        # TODO set up option to bin or not?
        assert self.bed_file.endswith(".gz")
        self.num_regions = self.get_num_regions()

        
    def get_num_regions(self):
        """count up how many regions there are
        """
        num_regions = 0
        with gzip.open(bed_file, "r") as fp:
            for line in fp:
                num_regions += 1
        
        return num_regions
        
        
    def build_generator(
            self,
            batch_size=256,
            task_indices=[],
            keys=[],
            skip_keys=[],
            targets=[([(DataKeys.LABELS, [])], {"reduce_type": "none"})],
            target_indices=[],
            examples_subset=[],
            seq_len=1000,
            lock=threading.Lock(),
            shuffle=True):
        """build generator function
        """
        
        class Generator(object):

            def __init__(self, fasta, batch_size):
                self.fasta = fasta
                self.batch_size = batch_size

                
            def __call__(self, bed_file, yield_single_examples=True):
                """run the generator"""

                batch_size = self.batch_size
                fasta = self.fasta
                
                if len(examples_subset) != 0:
                    batch_size = 1

                # set up interval to sequence converter
                converter = GenomicIntervalConverter(lock, fasta, batch_size)
                
                # open bed file
                with gzip.open(bed_file, "r") as fp:
                    
                    # set up batch id total and get batches
                    max_batches = int(math.ceil(h5_handle[test_key].shape[0]/float(batch_size)))
                    batch_ids = list(range(max_batches))

                    # shuffle batches as needed
                    if shuffle:
                        random.Random(42).shuffle(batch_ids)
                            
                    # logging
                    logging.debug("loading {0} with batch size {1} to get batches {2}".format(
                        os.path.basename(h5_file), batch_size, max_batches))
                    
                    # and go through batches
                    try:
                        assert len(clean_keys_to_load) != 0
                        
                        for batch_id in batch_ids:
                            batch_start = batch_id*batch_size
                            slice_array = H5DataLoader.h5_to_slices(
                                h5_handle,
                                batch_start,
                                batch_size,
                                keys_to_load=clean_keys_to_load,
                                targets=targets,
                                target_indices=target_indices)

                            # onehot encode on the fly
                            # TODO keep the string sequence
                            slice_array[DataKeys.FEATURES] = converter.convert(
                                slice_array["example_metadata"])
                            
                            # yield
                            if yield_single_examples: # NOTE: this is the most limiting step
                                for i in range(batch_size):
                                    yield ({
                                        key: value[i]
                                        for key, value in six.iteritems(slice_array)
                                    }, 1.)
                            else:
                                yield (slice_array, 1.)
                            
                    except ValueError as value_error:
                        logging.debug(value_error)
                        logging.info("Stopping {}".format(h5_file))
                        raise StopIteration

                    finally:
                        converter.close()
                        print("finished {}".format(h5_file))
                            
        # instantiate
        generator = Generator(self.fasta, batch_size)
        
        return generator, dtypes_dict, shapes_dict
    

    def build_raw_dataflow(
            self,
            batch_size,
            task_indices=[],
            features_key="features",
            label_keys=["labels"],
            metadata_key="example_metadata",
            shuffle=True,
            shuffle_seed=1337,
            num_epochs=1):
        """
        """
        # set up batch id producer (the value itself is unused - drives queue)
        num_regions = 0
        with gzip.open(self.bed_file, "r") as fp:
            for line in fp:
                num_regions += 1
        batch_id_queue = tf.train.range_input_producer(
            num_regions, shuffle=False, seed=0, num_epochs=num_epochs)
        batch_id = batch_id_queue.dequeue()
        logging.info("num_regions: {}".format(num_regions))

        # iterator: produces sequence and example metadata
        iterator = bed_to_sequence_iterator(
            self.bed_file,
            self.fasta_file,
            batch_size=batch_size)
        def example_generator(batch_id):
            return next(iterator)
        tensor_dtypes = [tf.string, tf.uint8]
        keys = ["example_metadata", "features"]
        
        # py_func
        inputs = tf.py_func(
            func=example_generator,
            inp=[batch_id],
            Tout=tensor_dtypes,
            stateful=False, name='py_func_batch_id_to_examples')

        # set shapes
        for i in range(len(inputs)):
            if "metadata" in keys[i]:
                inputs[i].set_shape([1, 1])
            else:
                inputs[i].set_shape([1, 1000])

        # set as dict
        inputs = dict(list(zip(keys, inputs)))
        
        # batch
        inputs = tf.train.batch(
            inputs,
            batch_size,
            capacity=1000,
            enqueue_many=True,
            name='batcher')

        # convert to onehot
        inputs["features"] = tf.map_fn(
            DataLoader.encode_onehot_sequence,
            inputs["features"],
            dtype=tf.float32)
        print(inputs["features"])
        
        return inputs

    
def setup_data_loader(args):
    """wrapper function to deal with dataloading 
    from different formats

    If making a custom dataloader, note that
    data must come in through either:
      args.data_dir
      args.data_files
      args.dataset_json
    """
    if args.data_format == "hdf5":
        data_loader = H5DataLoader(
            data_dir=args.data_dir,
            data_files=args.data_files,
            dataset_json=args.dataset_json,
            fasta=args.fasta)
    elif args.data_format == "vcf":
        data_loader = VariantDataLoader(
            vcf_file=args.vcf_file,
            ref_fasta=args.ref_fasta,
            alt_fasta=args.alt_fasta)
    elif args.data_format == "bed":
        raise ValueError("implement!")
    else:
        raise ValueError("unrecognized data format!")

    return data_loader
