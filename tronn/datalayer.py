"""Contains functions for I/O to tensorflow graphs
"""

import os
import six
import gzip
import glob
import h5py
import math
import random
import logging
import threading
import subprocess

import abc

import numpy as np
import pandas as pd
import tensorflow as tf

from tronn.preprocess.fasta import GenomicIntervalConverter
from tronn.preprocess.fasta import bed_to_sequence_iterator
from tronn.preprocess.fasta import sequence_string_to_onehot_converter # TODO deprecate
from tronn.preprocess.fasta import batch_string_to_onehot # TODO deprecate

from tronn.nets.filter_nets import filter_by_labels # TODO deprecate
from tronn.nets.filter_nets import filter_singleton_labels # TODO deprecate

from tronn.util.h5_utils import get_absolute_label_indices
from tronn.util.utils import DataKeys


class DataLoader(object):
    """build the base level dataloader"""

    __metaclass__ = abc.ABCMeta

    #@abc.abstractmethod
    #def __init__(self):
    #    """set up dataloader
    #    """
    #    pass
    
    @abc.abstractmethod
    def build_raw_dataflow(self, batch_size, task_indices=[]):
        """build a raw dataflow from the files to tensors
        """
        raise NotImplementedError, "implement in child class!"


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
        """add in various filters to the dataset
        """
        # build raw dataflow
        dataset = self.build_raw_dataflow(
            batch_size,
            task_indices=label_tasks,
            label_keys=label_keys,
            **kwargs)

        if False:
            # label filter fn
            def filter_by_labels(labels_key, filter_tasks, filter_type="any"):
                def filter_fn(features, labels):
                    mask_array = np.zeros((features[labels_key].get_shape()[0]))
                    mask_array[filter_tasks] = 1
                    results = tf.reduce_sum(
                        tf.multiply(features[labels_key], mask_array))
                    return tf.greater(results, 0)
                return filter_fn

            if len(filter_tasks) != 0:
                # go through the list to subset more finegrained
                # in a hierarchical way
                for key in filter_tasks.keys():
                    print key
                    print filter_tasks[key][0]
                    dataset = dataset.filter(
                        filter_by_labels(key, filter_tasks[key][0]))

            # only use this if you have more than one singleton task
            # singletons filter fn
            def filter_singletons(labels_key, filter_tasks):
                def filter_fn(features, labels):
                    tasks = tf.gather(features[labels_key], filter_tasks)
                    print tasks
                    results = tf.reduce_sum(tasks)
                    return tf.greater(results, 1)
                return filter_fn

            if len(singleton_filter_tasks) > 1:
                dataset = dataset.filter(
                    filter_singletons("labels", singleton_filter_tasks))

        return dataset
    
    
    def build_filtered_dataflow_OLD(
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
            # TODO adjust to only have a queue at the END, so that not wasting a lot of queue space
            # TODO convert these to dataset.filter
            for key in filter_tasks.keys(): # TODO consider an ordered dict
                print key
                print filter_tasks[key][0]
                transform_stack.append((
                    filter_by_labels,
                    {"labels_key": key,
                     "filter_tasks": filter_tasks[key][0],
                     "name": "label_filter_{}".format(key)}))
        if len(singleton_filter_tasks) != 0:
            transform_stack.append((
                filter_singleton_labels,
                {"labels_key": "labels",
                 "filter_tasks": singleton_filter_tasks,
                 "name": "singleton_label_filter"}))
            
        # build all together
        with tf.variable_scope("dataloader"):
            # build raw dataflow
            inputs = self.build_raw_dataflow(
                batch_size,
                task_indices=label_tasks,
                label_keys=label_keys,
                **kwargs)

            if True:
                return inputs
            
            # build transform stack
            input_params = {}
            input_params.update({"batch_size": batch_size})
            for transform_fn, params in transform_stack:
                input_params.update(params)
                print transform_fn
                inputs, input_params = transform_fn(inputs, input_params)
            
            # adjust outputs as needed
            if len(keep_keys) > 0:
                new_inputs = {}
                for key in keep_keys:
                    new_inputs[key] = inputs[key]
                inputs = new_inputs
                
        return inputs
    
    
    def build_input_fn(self, batch_size, shuffle=True, **kwargs):
        """build the dataflow function. will be called later in graph
        """
        def dataflow_fn():
            """dataflow function. must have no args.
            """
            # build dataset and filter as needed
            dataset = self.build_filtered_dataflow(batch_size, **kwargs)
            
            # shuffle
            if shuffle:
                min_after_dequeue = 1000 #10000
                capacity = min_after_dequeue + (len(self.h5_files)+10) * (batch_size/16)
                dataset = dataset.shuffle(capacity) # TODO adjust this as needed
            
            # batch
            #dataset = dataset.batch(batch_size)
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(batch_size))

            # map (ie convert sequence string to onehot)
            def make_onehot_sequence(features, labels):
                features[DataKeys.FEATURES] = tf.map_fn(
                    DataLoader.encode_onehot_sequence,
                    features[DataKeys.FEATURES],
                    dtype=tf.float32)
                return features, labels
            
            if False:
                def make_onehot_sequence(features, labels):
                    features[DataKeys.FEATURES] = DataLoader.encode_onehot_sequence(
                        features[DataKeys.FEATURES])
                    return features, labels
        
            dataset = dataset.map(
                make_onehot_sequence,
                num_parallel_calls=28)

            # batch
            #dataset = dataset.batch(batch_size)
            #dataset = dataset.apply(
            #    tf.contrib.data.batch_and_drop_remainder(batch_size))

            # prefetch
            dataset = dataset.prefetch(5)
            
            return dataset
            
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
            quit()
            #inputs, _ = pad_data(
            #    inputs,
            #    {"ignore": ["features"], "batch_size": batch_size})

            return inputs, None
        
        return dataflow_fn

    @staticmethod
    def encode_onehot_sequence(sequence):
        """adjust to onehot, given indices tensor
        """
        sequence = tf.one_hot(sequence, 5, axis=-1) # {seq_len, 5}
        sequence = tf.expand_dims(sequence, axis=0) # {1, seq_len, 5}
        sequence = tf.gather(sequence, [0, 1, 2, 3], axis=2) # {1, seq_len, 4]
        
        return sequence
    

class H5DataLoader(DataLoader):
    """build a dataloader from h5"""

    def __init__(self, h5_files, fasta=None, **kwargs):
        """initialize with data files
        """
        super(H5DataLoader, self).__init__(**kwargs)
        self.h5_files = h5_files
        self.fasta = fasta
        self.num_examples = self.get_num_examples(h5_files)
        self.num_examples_per_file = self.get_num_examples_per_file(h5_files)

        
    @staticmethod
    def setup_h5_files(data_dir):
        """helper function to go into h5 directory and organize h5 files
        """
        chroms = ["chr{}".format(i) for i in xrange(1,23)] + ["chrX", "chrY"]

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

    
    @staticmethod
    def get_num_tasks(h5_files, label_keys, label_keys_dict):
        """get number of labels
        """
        num_tasks =  len(get_absolute_label_indices(label_keys, label_keys_dict, h5_files[0]))
        logging.info("Found {} tasks across label set(s)".format(num_tasks))
        return num_tasks

    
    def get_classification_metrics(self, file_prefix, label_keys=["labels"]):
        """Get class imbalances, num of outputs, etc
        """
        # for each label key set
        for key in label_keys:
            # get num tasks
            with h5py.File(self.h5_files[0], "r") as hf:
                num_tasks = hf[key].shape[1]
                task_files = hf[key].attrs["filenames"]

            # go through the files and sum up
            total_examples = 0
            positives = np.zeros((num_tasks))
            for i in xrange(len(self.h5_files)):
                with h5py.File(self.h5_files[i], "r") as hf:
                    total_examples += hf[key].shape[0]
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


    def get_regression_metrics(self, file_prefix, label_keys=["labels"]):
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
            for i in xrange(len(self.h5_files)):
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

    
    @staticmethod
    def h5_to_slices(
            h5_handle,
            start_idx,
            batch_size,
            task_indices=[],
            keys=None,
            features_key=DataKeys.FEATURES,
            label_keys=["labels"],
            skip_keys=["labels", "features", "label_metadata"]):
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
                        ["features=chr1:0-1000" for i in xrange(batch_padding_num)]).reshape(
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
            fasta=None,
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

        # check skip keys
        num_examples = h5_handle[DataKeys.SEQ_METADATA].shape[0]
        for key in h5_handle.keys():
            # check if scalar
            if h5_handle[key].shape == 0:
                skip_keys.append(key)
            # check if different shape
            if h5_handle[key].shape[0] != num_examples:
                skip_keys.append(key)

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

        # set up batch count tracker
        batch_idx_producer = tf.train.range_input_producer(
            max_batches, shuffle=False, seed=0, num_epochs=num_epochs)
        batch_idx = batch_idx_producer.dequeue()
        
        # set up on-the-fly onehot encoder
        assert fasta is not None
        converter_in, converter_out, converter_close_fn = sequence_string_to_onehot_converter(fasta)

        def close_fn(close_val):
            converter_close_fn()
            h5_handle.close()
            return 0
        
        # determine the Tout
        tensor_dtypes = []
        for key in keys:
            if isinstance(h5_handle[key][0], basestring):
                tensor_dtypes.append(tf.string)
            elif h5_handle[key][0].dtype.char == "S":
                tensor_dtypes.append(tf.string)
            else:
                tensor_dtypes.append(tf.float32)

        # labels get set up in the file so need to append to get them out
        keys.append("labels")
        tensor_dtypes.append(tf.float32)

        if DataKeys.FEATURES not in keys:
            keys.append(DataKeys.FEATURES)
            tensor_dtypes.append(tf.uint8)
        else:
            # replace
            tensor_dtypes[keys.index(DataKeys.FEATURES)] = tf.uint8

        # set up tmp numpy array so that it's not re-created for every batch
        # TODO adjust seq length
        onehot_batch_array = np.zeros((batch_size, 1000), dtype=np.uint8)
        
        # function to get examples based on batch_id (for py_func)
        def batch_id_to_examples(batch_id, batch_idx):
            """Given a batch ID, get the features, labels, and metadata
            This is an important wrapper to be able to use TensorFlow's pyfunc.
            """
            batch_start = batch_id*batch_size

            try:
                slice_array = H5DataLoader.h5_to_slices(
                    h5_handle,
                    batch_start,
                    batch_size,
                    task_indices=task_indices,
                    features_key=features_key,
                    label_keys=label_keys)
            except ValueError:
                # deal with closed h5 file
                raise StopIteration
                
            # onehot encode on the fly
            # TODO keep the string sequence
            slice_array[DataKeys.FEATURES] = batch_string_to_onehot(
                slice_array["example_metadata"],
                converter_in,
                converter_out,
                onehot_batch_array)
            
            slice_list = []
            for key in keys:
                slice_list.append(slice_array[key])
                
            return slice_list
            
        # py_func
        inputs = tf.py_func(
            func=batch_id_to_examples,
            inp=[batch_id, batch_idx],
            Tout=tensor_dtypes,
            stateful=False, name='py_func_batch_id_to_examples')

        # py func for closing function
        close_op = tf.py_func(
            func=close_fn,
            inp=[0],
            Tout=tf.int64,
            stateful=False)
        tf.get_collection("DATACLEANUP")
        tf.add_to_collection("DATACLEANUP", close_op)
        
        # set shapes
        for i in xrange(len(inputs)):
            if "labels" == keys[i]:
                inputs[i].set_shape([batch_size, len(task_indices)])
            elif "metadata" in keys[i]:
                inputs[i].set_shape([batch_size, 1])
            elif "features" in keys[i]:
                inputs[i].set_shape([batch_size, 1000]) # TODO fix
            else:
                inputs[i].set_shape([batch_size] + list(h5_handle[keys[i]].shape[1:]))

        # make dict
        inputs = dict(zip(keys, inputs))

        # onehot encode the batch
        inputs[DataKeys.FEATURES] = tf.map_fn(
            DataLoader.encode_onehot_sequence,
            inputs["features"],
            dtype=tf.float32)
        
        return inputs

    
    @staticmethod
    def build_h5_generator(
            batch_size,
            task_indices=[],
            fasta=None,
            keys=None,
            skip_keys=["label_metadata"],
            features_key="features", # TODO do we need this?
            label_keys=["labels"],
            metadata_key="example_metadata",
            shuffle=True):
        """make a generator, pulls batches into numpy array from h5
        only goes through data ONCE
        """
        assert fasta is not None
        lock = threading.Lock()
        
        # make a convenience class (this is for tf.data.Dataset purposes)
        class Generator(object):
                
            def __call__(self, h5_file):
                """run the generator"""

                # set up interval to sequence converter
                converter = GenomicIntervalConverter(lock, fasta, batch_size)

                # open h5 file
                with h5py.File(h5_file, "r") as h5_handle:
                    test_key = h5_handle.keys()[0]
                    
                    # set up batch id total
                    max_batches = int(math.ceil(h5_handle[test_key].shape[0]/float(batch_size)))
                    logging.debug("loading {0} with max batches {1}".format(
                        os.path.basename(h5_file), max_batches))

                    # set up shuffled batches
                    batch_ids = range(max_batches)
                    if shuffle:
                        random.shuffle(batch_ids)
                    
                    # and go through batches
                    try:
                        for batch_id in batch_ids:
                            batch_start = batch_id*batch_size
                            slice_array = H5DataLoader.h5_to_slices(
                                h5_handle,
                                batch_start,
                                batch_size,
                                keys=keys,
                                task_indices=task_indices,
                                features_key=features_key,
                                label_keys=label_keys)

                            # onehot encode on the fly
                            # TODO keep the string sequence
                            slice_array[DataKeys.FEATURES] = converter.convert(
                                slice_array["example_metadata"])
                            
                            # yield
                            for i in xrange(batch_size):
                                yield ({
                                    key: value[i]
                                    for key, value in six.iteritems(slice_array)
                                }, 1.)
                            
                    except ValueError:
                        logging.info("Stopping {}".format(h5_file))
                        raise StopIteration

                    finally:
                        converter.close()
                        print "finished {}".format(h5_file)
                            
        # instantiate
        generator = Generator()

        return generator


    def get_tensor_dtypes(
            self,
            batch_size,
            task_indices,
            skip_keys=["label_metadata"]):
        """access an h5 file to get the dtypes
        """
        test_h5 = self.h5_files[0]

        with h5py.File(test_h5, "r") as h5_handle:
            # check skip keys
            num_examples = h5_handle[DataKeys.SEQ_METADATA].shape[0]
            for key in h5_handle.keys():
                # check if scalar
                if h5_handle[key].shape == 0:
                    skip_keys.append(key)
                    # check if different shape
                if h5_handle[key].shape[0] != num_examples:
                    skip_keys.append(key)

            # keys
            keys = sorted(h5_handle.keys())
            keys = [key for key in keys if key not in skip_keys]
            
            # determine the Tout
            tensor_dtypes = []
            dtype_dict = {}
            shape_dict = {}
            for key in keys:
                if isinstance(h5_handle[key][0], basestring):
                    tensor_dtypes.append(tf.string)
                    dtype_dict[key] = tf.string
                    shape_dict[key] = [1]
                elif h5_handle[key][0].dtype.char == "S":
                    tensor_dtypes.append(tf.string)
                    dtype_dict[key] = tf.string
                    shape_dict[key] = h5_handle[key].shape[1:]
                else:
                    tensor_dtypes.append(tf.float32)
                    dtype_dict[key] = tf.float32
                    shape_dict[key] = h5_handle[key].shape[1:]

            # labels get set up in the file so need to append to get them out
            keys.append("labels")
            tensor_dtypes.append(tf.float32)
            dtype_dict["labels"] = tf.float32
            shape_dict["labels"] = len(task_indices)
            
            if DataKeys.FEATURES not in keys:
                keys.append(DataKeys.FEATURES)
                tensor_dtypes.append(tf.uint8)
                dtype_dict[DataKeys.FEATURES] = tf.uint8
            else:
                # replace
                tensor_dtypes[keys.index(DataKeys.FEATURES)] = tf.uint8
                dtype_dict[DataKeys.FEATURES] = tf.uint8
            shape_dict[DataKeys.FEATURES] = 1000

        return dtype_dict, keys, shape_dict
    

    def build_raw_dataflow(
            self,
            batch_size,
            task_indices=[],
            features_key=DataKeys.FEATURES,
            label_keys=["labels"],
            metadata_key="example_metadata",
            shuffle=True,
            shuffle_seed=1337):
        """build dataflow from files to tf dataset
        """
        logging.info(
            "loading data for task indices {0} "
            "from {1} hdf5_files: "
            "{2} examples".format(
                task_indices, len(self.h5_files), self.num_examples))
        
        # adjust task indices as needed
        if len(task_indices) == 0:
            num_labels = 0
            with h5py.File(self.h5_files[0], "r") as hf:
                for key in label_keys:
                    num_labels += hf[key].shape[1]
            task_indices = range(num_labels)

        # set up dataset run params
        dtypes, keys, shape_dict = self.get_tensor_dtypes(
            batch_size, task_indices)

        # set up generator
        generator = H5DataLoader.build_h5_generator(
            batch_size,
            task_indices=task_indices,
            keys=keys,
            fasta=self.fasta,
            label_keys=label_keys)
        
        # set up dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.h5_files).apply(
            tf.contrib.data.parallel_interleave(
                lambda h5_file: tf.data.Dataset.from_generator(
                    generator,
                    (dtypes, tf.int32),
                    output_shapes=(shape_dict,()),
                    args=(h5_file,)),
                cycle_length=len(self.h5_files),
                sloppy=True))
        
        #if False:
            #iterator = dataset.make_one_shot_iterator()
            #inputs = iterator.get_next()
            #tf.get_collection("DATASETUP")
            #tf.add_to_collection("DATASETUP", iterator.initializer)

        return dataset
    
            
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


    def build_raw_dataflow_old(
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
                #H5DataLoader.h5_to_tf_dataset(
                    h5_file, batch_size, task_indices,
                    fasta=self.fasta, label_keys=label_keys, shuffle=True)
                for h5_file in self.h5_files]
            if len(example_slices_list) > 1:
                min_after_dequeue = 10000
            else:
                min_after_dequeue = batch_size * 3
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
        # TODO - bin regions here?
        

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
            return iterator.next()
        tensor_dtypes = [tf.string, tf.uint8]
        keys = ["example_metadata", "features"]
        
        # py_func
        inputs = tf.py_func(
            func=example_generator,
            inp=[batch_id],
            Tout=tensor_dtypes,
            stateful=False, name='py_func_batch_id_to_examples')

        # set shapes
        for i in xrange(len(inputs)):
            if "metadata" in keys[i]:
                inputs[i].set_shape([1, 1])
            else:
                inputs[i].set_shape([1, 1000])

        # set as dict
        inputs = dict(zip(keys, inputs))
        
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
        print inputs["features"]
        
        return inputs
