"""Description: contains code to generate per-base-pair importance scores
"""

import h5py
import logging
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from tronn.util.tf_utils import setup_tensorflow_session
from tronn.util.tf_utils import close_tensorflow_session
from tronn.visualization import plot_weights
from tronn.interpretation.regions import RegionImportanceTracker

from tronn.outlayer import ExampleGenerator



@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    """Replaces ReLUs with guided ReLUs in a tensorflow graph. Use to 
    allow guided backpropagation in interpretation mode. Generally only 
    turn on for a trained model.
    
    Args:
      op: the op to replace the gradient
      grad: the gradient value
    
    Returns:
      tensorflow operation that removes negative gradients
    """
    return tf.where(0. < grad,
                    gen_nn_ops._relu_grad(grad, op.outputs[0]),
                    tf.zeros(grad.get_shape()))


def layerwise_relevance_propagation(tensor, features):
    """Layer-wise Relevance Propagation (Batch et al), implemented
    as input * gradient (equivalence is demonstrated in deepLIFT paper,
    Shrikumar et al). Generally center the tensor on the logits.
    
    Args:
      tensor: the tensor from which to propagate gradients backwards
      features: the input tensor on which you want the importance scores
    
    Returns:
      Input tensor weighted by gradient backpropagation.
    """
    [feature_grad] = tf.gradients(tensor, [features])
    importances = tf.multiply(features, feature_grad, 'input_mul_grad')
    importances_squeezed = tf.transpose(tf.squeeze(importances), perm=[0, 2, 1])
    
    return importances_squeezed


def region_generator_v2(sess, tronn_graph, stop_idx):
    """Uses new regions class
    """
    # TODO(dk) eventually rewrite as a function of TronnGraph and also
    # make it work with variable keys, so that it's more extensible
    
    region_tracker = RegionImportanceTracker(tronn_graph.importances,
                                             tronn_graph.labels,
                                             tronn_graph.probs)
    region_idx = 0
    
    while region_idx < stop_idx:

        # run session to get batched results
        regions_array, importances_arrays, labels_array, probs_array = sess.run([
            tronn_graph.metadata,
            tronn_graph.importances,
            tronn_graph.labels,
            tronn_graph.probs])

        # Go through each example in batch
        for i in range(regions_array.shape[0]):

            # ignore univ region negs
            if np.sum(labels_array[i,:]) == 0:
                continue

            # extract example data
            example_region = regions_array[i,0]
            example_labels = labels_array[i,:]
            example_probs = probs_array[i,:]

            # extract example importances
            example_importances = {}
            for importance_key in importances_arrays.keys():
                example_importances[importance_key] = np.squeeze(
                    importances_arrays[importance_key][i,:,:,:]).transpose(1, 0)
                
            # check if overlap
            if region_tracker.check_downstream_overlap(example_region):
                # if so, merge
                region_tracker.merge(
                    example_region,
                    example_importances,
                    example_labels,
                    example_probs)
            else:
                # yield out the current sequence info if NOT the original fake init.
                if region_tracker.chrom is not None:
                    region_name, importances, labels, probs = region_tracker.get_region()
                    yield importances, region_name, region_idx, labels
                    region_idx += 1
                    if region_idx == stop_idx:
                        break
                    
                # reset with new info
                region_tracker.reset(
                    example_region,
                    example_importances,
                    example_labels,
                    example_probs)
    
    return


def extract_importances_old(
        tronn_graph,
        model_checkpoint,
        out_file,
        sample_size,
        method="guided_backprop",
        width=4096,
        pos_only=False):
    """Set up a graph and then run importance score extractor
    and save out to file.

    Args:
      tronn_graph: a TronnNeuralNetGraph instance
      model_dir: directory with trained model
      out_file: hdf5 file to store importances
      method: importance method to use
      sample_size: number of regions to run
      pos_only: only keep positive sequences

    Returns:
      None
    """
    with tf.Graph().as_default() as g:

        # build graph
        if method == "guided_backprop":
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                importances = tronn_graph.build_inference_graph()
        elif method == "simple_gradients":
            importances = tronn_graph.build_inference_graph()

        # set up session
        sess, coord, threads = setup_tensorflow_session()

        # restore
        #checkpoint_path = tf.train.latest_checkpoint(model_dir)
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint)

        # set up hdf5 file to store outputs
        with h5py.File(out_file, 'w') as hf:
            
            # set up datasets
            importances_datasets = {}
            for importance_key in importances.keys():
                importances_datasets[importance_key] = hf.create_dataset(importance_key, [sample_size, 4, width])
            labels_hf = hf.create_dataset('labels', [sample_size, tronn_graph.labels.get_shape()[1]])
            regions_hf = hf.create_dataset('regions', [sample_size, 1], dtype='S100')

            # run the region generator
            #for sequence, name, idx, labels_np in region_generator(
            #        sess, importances, tronn_graph.probs, tronn_graph.labels, tronn_graph.metadata, sample_size):

            for sequence, name, idx, labels_np in region_generator_v2(
                    sess, tronn_graph, sample_size):
            
                if idx % 1000 == 0:
                    print idx

                # For importances, pad/trim to make fixed length and store in hdf5
                for importance_key in importances.keys():
                    if sequence[importance_key].shape[1] < width:
                        zero_array = np.zeros((4, width - sequence[importance_key].shape[1]))
                        padded_sequence = np.concatenate((sequence[importance_key], zero_array), axis=1)
                    else:
                        trim_len = (sequence[importance_key].shape[1] - width) / 2
                        padded_sequence = sequence[importance_key][:,trim_len:width+trim_len]
                    importances_datasets[importance_key][idx,:,:] = padded_sequence

                # For other regions save also
                regions_hf[idx,] = name
                labels_hf[idx,] = labels_np
                # TODO(dk) save out predictions too

        close_tensorflow_session(coord, threads)

    return None


def extract_importances(
        tronn_graph,
        model_checkpoint,
        out_file,
        sample_size,
        method="guided_backprop",
        width=4096,
        pos_only=False,
        h5_batch_size=128):
    """Set up a graph and then run importance score extractor
    and save out to file.

    Args:
      tronn_graph: a TronnNeuralNetGraph instance
      model_dir: directory with trained model
      out_file: hdf5 file to store importances
      method: importance method to use
      sample_size: number of regions to run
      pos_only: only keep positive sequences

    Returns:
      None
    """
    with tf.Graph().as_default() as g:

        # build graph
        if method == "guided_backprop":
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                importances = tronn_graph.build_inference_graph()
        elif method == "simple_gradients":
            importances = tronn_graph.build_inference_graph()

        # set up session
        sess, coord, threads = setup_tensorflow_session()

        # restore
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint)

        # set up hdf5 file to store outputs
        with h5py.File(out_file, 'w') as hf:

            # set up datasets (use names to set up shapes?)
            for key in importances.keys():
                dataset_shape = [sample_size] + [int(i) for i in importances[key].get_shape()][1:]
                if "feature_metadata" in key:
                    hf.create_dataset(key, dataset_shape, maxshape=dataset_shape, dtype="S100")
                else:
                    hf.create_dataset(key, dataset_shape, maxshape=dataset_shape)
                    
            # set up outlayer
            example_generator = ExampleGenerator(
                sess,
                importances,
                64, # Fix this later
                reconstruct_regions=False,
                keep_negatives=False,
                filter_by_prediction=True,
                filter_tasks=tronn_graph.importances_tasks)

            try:
                total_examples = 0
                h5_batch_start = 0
                h5_batch_end = h5_batch_start + h5_batch_size
                while not coord.should_stop():

                    # set up batch numpy arrays
                    batched_region_arrays = {}
                    for key in importances.keys():
                        dataset_shape = [h5_batch_size] + [int(i) for i in importances[key].get_shape()][1:]
                        if "feature_metadata" in key:
                            batched_region_arrays[key] = np.array(["chrY:0-0" for i in xrange(h5_batch_size)], dtype="S100")
                        else:
                            batched_region_arrays[key] = np.zeros(dataset_shape)
                    
                    # grab a batch of regions
                    for region_idx in xrange(h5_batch_size):
                        region, region_arrays = example_generator.run()
                        total_examples += 1
                        batched_region_arrays["feature_metadata"][region_idx] = region
                        for key in region_arrays.keys():
                            if "importance" in key:
                                batched_region_arrays[key][region_idx,:,:] = region_arrays[key]
                            else:
                                batched_region_arrays[key][region_idx,:] = region_arrays[key]
                    
                    # put into h5 datasets
                    hf["feature_metadata"][h5_batch_start:h5_batch_end] = batched_region_arrays["feature_metadata"].reshape((h5_batch_size, 1))
                    for key in batched_region_arrays.keys():
                        if "importance" in key:
                            hf[key][h5_batch_start:h5_batch_end,:,:] = batched_region_arrays[key]
                        elif "feature_metadata" in key:
                            continue
                        else:
                            hf[key][h5_batch_start:h5_batch_end,:] = batched_region_arrays[key]

                    h5_batch_start = h5_batch_end
                    h5_batch_end = h5_batch_start + h5_batch_size
                    

            except tf.errors.OutOfRangeError:

                # TODO(dk) add in last of the examples
                hf["feature_metadata"][h5_batch_start:h5_batch_end] = batched_region_arrays["feature_metadata"].reshape((h5_batch_size, 1))
                for key in batched_region_arrays.keys():
                    if "importance" in key:
                        hf[key][h5_batch_start:h5_batch_end,:,:] = batched_region_arrays[key]
                    elif "feature_metadata" in key:
                        continue
                    else:
                        hf[key][h5_batch_start:h5_batch_end,:] = batched_region_arrays[key]

                # and then reshape
                metadata_shape = hf["feature_metadata"].shape
                metadata_shape[0] = total_examples
                hf["feature_metadata"].resize(metadata_shape)
                for key in region_arrays.keys():
                    shape = hf[key].shape
                    shape[0] = total_examples
                    hf[key].reshape(shape)

        close_tensorflow_session(coord, threads)

    return None


def call_importance_peaks_v2(
        importance_h5,
        feature_key,
        callpeak_graph,
        out_h5):
    """Calls peaks on importance scores
    
    Currently assumes a normal distribution of scores. Calculates
    mean and std and uses them to get a pval threshold.

    """
    logging.info("Calling importance peaks...")

    # determine some characteristics of data
    with h5py.File(importance_h5, 'r') as hf:
        num_examples = hf[feature_key].shape[0]
        seq_length = hf[feature_key].shape[2]
        num_tasks = hf['labels'].shape[1]

    # start the graph
    with tf.Graph().as_default() as g:

        # build graph
        thresholded_importances = callpeak_graph.build_graph()

        # setup session
        sess, coord, threads = setup_tensorflow_session()

        with h5py.File(out_h5, 'w') as out_hf:

            # initialize datasets
            importance_mat = out_hf.create_dataset(
                feature_key, [num_examples, 4, seq_length])
            labels_mat = out_hf.create_dataset(
                'labels', [num_examples, num_tasks])
            regions_mat = out_hf.create_dataset(
                'regions', [num_examples, 1], dtype='S100')

            # run through batches of sequence
            for batch_idx in range(num_examples / batch_size + 1):

                print batch_idx * batch_size

                batch_importances, batch_regions, batch_labels = sess.run([thresholded_tensor,
                                                                           metadata,
                                                                           labels])

                batch_start = batch_idx * batch_size
                batch_stop = batch_start + batch_size

                # TODO save out to hdf5 file
                if batch_stop < num_examples:
                    importance_mat[batch_start:batch_stop,:] = batch_importances
                    labels_mat[batch_start:batch_stop,:] = batch_labels
                    regions_mat[batch_start:batch_stop] = batch_regions.astype('S100')
                else:
                    importance_mat[batch_start:num_examples,:] = batch_importances[0:num_examples-batch_start,:]
                    labels_mat[batch_start:num_examples,:] = batch_labels[0:num_examples-batch_start]
                    regions_mat[batch_start:num_examples] = batch_regions[0:num_examples-batch_start].astype('S100')
        
        # close session
        close_tensorflow_session(coord, threads)

    return None



def visualize_sample_sequences(h5_file, num_task, out_dir, sample_size=10):
    """Quick check on importance scores. Find a set of positive
    and negative sequences to visualize

    Args:
      h5_file: hdf5 file of importance scores
      num_task: which task to focus on
      out_dir: where to store these sample sequences
      sample_size: number of regions to visualize

    Returns:
      Plots of visualized sequences
    """
    importances_key = 'importances_task{}'.format(num_task)
    
    with h5py.File(h5_file, 'r') as hf:
        labels = hf['labels'][:,0]

        for label_val in [1, 0]:

            visualized_region_num = 0
            region_idx = 0

            while visualized_region_num < sample_size:

                if hf['labels'][region_idx,0] == label_val:
                    # get sequence and plot it out
                    sequence = np.squeeze(hf[importances_key][region_idx,:,:])
                    name = hf['regions'][region_idx,0]

                    start = int(name.split(':')[1].split('-')[0])
                    stop = int(name.split('-')[1])
                    sequence_len = stop - start
                    sequence = sequence[:,0:sequence_len]

                    print name
                    print sequence.shape
                    out_plot = '{0}/task_{1}.label_{2}.region_{3}.{4}.png'.format(out_dir, num_task, label_val,
                                                                 visualized_region_num, 
                                                                 name.replace(':', '-'))
                    print out_plot
                    plot_weights(sequence, out_plot)

                    visualized_region_num += 1
                    region_idx += 1

                else:
                    region_idx += 1
                
    return None
