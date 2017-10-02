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
from tronn.outlayer import H5Handler

from tronn.nets.importance_nets import importances_stdev_cutoff
from tronn.nets.importance_nets import normalize_to_probs


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


def layerwise_relevance_propagation(tensor, features, probs=None, normalize=False):
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
    importances_raw = tf.multiply(features, feature_grad, 'input_mul_grad')
    importances_squeezed = tf.transpose(tf.squeeze(importances_raw), perm=[0, 2, 1])

    if normalize:
        thresholded = importances_stdev_cutoff(importances_squeezed)
        importances = normalize_to_probs(thresholded, probs)
    else:
        importances = importances_squeezed
    
    return importances


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
                importances = tronn_graph.build_inference_graph(normalize=True)
        elif method == "simple_gradients":
            importances = tronn_graph.build_inference_graph(normalize=True)

        # set up session
        sess, coord, threads = setup_tensorflow_session()

        # restore
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint)

        # set up hdf5 file to store outputs
        with h5py.File(out_file, 'w') as hf:

            h5_handler = H5Handler(
                hf, importances, sample_size, resizable=True, batch_size=4096)

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
                while not coord.should_stop():

                    region, region_arrays = example_generator.run()
                    region_arrays["feature_metadata"] = region

                    h5_handler.store_example(region_arrays)
                    total_examples += 1

            except tf.errors.OutOfRangeError:

                # add in last of the examples
                h5_handler.flush()

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


def split_importances_by_task_positives(importances_main_file, tasks, prefix):
    """Given an importance score file and given specific tasks,
    generate separated importance files for tasks
    """
    # keep a list of hdf5 files
    task_h5_files = []
    task_h5_handles = []
    task_h5_handlers = []
    for i in range(len(tasks)):
        task_h5_files.append("{}.task_{}.h5".format(prefix, tasks[i]))
        task_h5_handles.append(h5py.File(task_h5_files[i]))
        
    # for each file, open and create relevant datasets. keep the handle
    with h5py.File(importances_main_file, "r") as hf:
        
        for i in xrange(len(tasks)):

            # get positives
            positives = np.sum(hf["labels"][:,tasks[i]])

            # make the datasets
            array_dict = {}
            for key in hf.keys():
                array_dict[key] = hf[key]
            task_h5_handlers.append(
                H5Handler(
                    task_h5_handles[i], array_dict, positives, resizable=False, is_tensor_input=False))

        # then go through batches of importances
        for example_idx in xrange(hf["feature_metadata"].shape[0]):

            if example_idx % 1000 == 0:
                print example_idx
            
            # first check if it belongs to a certain task
            task_handle_indices = []
            for i in xrange(len(tasks)):
                if hf["labels"][example_idx, tasks[i]] == 1:
                    task_handle_indices.append(i)

            if len(task_handle_indices) > 0:
                # make into an example array
                example_array = {}
                for key in hf.keys():
                    if "feature_metadata" in key:
                        example_array[key] = hf[key][example_idx, 0]
                    elif "importance" in key:
                        example_array[key] = hf[key][example_idx,:,:]
                    else:
                        example_array[key] = hf[key][example_idx,:]

                for task_idx in task_handle_indices:
                    task_h5_handlers[task_idx].store_example(example_array)
                    

        # at very end, push all last batches
        for i in xrange(len(tasks)):
            task_h5_handlers[i].flush()

    for i in xrange(len(tasks)):
        task_h5_handles[i].close()
            
    return task_h5_files


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
