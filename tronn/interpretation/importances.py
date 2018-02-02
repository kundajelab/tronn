"""Description: contains code to generate per-base-pair importance scores
"""

import h5py
import logging
import math
import time
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

from tronn.util.tf_ops import restore_variables_op


# TODO - is this a defunct file? maybe delete

@ops.RegisterGradient("GuidedRelu_old")
def _GuidedReluGrad_old(op, grad):
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


def extract_importances_and_viz(
        tronn_graph,
        model_checkpoint,
        out_prefix,
        method="guided_backprop",
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
                outputs = tronn_graph.build_inference_graph(normalize=True)
        elif method == "simple_gradients":
            outputs = tronn_graph.build_inference_graph(normalize=False) # change this back
            
        # set up session
        sess, coord, threads = setup_tensorflow_session()

        # restore
        #saver = tf.train.Saver()
        #saver.restore(sess, model_checkpoint)
        init_assign_op, init_feed_dict = restore_variables_op(
            model_checkpoint, skip=["pwm"])
        sess.run(init_assign_op, init_feed_dict)

        # set up outlayer
        example_generator = ExampleGenerator(
            sess,
            outputs,
            1, # Fix this later
            reconstruct_regions=False,
            keep_negatives=True,
            filter_by_prediction=False,
            filter_tasks=tronn_graph.importances_tasks)
        
        for i in xrange(12):

            region, region_arrays = example_generator.run()
            region_arrays["example_metadata"] = region
            print region
            
            for key in region_arrays.keys():
                
                if "importance" in key:
                    # squeeze and visualize!
                    plot_name = "{}.{}.png".format(region.replace(":", "-"), key)
                    plot_weights(np.squeeze(region_arrays[key][:,400:600,:]), plot_name) # array, fig name
                if "prob" in key:
                    print region_arrays[key]

        # catch the exception ValueError - (only on sherlock, come back to this)
        try:
            close_tensorflow_session(coord, threads)
        except:
            pass

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
        # batch it because it's faster to pull from hdf5 in batches
        example_idx = 0
        batch_size = 4096
        num_batches = int(math.ceil(hf["feature_metadata"].shape[0] / float(batch_size)))

        for batch_idx in xrange(num_batches):

            if example_idx + batch_size > hf["feature_metadata"].shape[0]:
                batch_end = hf["feature_metadata"].shape[0]
                batch_example_num = batch_end - example_idx
            else:
                batch_end = example_idx + batch_size
                batch_example_num = batch_size

            # get batch of examples
            tmp_batch_arrays = {}
            for key in hf.keys():
                if "feature_metadata" in key:
                    tmp_batch_arrays[key] = hf[key][example_idx:batch_end, 0]
                elif "importance" in key:
                    tmp_batch_arrays[key] = hf[key][example_idx:batch_end,:,:]
                else:
                    tmp_batch_arrays[key] = hf[key][example_idx:batch_end,:]

            # then go through examples
            for batch_idx in xrange(batch_example_num):

                if example_idx % 1000 == 0:
                    print example_idx

                # first check if it belongs to a certain task
                task_handle_indices = []
                for i in xrange(len(tasks)):
                    if tmp_batch_arrays["labels"][batch_idx, tasks[i]] == 1:
                        task_handle_indices.append(i)

                if len(task_handle_indices) > 0:
                    # make into an example array
                    example_array = {}
                    for key in tmp_batch_arrays.keys():
                        if "feature_metadata" in key:
                            example_array[key] = tmp_batch_arrays[key][batch_idx]
                        elif "importance" in key:
                            example_array[key] = tmp_batch_arrays[key][batch_idx,:,:]
                        else:
                            example_array[key] = tmp_batch_arrays[key][batch_idx,:]

                    for task_idx in task_handle_indices:
                        task_h5_handlers[task_idx].store_example(example_array)

                example_idx += 1

        # at very end, push all last batches
        for i in xrange(len(tasks)):
            task_h5_handlers[i].flush()

    for i in xrange(len(tasks)):
        task_h5_handles[i].close()
            
    return task_h5_files

