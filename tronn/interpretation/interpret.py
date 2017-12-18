"""Description: Contains methods and routines for interpreting models
"""

import os
import glob
import json
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


def interpret(
        tronn_graph,
        model_checkpoint,
        h5_file,
        sample_size=None,
        pwm_list=None,
        method="input_x_grad",
        keep_negatives=False,
        filter_by_prediction=True,
        h5_batch_size=128):
    """Set up a graph and run inference stack
    """
    with tf.Graph().as_default() as g:

        # build graph
        if method == "input_x_grad":
            print "using input_x_grad"
            outputs = tronn_graph.build_inference_graph(pwm_list=pwm_list)
        elif method == "guided_backprop":
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                print "using guided backprop"
                outputs = tronn_graph.build_inference_graph_v3(pwm_list=pwm_list)
                
        # set up session
        sess, coord, threads = setup_tensorflow_session()

        # restore
        init_assign_op, init_feed_dict = restore_variables_op(
            model_checkpoint, skip=["pwm"])
        sess.run(init_assign_op, init_feed_dict)
        
        # set up hdf5 file to store outputs
        with h5py.File(h5_file, 'w') as hf:

            h5_handler = H5Handler(
                hf, outputs, sample_size, resizable=True, batch_size=4096)

            # set up outlayer
            example_generator = ExampleGenerator(
                sess,
                outputs,
                64, # Fix this later
                reconstruct_regions=False,
                keep_negatives=keep_negatives,
                filter_by_prediction=filter_by_prediction,
                filter_tasks=tronn_graph.importances_tasks)

            # run all samples unless sample size is defined
            try:
                total_examples = 0
                while not coord.should_stop():
                    
                    region, region_arrays = example_generator.run()
                    region_arrays["example_metadata"] = region

                    h5_handler.store_example(region_arrays)
                    total_examples += 1

                    # check condition
                    if (sample_size is not None) and (total_examples >= sample_size):
                        break

            except tf.errors.OutOfRangeError:
                print "Done reading data"
                # add in last of the examples

            finally:
                time.sleep(60)
                h5_handler.flush()
                h5_handler.chomp_datasets()

        # catch the exception ValueError - (only on sherlock, come back to this)
        try:
            close_tensorflow_session(coord, threads)
        except:
            pass

    return None




def run(args):

    # find data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    print 'Found {} chrom files'.format(len(data_files))

    # checkpoint file
    checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.model_dir))
    print checkpoint_path

    # set up scratch_dir
    os.system('mkdir -p {}'.format(args.scratch_dir))

    # load external data files
    with open(args.annotations, 'r') as fp:
        annotation_files = json.load(fp)

    # current manual choices
    task_nums = [0, 9, 10, 14]
    dendro_cutoffs = [7, 6, 7, 7]
    
    interpret(args,
              load_data_from_filename_list,
              data_files,
              models[args.model['name']],
              tf.losses.sigmoid_cross_entropy,
              args.prefix,
              args.out_dir,
              task_nums, 
              dendro_cutoffs, 
              annotation_files["motif_file"],
              annotation_files["motif_sim_file"],
              annotation_files["motif_offsets_file"],
              annotation_files["rna_file"],
              annotation_files["rna_conversion_file"],
              checkpoint_path,
              scratch_dir=args.scratch_dir,
              sample_size=args.sample_size)

    return
