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
import pandas as pd
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from tronn.util.tf_utils import setup_tensorflow_session
from tronn.util.tf_utils import close_tensorflow_session

from tronn.visualization import plot_weights
from tronn.interpretation.regions import RegionImportanceTracker

from tronn.util.tf_ops import restore_variables_op

from tronn.visualization import plot_weights


# TODO deprecate - should this ever be used?
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


def visualize_region(
        region,
        region_arrays,
        viz_params,
        prefix="viz",
        global_idx=10):
    """Given a dict representing a region, plot necessary things
    """
    # set up
    region_string = region.strip("\x00").split(";")[0].split("=")[1].replace(":", "-")
    region_prefix = "{}.{}".format(prefix, region_string)

    pwm_list = viz_params.get("pwms")
    grammars = viz_params.get("grammars")
    #assert pwm_list is not None

    pwm_x_pos_scores_file = None
    global_pwm_scores_file = None
    
    for key in region_arrays.keys():

        # plot importance scores across time (keys="importances.taskidx-{}")
        if "importances.taskidx" in key:
            # squeeze and visualize!
            plot_name = "{}.{}.pdf".format(region_prefix, key)
            print plot_name
            plot_weights(np.squeeze(region_arrays[key]), plot_name) # array, fig name

        elif False:
        #elif "global-pwm-scores" in key:
        
            # save out to matrix
            pwm_x_pos_scores_file = "{}.{}.txt".format(region_prefix, key)
            motif_pos_scores = np.transpose(np.squeeze(region_arrays[key]))
            
            # pad here as well to match the importance score array
            full_length = region_arrays["importances.taskidx-0"].shape[0]
            score_length = motif_pos_scores.shape[1]
            edge_pad_length = (full_length - score_length) / 2
            left_zero_pad = np.zeros((motif_pos_scores.shape[0], edge_pad_length))
            right_zero_pad = np.zeros(
                (motif_pos_scores.shape[0],
                 full_length - score_length - edge_pad_length))
            motif_pos_scores = np.concatenate(
                [left_zero_pad, motif_pos_scores, right_zero_pad], axis=1)
            motif_pos_scores_df = pd.DataFrame(
                motif_pos_scores,
                index=[pwm.name for pwm in pwm_list])
            motif_pos_scores_df.to_csv(pwm_x_pos_scores_file, header=False, sep="\t")

            if grammars is not None:
                # and also save out version with just the motifs in the grammar
                for grammar in grammars:
                    grammar_pwm_scores_file = "{}.{}.txt".format(
                        all_pwm_scores_file.split(".txt")[0],
                        grammar.name.replace(".", "-"))
                    grammar_pwm_scores_df = motif_pos_scores_df.loc[grammar.nodes]
                    grammar_pwm_scores_df.to_csv(grammar_pwm_scores_file, header=False, sep="\t")
            
        #elif "pwm-scores.taskidx-{}".format(global_idx) in key:
        elif False:
            # save out to matrix
            global_pwm_scores_file = "{}.{}.txt".format(region_prefix, key)
            motif_pos_scores = np.transpose(np.squeeze(region_arrays[key]))
            motif_pos_scores_df = pd.DataFrame(
                motif_pos_scores,
                index=[pwm.name for pwm in pwm_list])
            motif_pos_scores_df.to_csv(global_pwm_scores_file, header=False, sep="\t")
            
        elif "prob" in key:
            print "probs:", region_arrays[key][0:12]

        elif "logit" in key:
            print "logits:", region_arrays[key][0:12]

    # and plot
    #print pwm_x_pos_scores_file
    #print global_pwm_scores_file
    if False:
    #if (pwm_x_pos_scores_file is not None) and (global_pwm_scores_file is not None):
        plot_motif_x_pos = "plot.pwm_x_position.R {} {}".format(pwm_x_pos_scores_file, global_pwm_scores_file)
        print plot_motif_x_pos
        os.system(plot_motif_x_pos)

    return None


# BUT NOT YET - this is used in the other viz module, where you have specific sequences
# TODO remove this. eventually will want to visualize random samples on occasion
# or under a condition, but during the normal running of interpret - therefore will
# still want to save outputs to hdf5
# that you are quickly checking.
def interpret_and_viz(
        tronn_graph,
        model_checkpoint,
        batch_size,
        sample_size=None,
        method="input_x_grad",
        keep_negatives=False,
        filter_by_prediction=True):
    """Set up a graph and run inference stack
    """
    with tf.Graph().as_default() as g:

        # build graph
        if method == "input_x_grad":
            print "using input_x_grad"
            outputs = tronn_graph.build_inference_graph()
        elif method == "guided_backprop":
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                print "using guided backprop"
                outputs = tronn_graph.build_inference_graph()
                
        # set up session
        sess, coord, threads = setup_tensorflow_session()

        # restore
        init_assign_op, init_feed_dict = restore_variables_op(
            model_checkpoint, skip=["pwm"])
        sess.run(init_assign_op, init_feed_dict)
        
        # set up outlayer
        example_generator = ExampleGenerator(
            sess,
            outputs,
            batch_size,
            reconstruct_regions=False,
            keep_negatives=keep_negatives,
            filter_by_prediction=False,
            filter_tasks=tronn_graph.importances_tasks)

        # run all samples unless sample size is defined
        try:
            total_examples = 0
            while not coord.should_stop():
                    
                region, region_arrays = example_generator.run()
                region_arrays["example_metadata"] = region

                for key in region_arrays.keys():
                    if "pwm" in key:
                        # squeeze and visualize!
                        plot_name = "{}.{}.png".format(region.replace(":", "-"), key)
                        print plot_name
                        #plot_weights(np.squeeze(region_arrays[key][400:600,:]), plot_name) # array, fig name
                        plot_weights(np.squeeze(region_arrays[key]), plot_name) # array, fig name
                    if "prob" in key:
                        print region_arrays[key]

                total_examples += 1
                print total_examples
                        
                # check condition
                if (sample_size is not None) and (total_examples >= sample_size):
                    break

        except tf.errors.OutOfRangeError:
            print "Done reading data"
            # add in last of the examples

        finally:
            time.sleep(60)

        # catch the exception ValueError - (only on sherlock, come back to this)
        try:
            close_tensorflow_session(coord, threads)
        except:
            pass

    return None

