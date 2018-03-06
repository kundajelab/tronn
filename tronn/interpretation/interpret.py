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

from tronn.outlayer import ExampleGenerator
from tronn.outlayer import H5Handler

from tronn.util.tf_ops import restore_variables_op

from tronn.visualization import plot_weights


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
    assert pwm_list is not None

    pwm_x_pos_scores_file = None
    global_pwm_scores_file = None
    
    for key in region_arrays.keys():

        # plot importance scores across time (keys="importances.taskidx-{}")
        if "importances.taskidx" in key:
            # squeeze and visualize!
            plot_name = "{}.{}.pdf".format(region_prefix, key)
            print plot_name
            plot_weights(np.squeeze(region_arrays[key]), plot_name) # array, fig name
            
        elif "global-pwm-scores" in key:
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
            
        elif "pwm-scores.taskidx-{}".format(global_idx) in key:
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
    print pwm_x_pos_scores_file
    print global_pwm_scores_file
    if (pwm_x_pos_scores_file is not None) and (global_pwm_scores_file is not None):
        plot_motif_x_pos = "plot.pwm_x_position.R {} {}".format(pwm_x_pos_scores_file, global_pwm_scores_file)
        print plot_motif_x_pos
        os.system(plot_motif_x_pos)

    return None


def interpret(
        tronn_graph,
        model_checkpoint,
        batch_size,
        h5_file,
        sample_size=None,
        inference_params={},
        method="input_x_grad",
        keep_negatives=False,
        filter_by_prediction=True,
        h5_batch_size=128,
        visualize=False,
        num_to_visualize=10,
        scan_grammars=False,
        validate_grammars=False,
        viz_bp_cutoff=25):
    """Set up a graph and run inference stack
    """
    logger = logging.getLogger(__name__)
    logger.info("Running interpretation")

    if visualize:
        viz_dir = "{}/viz".format(os.path.dirname(h5_file))
        os.system("mkdir -p {}".format(viz_dir))
        
    with tf.Graph().as_default() as g:

        # set up inference graph
        outputs = tronn_graph.build_inference_graph(
            inference_params,
            scan_grammars=scan_grammars,
            validate_grammars=validate_grammars)
                
        # set up session
        sess, coord, threads = setup_tensorflow_session()
                    
        # restore from checkpoint as needed
        if model_checkpoint is not None:
            init_assign_op, init_feed_dict = restore_variables_op(
                model_checkpoint, skip=["pwm"])
            sess.run(init_assign_op, init_feed_dict)
        else:
            print "WARNING WARNING WARNING: did not use checkpoint. are you sure?"

        # set up hdf5 file to store outputs
        with h5py.File(h5_file, 'w') as hf:

            h5_handler = H5Handler(
                hf, outputs, sample_size, resizable=True, batch_size=4096)

            # set up outlayer
            example_generator = ExampleGenerator(
                sess,
                outputs,
                batch_size,
                reconstruct_regions=False,
                keep_negatives=keep_negatives,
                filter_by_prediction=filter_by_prediction,
                filter_tasks=tronn_graph.importances_tasks)

            # run all samples unless sample size is defined
            try:
                total_examples = 0
                total_visualized = 0
                while not coord.should_stop():
                    
                    region, region_arrays = example_generator.run()
                    region_arrays["example_metadata"] = region

                    h5_handler.store_example(region_arrays)
                    total_examples += 1
                    
                    logits = region_arrays["logits"][0:12]
                    
                    try:
                        num_pos_impt_bps = region_arrays["positive_importance_bp_sum"]
                    except:
                        num_pos_impt_bps = viz_bp_cutoff
                        
                    if visualize and (total_visualized < num_to_visualize):
                        # only visualize if logits > 0, enough impt bps, and under num to vis cutoff
                        if (np.max(logits) > 0) and (num_pos_impt_bps >= viz_bp_cutoff):
                            visualize_region(
                                region,
                                region_arrays,
                                {"pwms": inference_params.get("pwms"),
                                 "grammars": inference_params.get("grammars")},
                                prefix="{}/viz".format(viz_dir),
                                global_idx=10)
                            total_visualized += 1
                            
                        # check viz condition
                        # TODO maybe provide an option to just visualize?
                        #if total_visualized >= num_to_visualize:
                        #    break
                            
                    # check condition
                    if (sample_size is not None) and (total_examples >= sample_size):
                        break

            #except tf.errors.OutOfRangeError:
            except Exception as e:
                print e
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

