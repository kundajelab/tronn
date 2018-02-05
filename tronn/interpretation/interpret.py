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
    region_string = region.strip("\x00").split(";")[0].split("=")[1].replace(":", "-")
    region_prefix = "{}.{}".format(prefix, region_string)

    pwm_list = viz_params.get("pwms")
    grammars = viz_params.get("grammars")
    
    for key in region_arrays.keys():
        
        # plot importance scores across time (keys="importances.taskidx-{}")
        if "importances.taskidx" in key:
            # squeeze and visualize!
            plot_name = "{}.{}.pdf".format(region_prefix, key)
            print plot_name
            #plot_weights(np.squeeze(region_arrays[key][400:600,:]), plot_name) # array, fig name
            plot_weights(np.squeeze(region_arrays[key]), plot_name) # array, fig name
            
        elif "global-pwm-scores" in key:
            # save out to matrix
            # save it out with pwm names...
            all_pwm_scores_file = "{}.{}.txt".format(region_prefix, key)
            motif_pos_scores = np.transpose(np.squeeze(region_arrays[key]))
            motif_pos_scores_df = pd.DataFrame(
                motif_pos_scores,
                index=[pwm.name for pwm in pwm_list])
            motif_pos_scores_df.to_csv(all_pwm_scores_file, header=False, sep="\t")
            
            # and also save out version with just the motifs in the grammar
            for grammar in grammars:
                grammar_pwm_scores_file = "{}.{}.txt".format(
                    all_pwm_scores_file.split(".txt")[0],
                    grammar.name.replace(".", "-"))
                grammar_pwm_scores_df = motif_pos_scores_df.loc[grammar.nodes]
                grammar_pwm_scores_df.to_csv(grammar_pwm_scores_file, header=False, sep="\t")
                
            
        elif "pwm-scores.taskidx-{}".format(global_idx) in key:
            # save out to matrix
            file_name = "{}.{}.txt".format(region_prefix, key)
            motif_pos_scores = np.transpose(np.squeeze(region_arrays[key]))
            motif_pos_scores_df = pd.DataFrame(
                motif_pos_scores,
                index=[pwm.name for pwm in pwm_list])
            motif_pos_scores_df.to_csv(file_name, header=False, sep="\t")

        elif "prob" in key:
            print "probs:", region_arrays[key][0:12]

        elif "logit" in key:
            print "logits:", region_arrays[key][0:12]
            
    # after this, collect the global and pwm max scores and plot with R
    # TODO

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
        visualize_only=True,
        num_to_visualize=10,
        viz_bp_cutoff=25):
    """Set up a graph and run inference stack
    """
    with tf.Graph().as_default() as g:

        # build graph
        if method == "input_x_grad":
            print "using input_x_grad"
            outputs = tronn_graph.build_inference_graph(inference_params)
        #elif method == "guided_backprop":
        #    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        #        print "using guided backprop"
        #        outputs = tronn_graph.build_inference_graph(pwm_list=pwm_list)
        else:
            print "unsupported method"
            quit()
                
        # set up session
        sess, coord, threads = setup_tensorflow_session()
                    
        # restore
        if model_checkpoint is not None:
            init_assign_op, init_feed_dict = restore_variables_op(
                model_checkpoint, skip=["pwm"])
            sess.run(init_assign_op, init_feed_dict)
        else:
            print "WARNING"
            print "WARNING"
            print "WARNING: did not use checkpoint. are you sure? this should only happen for empty_net"

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
                    
                    # conditions for visualization: logits > 0,
                    # and mean(importances) > 0, AND not empty net
                    # set up real condition
                    logits = region_arrays["logits"][0:12]
                    num_pos_impt_bps = region_arrays["positive_importance_bp_sum"]
                    
                    if np.max(logits) > 0 and num_pos_impt_bps >= viz_bp_cutoff and total_visualized < num_to_visualize:
                        out_dir = "{}/viz".format(os.path.dirname(h5_file))
                        os.system("mkdir -p {}".format(out_dir))

                        visualize_region(
                            region,
                            region_arrays,
                            {"pwms": inference_params["pwms"],
                             "grammars": inference_params["grammars"]},
                            prefix="{}/viz".format(out_dir),
                            global_idx=10)
                        total_visualized += 1

                    # check condition
                    if (sample_size is not None) and (total_examples >= sample_size):
                        break

                    # check viz condition
                    if visualize_only and total_visualized >= num_to_visualize:
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


# TODO remove this. eventually will want to visualize random samples on occasion
# or under a condition, but during the normal running of interpret - therefore will
# still want to save outputs to hdf5
# BUT NOT YET - this is used in the other viz module, where you have specific sequences
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
