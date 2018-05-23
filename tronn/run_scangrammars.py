# description: scan for grammar scores

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from tronn.util.h5_utils import h5_dataset_to_text_file

#from tronn.graphs import TronnGraph
#from tronn.graphs import TronnNeuralNetGraph
from tronn.graphs import TronnGraphV2

#from tronn.datalayer import load_data_from_filename_list
#from tronn.datalayer import load_data_with_shuffles_from_filename_list
from tronn.datalayer import H5DataLoader

from tronn.nets.nets import net_fns

#from tronn.interpretation.interpret import interpret
from tronn.interpretation.interpret import interpret_v2

from tronn.interpretation.motifs import read_pwm_file
from tronn.interpretation.motifs import setup_pwms
from tronn.interpretation.motifs import setup_pwm_metadata

from tronn.interpretation.grammars import read_grammar_file
from tronn.interpretation.grammars import get_significant_delta_motifs
from tronn.interpretation.grammars import generate_grammars_from_dmim


def visualize_scores(
        h5_file,
        dataset_key):
    """Visualize clustering. Note that the R script is downsampling
    to make things visible.
    """
    # do this in R
    plot_example_x_pwm = (
        "plot.example_x_pwm_mut.from_h5.R {0} {1}").format(
            h5_file, dataset_key)
    print plot_example_x_pwm
    os.system(plot_example_x_pwm)
    
    return None


def run(args):
    """Scan and score grammars
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running grammar scan")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logging.info("Found {} chrom files".format(len(data_files)))

    # pull in motif annotation
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))

    # set up dataloader
    dataloader = H5DataLoader(data_files)
    input_fn = dataloader.build_input_fn(
        args.batch_size,
        filter_tasks=[
            args.inference_tasks,
            args.filter_tasks],
        singleton_filter_tasks=args.inference_tasks)

    # set up model
    model_manager = ModelManager(
        net_fns[args.model["name"]])

    # set up inference generator
    inference_generator = model_manager.infer(
        input_fn,
        args.out_dir,
        net_fns[args.inference_fn],
        inference_params={
            "model_fn": net_fns[args.model["name"]],
            "backprop": args.backprop,
            "importance_task_indices": args.inference_tasks,
            "pwms": pwm_list,
            "manifold": args.manifold_file},
        checkpoint=args.model_checkpoints[0],
        yield_single_examples=True)

    # run inference and save out
    results_h5_file = "{0}/{1}.inference.h5".format(
        args.tmp_dir, args.prefix)
    if not os.path.isfile(results_h5_file):
        model_manager.infer_and_save_to_h5(
            inference_generator,
            results_h5_file,
            args.sample_size)
    
        # save out additional useful information
        with h5py.File(results_h5_file, "a") as hf:

            # save master pwm vector
            with h5py.File(args.manifold_file, "r") as manifold:
                hf.create_dataset("master_pwm_vector", data=manifold["master_pwm_vector"][:])
                
            # attach to delta logits and mutated scores
            hf["delta_logits"].attrs["pwm_mut_names"] = pwm_names
            for task_idx in args.inference_tasks:
                hf["dmim-scores.taskidx-{}".format(task_idx)].attrs["pwm_mut_names"] = pwm_names

    generate_grammars_from_dmim(results_h5_file, args.inference_tasks, pwm_list)
    "plot.pwm_x_pwm.mut3.from_h5.R {} dmim-scores.merged.master".format(results_h5_file)



    quit()
    # set up graph
    tronn_graph = TronnGraphV2(
        dataloader,
        net_fns[args.model["name"]],
        args.model,
        args.batch_size,
        final_activation_fn=tf.nn.sigmoid,
        checkpoints=args.model_checkpoints)

    # run interpretation graph
    results_h5_file = "{0}/{1}.inference.h5".format(
        args.tmp_dir, args.prefix)
    if not os.path.isfile(results_h5_file):
    #if False:
        infer_params = {
            "model_fn": net_fns[args.model["name"]],
            "inference_fn": net_fns[args.inference_fn],
            "importances_fn": args.backprop,
            "importance_task_indices": args.inference_tasks,
            "pwms": pwm_list,
            "manifold": args.manifold_file}
        interpret_v2(tronn_graph, results_h5_file, infer_params, num_evals=args.sample_size)
        
    # with this vector, generate a reduced heatmap
    # and then threshold and make a directed graph
    if True:
        from tronn.interpretation.grammars import generate_networks
        generate_networks(
            results_h5_file,
            dmim_motifs_key,
            args.inference_tasks,
            pwm_list,
            pwm_dict)
    
    
    return None
