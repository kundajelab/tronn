# description: scan for grammar scores

import os
import json
import h5py
import glob
import logging

import numpy as np
import tensorflow as tf

from tronn.graphs import ModelManager
from tronn.datalayer import H5DataLoader
from tronn.nets.nets import net_fns

from tronn.interpretation.motifs import read_pwm_file
from tronn.interpretation.grammars import generate_grammars_from_dmim
from tronn.interpretation.grammars import aggregate_dmim_results

from tronn.interpretation.clustering import aggregate_pwm_results
from tronn.interpretation.clustering import aggregate_pwm_results_per_cluster

from tronn.visualization import visualize_h5_dataset
from tronn.visualization import visualize_h5_dataset_by_cluster
from tronn.visualization import visualize_clustering_results

from tronn.visualization import visualize_agg_pwm_results
from tronn.visualization import visualize_agg_delta_logit_results
from tronn.visualization import visualize_agg_dmim_adjacency_results

def _visualize_mut_results(
        h5_file,
        pwm_scores_key,
        delta_logits_key,
        dmim_adjacency_key,
        visualize_task_indices,
        pwm_names_attribute,
        mut_pwm_names_attribute,
        master_pwm_vector_key="master_pwm_vector",
        motif_filter_key="mut_pwm_vectors.agg"):
    """visualize out results
    """
    # (1) visualize all the cluster results (which are filtered for motif presence)
    visualize_agg_pwm_results(h5_file, pwm_scores_key, pwm_names_attribute, master_pwm_vector_key)
        
    # visualize delta logits (in groups) (delta_logits)
    for idx_set in visualize_task_indices:
        visualize_agg_delta_logit_results(
            h5_file,
            delta_logits_key,
            motif_filter_key,
            idx_set,
            mut_pwm_names_attribute)

    # adjacency results - {task, pwm, pwm} for specific indices (all indices in list)
    # dmim-scores.mut_only
    visualize_agg_dmim_adjacency_results(
        h5_file,
        dmim_adjacency_key,
        motif_filter_key,
        mut_pwm_names_attribute)
    
    return None


def run(args):
    """run delta motif interaction mutagenesis (DMIM)
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running dmim scan")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir
        
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    data_files = [h5_file for h5_file in data_files if "negative" not in h5_file]
    logging.info("Found {} chrom files".format(len(data_files)))

    # set up dataloader
    dataloader = H5DataLoader(data_files)
    input_fn = dataloader.build_input_fn(
        args.batch_size,
        label_keys=args.model_info["label_keys"],
        filter_tasks=[
            args.inference_task_indices,
            args.filter_task_indices],
        singleton_filter_tasks=args.inference_task_indices)

    # set up model
    model_manager = ModelManager(
        net_fns[args.model_info["name"]],
        args.model_info["params"])

    # set up inference generator
    inference_generator = model_manager.infer(
        input_fn,
        args.out_dir,
        net_fns[args.inference_fn],
        inference_params={
            "model_fn": net_fns[args.model_info["name"]],
            "backprop": args.backprop,
            "importances_fn": args.backprop, # TODO fix this
            "importance_task_indices": args.inference_task_indices,
            "pwms": args.pwm_list,
            "manifold": args.manifold_file},
        checkpoint=args.model_info["checkpoint"],
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
                #del hf["master_pwm_vector"]
                hf.create_dataset("master_pwm_vector", data=manifold["master_pwm_vector"][:])
                mut_indices = np.where(manifold["master_pwm_vector"][:] > 0)[0]
                pwm_mut_names = [args.pwm_list[i].name for i in mut_indices]
                
            # attach to delta logits and mutated scores
            hf["delta_logits"].attrs["pwm_mut_names"] = pwm_mut_names
            for task_idx in args.inference_task_indices:
                hf["dmim-scores.taskidx-{}".format(task_idx)].attrs["pwm_mut_names"] = pwm_mut_names

            # pwm names
            for dataset_key in hf.keys():
                if "pwm-scores" in dataset_key:
                    hf[dataset_key].attrs["pwm_names"] = [
                        pwm.name for pwm in args.pwm_list]

        # aggregate the pwm cluster results
        dataset_keys = [
            "pwm-scores.taskidx-{}".format(i)
            for i in args.inference_task_indices]
        global_agg_key = "pwm-scores.tasks_x_pwm.global"

        aggregate_pwm_results(
            results_h5_file,
            dataset_keys,
            global_agg_key,
            args.manifold_file)

        # TODO need to maintain cluster numbering
        agg_key = "pwm-scores.tasks_x_pwm.per_cluster"
        aggregate_pwm_results_per_cluster(
            results_h5_file,
            "manifold_clusters",
            dataset_keys,
            agg_key,
            args.manifold_file,
            soft_clustering=True)
        
    visualize = True
    if visualize:
        visualize_clustering_results(
            results_h5_file,
            "manifold_clusters.onehot",
            args.inference_task_indices,
            args.visualize_task_indices,
            args.visualize_signals,
            soft_cluster_key="manifold_clusters",
            remove_final_cluster=False)
        global_agg_key = "pwm-scores.tasks_x_pwm.global"
        agg_key = "pwm-scores.tasks_x_pwm.per_cluster"
        visualize_h5_dataset(results_h5_file, global_agg_key)        
        visualize_h5_dataset_by_cluster(results_h5_file, agg_key)

    # aggregate results
    dmim_keys = ["dmim-scores.taskidx-{}".format(i) for i in args.inference_task_indices]
    pwm_score_keys = ["pwm-scores.taskidx-{}".format(i) for i in args.inference_task_indices]
    if True:
        aggregate_dmim_results(
            results_h5_file,
            "manifold_clusters",
            args.inference_task_indices,
            dmim_keys,
            pwm_score_keys,
            args.pwm_list)

    visualize = True
    visualize_task_indices = [args.inference_task_indices] + args.visualize_task_indices
    if visualize:
        _visualize_mut_results(
            results_h5_file,
            "pwm-scores.agg",
            "delta_logits.agg",
            "dmim-scores.agg.mut_only",
            visualize_task_indices,
            "pwm_names",
            "mut_pwm_names")
    
    return None
