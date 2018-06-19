# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter

from tronn.graphs import ModelManager

from tronn.datalayer import H5DataLoader
from tronn.nets.nets import net_fns

from tronn.interpretation.clustering import generate_simple_metaclusters
from tronn.interpretation.clustering import refine_clusters
from tronn.interpretation.clustering import aggregate_pwm_results
from tronn.interpretation.clustering import get_manifold_centers

from tronn.interpretation.clustering import aggregate_pwm_results
from tronn.interpretation.clustering import aggregate_pwm_results_per_cluster

from tronn.visualization import visualize_clustered_h5_dataset_full
from tronn.visualization import visualize_aggregated_h5_datasets
from tronn.visualization import visualize_datasets_by_cluster_map
from tronn.visualization import visualize_h5_dataset
from tronn.visualization import visualize_h5_dataset_by_cluster


def _visualize_results(
        h5_file,
        cluster_key,
        inference_task_indices,
        visualize_task_indices,
        visualize_signals):
    """visualize the results
    """
    dataset_keys = [
        "pwm-scores.taskidx-{}".format(i)
        for i in inference_task_indices]
    visualize_task_indices = [inference_task_indices] + visualize_task_indices

    # (1)
    # look at examples x key of interest (all same R script):
    for i in xrange(len(dataset_keys)):
        visualize_clustered_h5_dataset_full(
            h5_file,
            cluster_key,
            dataset_keys[i],
            normalize=True,
            cluster_columns=True)

    for task_indices in visualize_task_indices:
        # first probs
        visualize_clustered_h5_dataset_full(
            h5_file,
            cluster_key,
            "probs",
            indices=task_indices)

        # then the label set
        visualize_clustered_h5_dataset_full(
            h5_file,
            cluster_key,
            "labels",
            indices=task_indices)

    # (2)
    # key: all signals desired (use keys and the indices with the keys)
    for signal_key in visualize_signals:
        signal_indices = visualize_signals[signal_key][0]
        visualize_clustered_h5_dataset_full(
            h5_file,
            cluster_key,
            signal_key,
            indices=signal_indices)

    # (3)
    # look at aggregate, comparing probs to labels/signal
    for task_indices in visualize_task_indices:
        visualize_aggregated_h5_datasets(
            h5_file,
            cluster_key,
            ["probs", "labels"],
            [task_indices, task_indices])

    # (4)
    # key: probs + signals (1 plot) (split by task index set?)
    # need to have {cluster, (prob/label), idx}
    # script needs to take key (probs), absolute task indices (probs), signal key, signal indices
    # need to match absolute indices with signal key....
    for key in visualize_signals:
        label_key = visualize_signals[key][1].get("label_key", None)
        if label_key is None:
            dataset_keys = ["probs", key]
        else:
            dataset_keys = ["probs", label_key, key]
        key_indices = visualize_signals[key][0]
        key_indices = [key_indices for i in xrange(len(dataset_keys))]
        visualize_aggregated_h5_datasets(
            h5_file,
            cluster_key,
            dataset_keys,
            key_indices)

    # (5)
    # can also do a correlation between clusters and specific label sets?
    # ie for a cluster, what is the average (or summed) signal?
    # need to have {cluster, label}
    for task_indices in visualize_task_indices:
        for dataset_key in ["probs", "labels"]:
            visualize_datasets_by_cluster_map(
                h5_file,
                cluster_key,
                dataset_key,
                indices=task_indices)

    for key in visualize_signals:
        visualize_datasets_by_cluster_map(
            h5_file,
            cluster_key,
            key,
            indices=visualize_signals[key][0])

    return None


def run(args):
    """Scan motifs from a PWM file
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running motif scan")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir
        
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    data_files = [h5_file for h5_file in data_files if "negative" not in h5_file]
    logger.info("Found {} chrom files".format(len(data_files)))

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
            "backprop": args.backprop,
            "importance_task_indices": args.inference_task_indices,
            "pwms": args.pwm_list},
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

        # add in PWM names to the datasets
        with h5py.File(results_h5_file, "a") as hf:
            for dataset_key in hf.keys():
                if "pwm-scores" in dataset_key:
                    hf[dataset_key].attrs["pwm_names"] = [
                        pwm.name for pwm in args.pwm_list]
                        
    # now run clustering
    if args.cluster:
        visualize = True # TODO adjust later
        logging.info("running clustering - louvain (Phenograph)")
        dataset_keys = [
            "pwm-scores.taskidx-{}".format(i)
            for i in args.inference_task_indices]

        # clustering: do this using all information (across all tasks)
        metacluster_key = "metaclusters"
        #cluster_keys = ["final_hidden"]
        cluster_keys = dataset_keys
        if metacluster_key not in h5py.File(results_h5_file, "r").keys():
            generate_simple_metaclusters(results_h5_file, cluster_keys, metacluster_key)

        # refine
        refined_metacluster_key = "metaclusters-refined"
        if refined_metacluster_key not in h5py.File(results_h5_file, "r").keys():
            refine_clusters(
                results_h5_file,
                metacluster_key,
                refined_metacluster_key,
                null_cluster_present=False)

        # visualize in R
        if visualize:

            #print args.inference_tasks
            #print args.visualize_tasks
            #print args.visualize_task_indices
            #print args.visualize_signals

            # TODO need to adjust the output name
            _visualize_results(
                results_h5_file,
                refined_metacluster_key,
                args.inference_task_indices,
                args.visualize_task_indices,
                args.visualize_signals)
                
        # get the manifold descriptions out per cluster
        # TODO also set up new cluster definitions, and re-visualize?
        manifold_key = "motifspace-centers"
        manifold_h5_file = "{0}/{1}.manifold.h5".format(
            args.tmp_dir, args.prefix)
        dataset_keys = [
            "pwm-scores.taskidx-{}".format(i)
            for i in args.inference_task_indices]
        global_agg_key = "pwm-scores.tasks_x_pwm.global"
        agg_key = "pwm-scores.tasks_x_pwm.per_cluster"
        if not os.path.isfile(manifold_h5_file):
            get_manifold_centers(
                results_h5_file,
                dataset_keys,
                refined_metacluster_key,
                manifold_h5_file,
                args.pwm_list,
                args.pwm_dict)

            # get the overall subset of pwms with some significance
            aggregate_pwm_results(
                results_h5_file,
                dataset_keys,
                global_agg_key,
                manifold_h5_file)
            
            # get by cluster
            aggregate_pwm_results_per_cluster(
                results_h5_file,
                refined_metacluster_key,
                dataset_keys,
                agg_key,
                manifold_h5_file)
            
        if visualize:
            visualize_h5_dataset(results_h5_file, global_agg_key)        
            visualize_h5_dataset_by_cluster(results_h5_file, agg_key)
        
    # TODO consider optional correlation matrix?

    return None

