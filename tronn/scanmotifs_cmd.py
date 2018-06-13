# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import json
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

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import read_pwm_file

from tronn.interpretation.clustering import generate_simple_metaclusters
from tronn.interpretation.clustering import refine_clusters
from tronn.interpretation.clustering import visualize_clusters

from tronn.interpretation.clustering import aggregate_pwm_results
from tronn.interpretation.clustering import get_manifold_centers


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

    # set up model info
    with open(args.model_info, "r") as fp:
        model_info = json.load(fp)
        
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    data_files = [h5_file for h5_file in data_files if "negative" not in h5_file]
    logger.info("Found {} chrom files".format(len(data_files)))
    
    # motif annotations
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))

    # set up dataloader
    dataloader = H5DataLoader(data_files)
    input_fn = dataloader.build_input_fn(
        args.batch_size,
        label_keys=model_info["label_keys"],
        filter_tasks=[
            args.inference_tasks,
            args.filter_tasks],
        singleton_filter_tasks=args.inference_tasks)

    # set up model
    model_manager = ModelManager(
        net_fns[model_info["name"]],
        model_info["params"])

    # set up inference generator
    inference_generator = model_manager.infer(
        input_fn,
        args.out_dir,
        net_fns[args.inference_fn],
        inference_params={
            "backprop": args.backprop,
            "importance_task_indices": args.inference_tasks,
            "pwms": pwm_list},
        checkpoint=model_info["checkpoint"],
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
                        pwm.name for pwm in pwm_list]
                        
    # now run clustering
    if args.cluster:
        visualize = True # TODO adjust later
        logging.info("running clustering - louvain (Phenograph)")
        dataset_keys = [
            "pwm-scores.taskidx-{}".format(i)
            for i in args.inference_tasks]

        # clustering
        metacluster_key = "metaclusters"
        if metacluster_key not in h5py.File(results_h5_file, "r").keys():
            generate_simple_metaclusters(results_h5_file, dataset_keys, metacluster_key)

        # refine
        refined_metacluster_key = "metaclusters-refined"
        if refined_metacluster_key not in h5py.File(results_h5_file, "r").keys():
            refine_clusters(results_h5_file, metacluster_key, refined_metacluster_key, null_cluster_present=False)
            if visualize:
                for i in xrange(len(dataset_keys)):
                    visualize_clusters(
                        results_h5_file,
                        dataset_keys[i],
                        refined_metacluster_key, 0,
                        remove_final_cluster=1)

        # get the manifold descriptions out per cluster
        manifold_key = "motifspace-centers"
        manifold_h5_file = "{0}/{1}.manifold.h5".format(
            args.tmp_dir, args.prefix)
        if not os.path.isfile(manifold_h5_file):
            get_manifold_centers(
                results_h5_file,
                dataset_keys,
                refined_metacluster_key,
                manifold_h5_file,
                pwm_list,
                pwm_dict)

        # get the overall subset of pwms with some significance in some cluster
        aggregate_pwm_results(results_h5_file, dataset_keys, manifold_h5_file)

        
    # TODO consider optional correlation matrix


    return None

