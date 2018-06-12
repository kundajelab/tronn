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

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import read_pwm_file

from tronn.interpretation.clustering import generate_simple_metaclusters
from tronn.interpretation.clustering import refine_clusters
from tronn.interpretation.clustering import visualize_clusters

from tronn.interpretation.clustering import aggregate_pwm_results
from tronn.interpretation.clustering import get_manifold_centers


def run(args):
    """command to analyze variants
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Analyzing variants")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logger.info("Found {} chrom files".format(len(data_files)))
    
    # motif annotations
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))

    # set up dataloader
    dataloader = H5DataLoader(data_files)
    input_fn = dataloader.build_variant_input_fn(
        args.batch_size,
        label_keys=args.label_keys,
        shuffle=False)

    # set up model
    model_manager = ModelManager(
        net_fns[args.model["name"]],
        args.model)

    # set up inference generator
    inference_generator = model_manager.infer(
        input_fn,
        args.out_dir,
        net_fns[args.inference_fn],
        inference_params={
            "use_filtering": False,
            "backprop": args.backprop,
            "importance_task_indices": args.inference_tasks,
            "pwms": pwm_list},
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

        # add in PWM names to the datasets
        with h5py.File(results_h5_file, "a") as hf:
            for dataset_key in hf.keys():
                if "pwm-scores" in dataset_key:
                    hf[dataset_key].attrs["pwm_names"] = [
                        pwm.name for pwm in pwm_list]
                    
    # here, downstream processing
    # TODO look at ref vs alt in the output predictions for ATAC and H3K27ac

    

    return None

