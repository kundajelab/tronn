# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import h5py
import glob
import logging

import phenograph

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter

from tronn.graphs import TronnGraphV2
from tronn.graphs import ModelManager
from tronn.graphs import infer_and_save_to_hdf5

from tronn.datalayer import H5DataLoader
from tronn.nets.nets import net_fns

#from tronn.interpretation.interpret import interpret
#from tronn.interpretation.interpret import interpret_v2

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import read_pwm_file
#from tronn.interpretation.motifs import setup_pwms
#from tronn.interpretation.motifs import setup_pwm_metadata

from tronn.interpretation.motifs import get_minimal_motifsets
from tronn.interpretation.motifs import distill_to_linear_models
from tronn.interpretation.motifs import threshold_motifs
from tronn.interpretation.motifs import reduce_pwm_redundancy

from tronn.interpretation.clustering import cluster_by_task
from tronn.interpretation.clustering import enumerate_metaclusters
from tronn.interpretation.clustering import generate_simple_metaclusters
from tronn.interpretation.clustering import refine_clusters
from tronn.interpretation.clustering import visualize_clusters

from tronn.interpretation.clustering import get_correlation_file
from tronn.interpretation.clustering import get_manifold_centers

from tronn.interpretation.learning import build_lasso_regression_models


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
    logger.info("Found {} chrom files".format(len(data_files)))
    
    # motif annotations
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))

    # set up dataloader
    dataloader = H5DataLoader(
        {"data": data_files},
        filter_tasks=[
            args.inference_tasks,
            args.filter_tasks],
        singleton_filter_tasks=args.inference_tasks)
    input_fn = dataloader.build_estimator_input_fn("data", args.batch_size)

    if True:
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
                "checkpoint": args.model_checkpoints[0],
                "backprop": args.backprop,
                "importance_task_indices": args.inference_tasks,
                "pwms": pwm_list},
            #checkpoint="blah", #args.model_checkpoints[0],
            yield_single_examples=True)

        # run inference and save out
        results_h5_file = "{0}/{1}.inference.h5".format(
            args.tmp_dir, args.prefix)
        if not os.path.isfile(results_h5_file):
            infer_and_save_to_hdf5(
                inference_generator,
                results_h5_file,
                args.sample_size)

    if True:
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
            infer_params = {
                "inference_fn": net_fns[args.inference_fn],
                "importances_fn": args.backprop,
                "importance_task_indices": args.inference_tasks,
                "pwms": pwm_list}
            interpret_v2(tronn_graph, results_h5_file, infer_params)

    # attach useful information
    with h5py.File(results_h5_file, "a") as hf:
        # add in PWM names to the datasets
        for dataset_key in hf.keys():
            if "pwm-scores" in dataset_key:
                hf[dataset_key].attrs["pwm_names"] = [
                    pwm.name for pwm in pwm_list]
                
    # run region clustering/motif sets. default is true, but user can turn off
    # TODO split this out into another function
    pwm_scores_h5 = results_h5_file
    if not args.no_groups:
        visualize = True
        dataset_keys = ["pwm-scores.taskidx-{}".format(i)
                        for i in args.inference_tasks] # eventually, this is the right one
        
        # 1) cluster communities 
        cluster_key = "louvain_clusters" # later, change to pwm-louvain-clusters
        if False:
        #if cluster_key not in h5py.File(pwm_scores_h5, "r").keys():
            cluster_by_task(pwm_scores_h5, dataset_keys, cluster_key)
            if visualize:
                for i in xrange(len(dataset_keys)):
                    visualize_clusters(
                        pwm_scores_h5,
                        dataset_keys[i],
                        cluster_key, i)

        # refine - remove small clusters
        refined_cluster_key = "task-clusters-refined"
        #if refined_cluster_key not in h5py.File(pwm_scores_h5, "r").keys():
        if False:
        #if True:
            refine_clusters(pwm_scores_h5, cluster_key, refined_cluster_key)
            if visualize:
                for i in xrange(len(dataset_keys)):
                    visualize_clusters(
                        pwm_scores_h5,
                        dataset_keys[i],
                        refined_cluster_key, i,
                        remove_final_cluster=1)
            
        # 2) optional - correlation matrix.
        correlations_key = "task-pwm_x_pwm-correlations"
        if correlations_key not in h5py.File(pwm_scores_h5, "r").keys():
            # TODO
            pass
    
        # 3) enumerate metaclusters. dont visualize here because you must refine first
        # HERE - need to figure out a better way to metacluster.
        # enumeration is losing things that are dynamic
        metacluster_key = "metaclusters"
        if metacluster_key not in h5py.File(pwm_scores_h5, "r").keys():
        #if True:
            generate_simple_metaclusters(pwm_scores_h5, dataset_keys, metacluster_key)
        
        #if metacluster_key not in h5py.File(pwm_scores_h5, "r").keys():
        #if True:
        #    enumerate_metaclusters(pwm_scores_h5, cluster_key, metacluster_key)

        # refine - remove small clusters
        # TODO - put out BED files - write a separate function to pull BED from cluster set
        refined_metacluster_key = "metaclusters-refined"
        #if refined_metacluster_key not in h5py.File(pwm_scores_h5, "r").keys():
        if True:
            #refine_clusters(pwm_scores_h5, metacluster_key, refined_metacluster_key, null_cluster_present=False)
            if visualize:
                for i in xrange(len(dataset_keys)):
                    visualize_clusters(
                        pwm_scores_h5,
                        dataset_keys[i],
                        refined_metacluster_key, 0,
                        remove_final_cluster=1)

        # TODO
        # get the manifold descriptions out per cluster
        # ie the manifold is {task, pwm}, {task, threshold}
        # so per cluster, get the pwm mask and threshold and save to an hdf5 file
        manifold_key = "motifspace-centers"
        manifold_h5_file = "{0}/{1}.manifold.h5".format(
            args.tmp_dir, args.prefix)
        if not os.path.isfile(manifold_h5_file):
            get_manifold_centers(
                pwm_scores_h5,
                dataset_keys,
                refined_metacluster_key,
                manifold_h5_file,
                pwm_list,
                pwm_dict)

        quit()
        
        # TODO generate a dataset that is just the called pwms
        from tronn.interpretation.clustering import aggregate_pwm_results
        aggregate_pwm_results(results_h5_file, dataset_keys, manifold_h5_file)

        # and plot
        

        quit()
                    
        # 4) extract the constrained motif set for each metacommunity, for each task
        # new pwm vectors for each dataset..
        # save out initial grammar file, use labels to set a threshold
        # if visualize - save out mean vector and plot, also network vis?
        metacluster_motifs_key = "metaclusters-motifs"
        motifset_metacluster_key = "metaclusters-motifset-refined"
        if True:
        #if metacluster_motifs_key not in h5py.File(pwm_scores_h5, "r").keys():
            distill_to_linear_models(
                pwm_scores_h5,
                dataset_keys,
                refined_metacluster_key,
                motifset_metacluster_key,
                metacluster_motifs_key,
                pwm_list, pwm_dict,
                pwm_file=args.pwm_file,
                label_indices=args.inference_tasks) # eventually shouldnt need this, access name
            # or access some kind of attribute
            if visualize:
                for i in xrange(len(dataset_keys)):
                    visualize_clusters(
                        pwm_scores_h5,
                        dataset_keys[i],
                        motifset_metacluster_key, 0,
                        remove_final_cluster=1)

        # 5) optional - separately, get metrics on all tasks and save out (AUPRC, etc)
        # think of as a giant confusion matrix


    return None

