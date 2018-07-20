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
from tronn.datalayer import BedDataLoader
from tronn.nets.nets import net_fns

from tronn.interpretation.motifs import select_pwms
from tronn.interpretation.motifs import select_pwms_by_cluster

from tronn.interpretation.clustering import cluster_dataset

from tronn.interpretation.clustering import generate_simple_metaclusters
from tronn.interpretation.clustering import refine_clusters
from tronn.interpretation.clustering import aggregate_pwm_results
from tronn.interpretation.clustering import get_manifold_centers

from tronn.interpretation.clustering import aggregate_pwm_results
from tronn.interpretation.clustering import aggregate_pwm_results_per_cluster

#from tronn.visualization import visualize_clustered_h5_dataset_full
#from tronn.visualization import visualize_aggregated_h5_datasets
#from tronn.visualization import visualize_datasets_by_cluster_map
from tronn.visualization import visualize_h5_dataset
from tronn.visualization import visualize_h5_dataset_by_cluster

from tronn.visualization import visualize_clustering_results

from tronn.interpretation.grammars import make_bed

from tronn.util.h5_utils import add_pwm_names_to_h5

from tronn.util.utils import DataKeys


def _get_cluster_bed(h5_file, cluster_key, metadata_key, soft_clustering=False):
    """
    """
    prefix = h5_file.split(".h5")[0]
    
    with h5py.File(h5_file, "r") as hf:
        clusters = hf[cluster_key][:,0]
        cluster_ids = sorted(list(set(clusters.tolist())))


    for cluster_idx in xrange(len(cluster_ids)):
        cluster_id = cluster_ids[cluster_idx]
        in_cluster = clusters == cluster_id
    
        with h5py.File(h5_file, "r") as hf:
            metadata = hf["example_metadata"][:][in_cluster]
        cluster_prefix = "{0}.cluster-{1}".format(prefix, cluster_id)
        metadata_file = "{}.metadata.txt".format(cluster_prefix)
        metadata_bed = "{}.bed".format(cluster_prefix)
        make_bed(metadata, metadata_file, metadata_bed)

    
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
        
    # set up dataloader and input function
    if args.data_dir is not None:
        data_files = glob.glob('{}/*.h5'.format(args.data_dir))
        data_files = [h5_file for h5_file in data_files if "negative" not in h5_file]
        logger.info("Found {} chrom files".format(len(data_files)))
        dataloader = H5DataLoader(data_files, fasta=args.fasta)
        input_fn = dataloader.build_input_fn(
            args.batch_size,
            label_keys=args.model_info["label_keys"],
            filter_tasks=args.filter_tasks,
            singleton_filter_tasks=args.inference_task_indices)
        
    elif args.bed_input is not None:
        dataloader = BedDataLoader(args.bed_input, args.fasta)
        input_fn = dataloader.build_input_fn(
            args.batch_size,
            label_keys=args.model_info["label_keys"])

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
            # TODO can we clean this up?
            "model_fn": net_fns[args.model_info["name"]],
            "num_tasks": args.model_info["params"]["num_tasks"],
            "use_filtering": False if args.bed_input is not None else True, # TODO do this better
            "backprop": args.backprop, # change this to importance_method
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
            args.sample_size,
            debug=args.debug)

        # add in PWM names to the datasets
        add_pwm_names_to_h5(
            results_h5_file,
            [pwm.name for pwm in args.pwm_list])
                        
    # now run clustering
    if args.cluster and not args.debug:
        visualize = True # TODO adjust later
        logging.info("running clustering - louvain (Phenograph)")

        # cluster
        # TODO try hidden layer again sometime
        if DataKeys.CLUST_FILT not in h5py.File(results_h5_file, "r").keys():
            cluster_dataset(results_h5_file, DataKeys.FEATURES)
        
        # visualize in R
        # TODO adjust for -1 vals
        visualize = False
        if visualize:
            visualize_clustering_results(
                results_h5_file,
                DataKeys.CLUST_FILT,
                args.inference_task_indices,
                args.visualize_task_indices,
                args.visualize_signals,
                remove_final_cluster=False if args.bed_input else True)

            # TODO there's gotta be a better place to put this
            _get_cluster_bed(
                results_h5_file,
                DataKeys.CLUST_FILT,
                DataKeys.SEQ_METADATA,
                soft_clustering=False)
            
        # TODO calling significant pwms per cluster
        #select_pwms(results_h5_file, args.pwm_list, "test")
        select_pwms_by_cluster(
            results_h5_file,
            args.pwm_list,
            "cluster_pwms",
            cluster_key=DataKeys.CLUST_FILT)

        quit()
        
        dataset_keys = [
            "pwm-scores.taskidx-{}".format(i)
            for i in args.inference_task_indices]

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
            # TODO check this again using importance scores (compare to median cluster scores, NOT mean)

            # also rename - get_manifold_metrics()
            
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

