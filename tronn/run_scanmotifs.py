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

    if False:
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
            checkpoint="blah", #args.model_checkpoints[0],
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
        if refined_metacluster_key not in h5py.File(pwm_scores_h5, "r").keys():
            refine_clusters(pwm_scores_h5, metacluster_key, refined_metacluster_key, null_cluster_present=False)
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

        # TODO also call significant motifs

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
        
        



    
    quit()
    
    if not args.no_groups:
        logger.info("Clustering regions per task.")
    
        # now for each timepoint task, go through and calculate communities
        for i in xrange(len(args.inference_tasks)):

            interpretation_task_idx = args.inference_tasks[i]
            logger.info("finding communities for task {}".format(interpretation_task_idx))

            # extract motif mat (region, motif) and save out to text file (to handle in R or python)
            region_x_pwm_mat_file = "{0}/{1}.task-{2}.region_x_pwm.txt".format(
                args.tmp_dir, args.prefix, interpretation_task_idx)
            logger.info("extracting region x pwm matrix")
            if not os.path.isfile(region_x_pwm_mat_file):
                h5_dataset_to_text_file(
                    pwm_scores_h5,
                    "pwm-scores.taskidx-{}".format(i), # use i because the ordering in the file is just 0-10
                    region_x_pwm_mat_file,
                    range(len(pwm_list)),
                    pwm_names)

            # get a sorted (ie clustered) version of the motif mat using phenograph (Louvain)
            # TODO somehow need to get the labels to exist with this file also....
            region_x_pwm_sorted_mat_file = "{0}.phenograph_sorted.txt".format(
                region_x_pwm_mat_file.split(".txt")[0])
            logger.info("using louvain communities to cluster regions")
            if not os.path.isfile(region_x_pwm_sorted_mat_file):
                phenograph_cluster(region_x_pwm_mat_file, region_x_pwm_sorted_mat_file)

            if visualize:
                # here, plot a example x pwm plot
                viz_file = "{0}/{1}.pdf".format(
                    viz_dir,
                    os.path.basename(region_x_pwm_sorted_mat_file).split(".txt")[0])
                plot_example_x_pwm = "plot.example_x_pwm.R {0} {1}".format(
                    region_x_pwm_sorted_mat_file, viz_file)
                print(plot_example_x_pwm)
                os.system(plot_example_x_pwm)
                
            # get the correlation matrix to look at, not really necessary?
            pwm_x_pwm_corr_file = "{0}/{1}.task-{2}.pwm_x_pwm.corr.mat.txt".format(
                args.tmp_dir, args.prefix, interpretation_task_idx)
            logger.info("get the correlation matrix")
            if not os.path.isfile(pwm_x_pwm_corr_file):
                get_correlation_file(
                    region_x_pwm_sorted_mat_file,
                    pwm_x_pwm_corr_file,
                    corr_method="continuous_jaccard")

            if visualize:
                # plot the correlation plot too
                viz_file = "{0}/{1}.pdf".format(
                    viz_dir,
                    os.path.basename(pwm_x_pwm_corr_file).split(".mat.txt")[0])
                plot_pwm_x_pwm = "plot.pwm_x_pwm.R {0} {1}".format(
                    pwm_x_pwm_corr_file, viz_file)
                print(plot_pwm_x_pwm)
                os.system(plot_pwm_x_pwm)

        # and then enumerate
        community_files = [
            "{0}/{1}.task-{2}.region_x_pwm.phenograph_sorted.txt".format(
                args.out_dir, args.prefix, i)
            for i in args.inference_tasks]

        # maybe put these into separate folders?
        # TODO figure out how to get labels in these h5 files too
        logger.info("enumerating metacommunities")
        metacommunity_files = sorted(
            glob.glob("{}/{}.metacommunity_*.h5".format(
                args.out_dir, args.prefix)))
        if len(metacommunity_files) == 0:
            enumerate_motifspace_communities(
                community_files,
                args.inference_tasks,
                "{}/{}".format(args.out_dir, args.prefix),
                pwm_list)
        metacommunity_files = sorted(
            glob.glob("{}/{}.metacommunity_*.h5".format(
                args.out_dir, args.prefix)))

        if visualize:
            # plot!
            pass

        # get the constrained motif set
        for i in xrange(len(metacommunity_files)):

            metacommunity_file = metacommunity_files[i]
            metacommunity_prefix = os.path.basename(metacommunity_file).split(".h5")[0]
            metacommunity_region_file = "{}.region_ids.refined.txt".format(metacommunity_file.split(".h5")[0])
            metacommunity_bed_file = "{}.bed".format(metacommunity_region_file.split(".txt")[0])
            metacommunity_regions = []
            metacommunity_pwms = []
            metacommunity_pwm_x_task_means_file = "{}.means.txt".format(
                metacommunity_region_file.split(".txt")[0])
            print metacommunity_file
            
            with h5py.File(metacommunity_file, "r") as hf:
                for task_idx in xrange(len(args.inference_tasks)):
                    inference_task_idx = args.inference_tasks[task_idx]

                    # get arrays
                    data_tmp = hf["features"][:,:,task_idx] # {N, pwm}
                    pwm_names = hf["pwm_names"][:]
                    regions = hf["example_metadata"][:]
                    
                    # threshold motifs                    
                    pwm_keep_indices = threshold_motifs(data_tmp)
                    data_tmp = data_tmp[:, pwm_keep_indices[0]]
                    pwm_names_thresh = pwm_names[pwm_keep_indices[0]]
                    
                    # reduce by motif similarity
                    if len(pwm_names) > 1:
                        task_pwms = [pwm_dict[pwm_name] for pwm_name in pwm_names_thresh]
                        pwm_subset = reduce_pwm_redundancy(task_pwms, pwm_dict, data_tmp)
                        pwm_subset = [pwm.name for pwm in pwm_subset]
                        pwm_keep_indices = np.where([True if pwm.name in pwm_subset else False
                                                     for pwm in task_pwms])
                        data_tmp = data_tmp[:, pwm_keep_indices[0]]
                        pwm_names = pwm_names[pwm_keep_indices[0]]
                    else:
                        pwm_subset = pwm_names
                    
                    # check set coverage
                    region_keep_indices = np.where(~np.any(data_tmp == 0, axis=1))
                    data_tmp = data_tmp[region_keep_indices[0],:]                    
                    regions = regions[region_keep_indices[0]]

                    # TODO - set up thresholds here. threshold at FDR 0.05?
                    # get back TPR and FPR
                    # now need a label matrix
                    global_pwm_keep_indices = np.where([True if pwm_name in pwm_subset else False
                                                        for pwm_name in pwm_names])
                    scores = np.sum(hf["features"][:,:,task_idx][:, global_pwm_keep_indices[0]], axis=1) # {N}
                    
                    
                    # save out a grammar file of the core PWMs
                    grammar_file = "{}.motifset.grammar".format(
                        metacommunity_file.split(".h5")[0])
                    node_dict = {}
                    for pwm in pwm_subset:
                        node_dict[pwm] = 1.0 # motifset, so all are equal
                    metacommunity_task_grammar = Grammar(
                        args.pwm_file,
                        node_dict,
                        {},
                        "taskidx={0};type=metacommunity;directed=no".format(task_idx),
                        "{0}.taskidx-{1}".format(
                            metacommunity_prefix,
                            inference_task_idx))
                    metacommunity_task_grammar.to_file(grammar_file)

                    # keep union of all regions
                    metacommunity_regions += regions.tolist()
                    metacommunity_regions = list(set(metacommunity_regions))

                    # keep union of all pwms
                    metacommunity_pwms += pwm_subset
                    metacommunity_pwms = list(set(metacommunity_pwms))
                    
                # use union of all regions and union of all pwms to get a pwm x task matrix
                pwm_x_task_scores = pd.DataFrame()
                for task_idx in xrange(len(args.inference_tasks)):
                    inference_task_idx = args.inference_tasks[task_idx]
                    
                    # get arrays into a dataframe
                    data_df = pd.DataFrame(
                        hf["features"][:,:,task_idx],
                        index=hf["example_metadata"][:],
                        columns=hf["pwm_names"][:])

                    # filter
                    data_df = data_df.loc[metacommunity_pwms, metacommunity_pwms]

                    # get the mean
                    data_mean = data_df.mean(axis=0)
                    
                    # append
                    pwm_x_task_scores = pwm_x_task_scores.append(
                        data_mean, ignore_index=True)

                # and save out
                pwm_x_task_scores = pwm_x_task_scores.fillna(0)
                pwm_x_task_scores.to_csv(metacommunity_pwm_x_task_means_file, sep="\t")
                
                # visualize
                if visualize:
                    viz_file = "{}.pdf".format(
                        metacommunity_pwm_x_task_means_file.split(".txt")[0])
                    plot_pwm_x_task = "plot.pwm_x_task.R {} {}".format(
                        metacommunity_pwm_x_task_means_file, viz_file)
                    print(plot_pwm_x_task)
                    os.system(plot_pwm_x_task)

                # write out master region set to file
                with open(metacommunity_region_file, "w") as out:
                    for region in metacommunity_regions:
                        out.write("{}\n".format(region))

                # convert to a bed file
                to_bed = (
                    "cat {0} | "
                    "awk -F ';' '{{ print $3 }}' | "
                    "awk -F '=' '{{ print $2 }}' | "
                    "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
                    "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
                    "sort -k1,1 -k2,2n | "
                    "bedtools merge -i stdin > "
                    "{1}").format(
                        metacommunity_region_file,
                        metacommunity_bed_file)
                print to_bed
                os.system(to_bed)

    return None

