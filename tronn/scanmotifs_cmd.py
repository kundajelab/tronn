# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import json
import h5py
import glob
import logging

from tronn.datalayer import H5DataLoader
from tronn.datalayer import BedDataLoader

from tronn.models import ModelManager
from tronn.models import KerasModelManager

from tronn.interpretation.clustering import run_clustering
from tronn.interpretation.clustering import summarize_clusters_on_manifold
from tronn.interpretation.clustering import get_cluster_bed_files
from tronn.interpretation.clustering import visualize_clustered_features_R
from tronn.interpretation.clustering import visualize_clustered_outputs_R
from tronn.interpretation.clustering import visualize_multikey_outputs_R

from tronn.interpretation.motifs import extract_significant_pwms
from tronn.interpretation.motifs import visualize_significant_pwms_R

from tronn.nets.nets import net_fns

from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.h5_utils import copy_h5_datasets

from tronn.util.utils import DataKeys


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
    if args.model_info.get("model_type") == "keras":
        model_manager = KerasModelManager(
            keras_model_path=args.model_info["checkpoint"],
            model_params=args.model_info.get("params", {}),
            model_dir=args.out_dir)
    else:
        model_manager = ModelManager(
            net_fns[args.model_info["name"]],
            args.model_info["params"],
            model_checkpoint=args.model_info["checkpoint"])

    # set up inference generator
    inference_generator = model_manager.infer(
        input_fn,
        args.out_dir,
        net_fns[args.inference_fn],
        inference_params={
            # TODO can we clean this up?
            "model_fn": model_manager.model_fn,
            "num_tasks": args.model_info["params"]["num_tasks"],
            "use_filtering": False if args.bed_input is not None else True, # TODO do this better
            #"use_filtering": False,
            "backprop": args.backprop, # change this to importance_method
            "importance_task_indices": args.inference_task_indices,
            "pwms": args.pwm_list},
        checkpoint=model_manager.model_checkpoint,
        yield_single_examples=True)

    # run inference and save out
    # TODO change this to pwm_results
    results_h5_file = "{0}/{1}.inference.h5".format(
        args.out_dir, args.prefix)
    if not os.path.isfile(results_h5_file):
        model_manager.infer_and_save_to_h5(
            inference_generator,
            results_h5_file,
            args.sample_size,
            debug=args.debug)

        # add in PWM names to the datasets
        add_pwm_names_to_h5(
            results_h5_file,
            [pwm.name for pwm in args.pwm_list],
            other_keys=[DataKeys.FEATURES])
        
    # run clustering analysis
    if args.cluster and not args.debug:
        cluster_file_prefix = "{0}/{1}.{2}".format(
            args.out_dir, args.prefix, DataKeys.CLUSTERS)
        
        # cluster
        # TODO try hidden layer again sometime
        if DataKeys.CLUSTERS not in h5py.File(results_h5_file, "r").keys():
            logging.info("running clustering - louvain (Phenograph)")
            run_clustering(results_h5_file, DataKeys.FEATURES)
            get_cluster_bed_files(
                results_h5_file,
                cluster_file_prefix)
            extract_significant_pwms(results_h5_file, args.pwm_list)
            
        if len(args.visualize_R) > 0:
            visualize_clustered_features_R(
                results_h5_file)
            visualize_clustered_outputs_R(
                results_h5_file,
                args.visualize_R)
            visualize_significant_pwms_R(
                results_h5_file)
        if len(args.visualize_multikey_R) > 0:
            visualize_multikey_outputs_R(
                results_h5_file,
                args.visualize_multikey_R)

    # run manifold analysis
    # NOTE relies on cluster analysis
    if args.summarize_manifold and not args.debug:
        manifold_file_prefix = "{0}/{1}.{2}".format(
            args.out_dir, args.prefix, DataKeys.MANIFOLD_ROOT)

        # get the manifold descriptions out per cluster
        manifold_h5_file = "{0}/{1}.manifold.h5".format(
            args.out_dir, args.prefix)
        if not os.path.isfile(manifold_h5_file):

            summarize_clusters_on_manifold(
                results_h5_file)
            get_cluster_bed_files(
                results_h5_file,
                manifold_file_prefix,
                clusters_key=DataKeys.MANIFOLD_CLUST)
            extract_significant_pwms(
                results_h5_file,
                args.pwm_list,
                clusters_key=DataKeys.MANIFOLD_CLUST,
                pwm_sig_global_key=DataKeys.MANIFOLD_PWM_SIG_GLOBAL,
                pwm_scores_agg_global_key=DataKeys.MANIFOLD_PWM_SCORES_AGG_GLOBAL,
                pwm_sig_clusters_key=DataKeys.MANIFOLD_PWM_SIG_CLUST,
                pwm_sig_clusters_all_key=DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL,
                pwm_scores_agg_clusters_key=DataKeys.MANIFOLD_PWM_SCORES_AGG_CLUST)

            # copy over to (smaller) manifold file for ease of use
            copy_h5_datasets(
                results_h5_file,
                manifold_h5_file,
                keys=[
                    DataKeys.MANIFOLD_CENTERS,
                    DataKeys.MANIFOLD_THRESHOLDS,
                    DataKeys.MANIFOLD_PWM_SIG_CLUST,
                    DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL])
            
        if len(args.visualize_R) > 0:
            visualize_clustered_features_R(
                results_h5_file)
            visualize_clustered_outputs_R(
                results_h5_file,
                args.visualize_R)
            visualize_significant_pwms_R(
                results_h5_file,
                pwm_scores_agg_clusters_key=DataKeys.MANIFOLD_PWM_SCORES_AGG_CLUST)

        if len(args.visualize_multikey_R) > 0:
            visualize_multikey_outputs_R(
                results_h5_file,
                args.visualize_multikey_R)

    analyze_density = False
    if analyze_density and not args.debug:
        # motif density analysis here
        pass

    # save out additional info to model json
    args.model_info["pwm_file"] = args.pwm_file
    args.model_info["inference_tasks"] = args.inference_tasks
    args.model_info["backprop"] = args.backprop
    with open("{}/model_info.json".format(args.out_dir), "w") as fp:
        json.dump(args.model_info, fp, sort_keys=True, indent=4)

    return None

