# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import json
import h5py
import glob
import logging

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager

from tronn.interpretation.clustering import run_clustering
from tronn.interpretation.clustering import summarize_clusters_on_manifold
from tronn.interpretation.clustering import get_cluster_bed_files
from tronn.interpretation.clustering import visualize_clustered_features_R
from tronn.interpretation.clustering import visualize_clustered_outputs_R
from tronn.interpretation.clustering import visualize_multikey_outputs_R

from tronn.interpretation.motifs import extract_significant_pwms
from tronn.interpretation.motifs import run_hypergeometric_test_on_motif_hits
from tronn.interpretation.motifs import run_bootstrap_differential_score_test
from tronn.interpretation.motifs import threshold_and_save_pwms
from tronn.interpretation.motifs import visualize_significant_pwms_R

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

    # set up dataloader and input fn
    data_loader = setup_data_loader(args)
    data_loader = data_loader.setup_positives_only_dataloader()
    input_fn = data_loader.build_input_fn(
        args.batch_size,
        targets=args.targets,
        target_indices=args.target_indices,
        filter_targets=args.filter_targets,
        singleton_filter_targets=args.singleton_filter_targets,
        use_queues=True)

    # set up model
    model_manager = setup_model_manager(args)

    # add model to inference params
    args.inference_params.update({"model": model_manager})
    
    # set up inference generator
    inference_generator = model_manager.infer(
        input_fn,
        args.out_dir,
        args.inference_params,
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

    # somewhere here need to run another file of negatives?
    # a set of regions that are NOT positive in the desired set
    # and match the number of regions that were extracted...
    # add an arg to reuse a background if it was already generated
    # --background_targets ATAC_LABELS=3,4,5,6
    # background_sample_size = 4*sample_size
    background_h5_file = "{0}/{1}.background.h5".format(
        args.out_dir, args.prefix)
    if not os.path.isfile(background_h5_file):
        # set up new input fn
        input_fn = data_loader.build_input_fn(
            args.batch_size,
            targets=args.targets,
            target_indices=args.target_indices,
            filter_targets=args.background_targets + args.background_filter_targets,
            singleton_filter_targets=args.singleton_filter_targets,
            use_queues=True)

        # set up inference generator
        inference_generator = model_manager.infer(
            input_fn,
            args.out_dir,
            args.inference_params,
            checkpoint=model_manager.model_checkpoint,
            yield_single_examples=True)

        # determine desired sample size
        with h5py.File(results_h5_file, "r") as hf:
            background_sample_size = args.background_sample_multiplier * hf[DataKeys.FEATURES].shape[0]
        
        # infer
        model_manager.infer_and_save_to_h5(
            inference_generator,
            background_h5_file,
            background_sample_size,
            debug=args.debug)

        # add in PWM names to the datasets
        add_pwm_names_to_h5(
            background_h5_file,
            [pwm.name for pwm in args.pwm_list],
            other_keys=[DataKeys.FEATURES])

    # then bootstrap from the background set (GC matched) to get
    # probability that summed motif hits in foreground is due to random chance
    if False:
        run_hypergeometric_test_on_motif_hits(
            results_h5_file,
            background_h5_file,
            "TRAJ_LABELS",
            #[0,7,8,9,10,11])
            [12,13,14,1])

    if True:
        for i in xrange(len(args.foreground_targets)):

            from tronn.interpretation.motifs import save_subset_patterns_to_txt

            if True:
                save_subset_patterns_to_txt(
                    results_h5_file,
                    args.foreground_targets[i])
            
            # get sig pwms
            pvals = run_bootstrap_differential_score_test(
                results_h5_file,
                background_h5_file,
                args.foreground_targets[i][0],
                args.foreground_targets[i][1],
                args.background_targets[i][0],
                args.background_targets[i][1])

            import ipdb
            ipdb.set_trace()
            
            for traj_i in xrange(pvals.shape[0]):
                out_pwm_file = "{}/{}.{}-{}.pwms.txt".format(
                    args.out_dir,
                    args.prefix,
                    args.foreground_targets[i][0],
                    args.foreground_targets[i][1][traj_i])
                print out_pwm_file
                threshold_and_save_pwms(
                    pvals[traj_i],
                    args.pwm_list,
                    out_pwm_file,
                    pval_thresh=0.05)
            
            # save out
            import ipdb
            ipdb.set_trace()
    
    quit()
    

    # return a PWM file with just significant PWMs for further analysis.
    
        
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
    args.model["pwm_file"] = args.pwm_file
    args.model["inference_tasks"] = args.inference_tasks
    args.model["backprop"] = args.backprop
    with open("{}/model_info.json".format(args.out_dir), "w") as fp:
        json.dump(args.model, fp, sort_keys=True, indent=4)

    return None

