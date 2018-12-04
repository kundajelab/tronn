# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import json
import h5py
import glob
import logging

from tronn.datalayer import H5DataLoader

from tronn.interpretation.inference import run_inference
from tronn.interpretation.inference import run_multi_model_inference

from tronn.interpretation.clustering import run_clustering
from tronn.interpretation.clustering import summarize_clusters_on_manifold
from tronn.interpretation.clustering import get_cluster_bed_files
from tronn.interpretation.clustering import visualize_clustered_features_R
from tronn.interpretation.clustering import visualize_clustered_outputs_R
from tronn.interpretation.clustering import visualize_multikey_outputs_R

#from tronn.interpretation.motifs import extract_significant_pwms # TODO DEPRECATE
#from tronn.interpretation.motifs import run_bootstrap_differential_score_test
#from tronn.interpretation.motifs import test_differential_motifs
from tronn.interpretation.motifs import visualize_significant_pwms_R

from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.h5_utils import copy_h5_datasets
from tronn.util.formats import write_to_json
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

    # run all files together or run rotation of models
    if args.model["name"] == "kfold_models":
        inference_files = run_multi_model_inference(args, positives_only=True)
    else:
        inference_files = run_inference(args, positives_only=True)

    # put the files into a dataloader
    results_data_log = "{}/dataset.{}.json".format(args.out_dir, args.subcommand_name)
    results_data_loader = H5DataLoader(
        data_dir=args.out_dir, data_files=inference_files, fasta=args.fasta)
    write_to_json(results_data_loader.describe(), results_data_log)
    
    # add in PWM names to the datasets
    for inference_file in inference_files:
        add_pwm_names_to_h5(
            inference_file,
            [pwm.name for pwm in args.pwm_list],
            other_keys=[DataKeys.FEATURES])

    # save out relevant inference run details for downstream runs
    infer_log = "{}/infer.{}.json".format(args.out_dir, args.subcommand_name)
    infer_vals = {
        "infer_dir": args.out_dir,
        "model_json": args.model_json,
        "inference_targets": args.inference_targets,
        "pwm_file": args.pwm_file}
    write_to_json(infer_vals, infer_log)
    
    quit()

    # is this better as a standalone script?
    # if background file is given, then run differential test
    # against bootstrapped samples from background
    # TODO - one wrinkle here - what if you want to look at across multiple foreground indices?
    # not a good way to do that here yet.
    # TODO - better thing to do might be to separate each, so that each thing in list is foreground
    # cmd line TRAJ_LABELS TRAJ_LABELS=1 TRAJ_LABELS=2:H3K27ac=0 TRAJ_LABELS=2,3::reduce_type=any
    # files    TRAJ_LABELS TRAJ_LABELS-1 TRAJ_LABELS-2.H3K27ac-0 TRAJ_LABELS-2-3.reduce_type-any
    if args.foreground_targets is not None:

        # TODO load background data
        background_data_loader = H5DataLoader(args.background_dataset)
        
        # run with nn weighted scores
        for i in xrange(len(args.foreground_targets)):
            run_bootstrap_differential_score_test(
                results_h5_file,
                args.background_scores, # TODO convert this to a background dataloader
                args.foreground_targets[i][0],
                args.foreground_targets[i][1],
                DataKeys.LABELS,
                args.inference_targets,
                qval_thresh=0.05)
    
        # run with raw hits (HOMER style)
        for i in xrange(len(args.foreground_targets)):
            run_bootstrap_differential_score_test(
                results_h5_file,
                args.background_scores,
                args.foreground_targets[i][0],
                args.foreground_targets[i][1],
                DataKeys.LABELS,
                args.inference_targets,
                pwm_hits_key=DataKeys.ORIG_SEQ_PWM_SCORES_SUM,
                qval_thresh=0.05,
                out_key="pwms.hits.differential")
        
            
            
    quit()
    
    
        
    # run clustering analysis?
    # TODO probably deprecate this
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

    # TODO deprecate this also
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

