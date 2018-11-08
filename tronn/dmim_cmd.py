# description: scan for grammar scores

import os
import h5py
import glob
import logging

import numpy as np

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager

from tronn.interpretation.clustering import get_cluster_bed_files
from tronn.interpretation.clustering import visualize_clustered_features_R
from tronn.interpretation.clustering import visualize_clustered_outputs_R

from tronn.interpretation.motifs import get_sig_pwm_vector
from tronn.interpretation.motifs import extract_significant_pwms
from tronn.interpretation.motifs import visualize_significant_pwms_R

from tronn.interpretation.networks import get_motif_hierarchies

from tronn.interpretation.variants import get_significant_delta_logit_responses
from tronn.interpretation.variants import get_interacting_motifs
from tronn.interpretation.variants import visualize_interacting_motifs_R

from tronn.nets.nets import net_fns

from tronn.stats.nonparametric import run_delta_permutation_test

from tronn.util.h5_utils import AttrKeys
from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.h5_utils import copy_h5_datasets

from tronn.util.utils import DataKeys

# TODO clean this up
from tronn.visualization import visualize_agg_pwm_results
from tronn.visualization import visualize_agg_delta_logit_results
from tronn.visualization import visualize_agg_dmim_adjacency_results


# TODO move this out
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

    # set up dataloader and input fn
    data_loader = setup_data_loader(args)
    data_loader = data_loader.setup_positives_only_dataloader()
    input_fn = data_loader.build_input_fn(
        args.batch_size,
        targets=args.targets,
        target_indices=args.target_indices,
        filter_targets=args.filter_targets,
        singleton_filter_targets=args.singleton_filter_targets,
        use_queues=True,
        shuffle=False)
        
    # set up model
    model_manager = setup_model_manager(args)
    if args.processed_inputs:
        args.model["name"] = "empty_net"
        reuse = False
    else:
        reuse = True
    input_model_manager = setup_model_manager(args)

    # set up sig pwms
    if args.sig_pwms_file is None:
        args.sig_pwms_file = data_loader.data_files[0]
    sig_pwms = get_sig_pwm_vector(
        args.sig_pwms_file,
        args.sig_pwms_key,
        args.foreground_targets[0][0],
        args.foreground_targets[0][1],
        reduce_type="any")
    logging.info("Loaded {} pwms to perturb".format(np.sum(sig_pwms)))
    
    # set up inference generator
    inference_generator = input_model_manager.infer(
        input_fn,
        args.out_dir,
        net_fns[args.inference_fn],
        inference_params={
            # TODO can we clean this up?
            "model_fn": model_manager.model_fn,
            "model_reuse": reuse,
            "num_tasks": args.model["params"]["num_tasks"],
            "backprop": args.backprop,
            "importances_fn": args.backprop, # TODO fix this
            "importance_task_indices": args.inference_targets,
            "pwms": args.pwm_list,
            "sig_pwms": sig_pwms)
            #"manifold": args.manifold_file}, # <- change this
        checkpoint=model_manager.model_checkpoint,
        yield_single_examples=True)

    # run inference and save out
    results_h5_file = "{0}/{1}.dmim_results.h5".format(
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
            other_keys=[])

        # use copy_h5_datasets
        copy_h5_datasets(
            args.manifold_file,
            results_h5_file,
            keys=[
                DataKeys.MANIFOLD_PWM_SIG_CLUST,
                DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL])

        # TODO set up attr for delta logits and clusters
        with h5py.File(args.manifold_file, "r") as hf:
            with h5py.File(results_h5_file, "a") as out:
                num_clusters = out[DataKeys.MANIFOLD_CLUST].shape[1]
                out[DataKeys.MANIFOLD_CLUST].attrs[AttrKeys.CLUSTER_IDS] = range(num_clusters)
            
    # check manifold - was it consistent
    # TODO make an arg
    check_manifold = False
    if check_manifold:
        manifold_file_prefix = "{0}/{1}.{2}".format(
            args.out_dir, args.prefix, DataKeys.MANIFOLD_ROOT)
        
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

    args.visualize_R = [] # TODO remove
    if len(args.visualize_R) > 0:
        visualize_clustered_features_R(
            results_h5_file,
            data_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
            clusters_key=DataKeys.MANIFOLD_CLUST)
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

    # DMIM ANALYSES
    if False:
        get_interacting_motifs(
            results_h5_file,
            DataKeys.MANIFOLD_CLUST,
            DataKeys.DMIM_SIG_RESULTS)

    # and plot these out with R
    if False:
        visualize_interacting_motifs_R(
            results_h5_file,
            DataKeys.DMIM_SIG_RESULTS)

    # TODO also run sig analysis on the logit output? how to integrate this information?

    
    # and then build hierarcy
    paths = get_motif_hierarchies(
        results_h5_file,
        DataKeys.DMIM_SIG_RESULTS,
        DataKeys.FEATURES,
        extra_keys = [
            "ATAC_SIGNAL.NORM",
            "H3K27ac_SIGNAL.NORM",
            "H3K4me1_SIGNAL.NORM"])

    # TODO figure out how to save out paths (as vectors? adjacency?)
    # to be able to run the synergy calculations
    
    # TODO write another function to extract the top, to do MPRA
    
    
    # TODO still do this to show that all motifs have sig effects
    #get_significant_delta_logit_responses(
    #    results_h5_file, DataKeys.MANIFOLD_CLUST)
    
    return None
