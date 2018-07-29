# description: scan for grammar scores

import os
import h5py
import glob
import logging

import numpy as np

from tronn.datalayer import H5DataLoader
from tronn.datalayer import BedDataLoader

from tronn.graphs import ModelManager

from tronn.interpretation.clustering import get_cluster_bed_files
from tronn.interpretation.clustering import visualize_clustered_features_R
from tronn.interpretation.clustering import visualize_clustered_outputs_R

from tronn.interpretation.motifs import extract_significant_pwms
from tronn.interpretation.motifs import visualize_significant_pwms_R

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

    # TODO need to better handle input of one file
    # set up dataloader and input function
    if args.data_dir is not None:
        data_files = glob.glob('{}/*.h5'.format(args.data_dir))
        data_files = [h5_file for h5_file in data_files if "negative" not in h5_file]
        data_files = [h5_file for h5_file in data_files if "manifold" not in h5_file]
        logging.info("Found {} chrom files".format(len(data_files)))
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
    if args.processed_inputs:
        model_fn = net_fns["empty_net"]
        reuse = False
    else:
        model_fn = net_fns[args.model_info["name"]]
        reuse = True
    
    model_manager = ModelManager(
        model_fn,
        args.model_info["params"])

    # set up inference generator
    inference_generator = model_manager.infer(
        input_fn,
        args.out_dir,
        net_fns[args.inference_fn],
        inference_params={
            # TODO can we clean this up?
            "model_fn": net_fns[args.model_info["name"]],
            "model_reuse": reuse,
            "num_tasks": args.model_info["params"]["num_tasks"],
            "backprop": args.backprop,
            "importances_fn": args.backprop, # TODO fix this
            "importance_task_indices": args.inference_task_indices,
            "pwms": args.pwm_list,
            "manifold": args.manifold_file},
        checkpoint=args.model_info["checkpoint"],
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

    # TODO some issue in dmim
        
    # outputs:
    # mut effects - {N, mutM, task, M} - this is a delta
    # delta logits - {N, mutM, logit}, partner with logits {N, logit}
    with h5py.File(results_h5_file, "r") as hf:
        true_logits = np.expand_dims(hf[DataKeys.LOGITS][:], axis=1)
        mut_logits = hf[DataKeys.MUT_MOTIF_LOGITS][:]
        delta_logits = np.subtract(mut_logits, true_logits)
        sig_delta_logits = run_delta_permutation_test(delta_logits)

    import ipdb
    ipdb.set_trace()
    
    
    # calculate OUTPUT effects {N, mutM, logit} vs {N, logit}
    # for each mutation, for each logit, permutation test
    # make a sig mask.
    # plot significant ones - box/whisker, and aggregate plot?
    # for each mutation, {N, logit} vs {N, logit}, subtract to get {N, delta_logit}
    # use randint to randomly flip the sign, and subtract each from each
    # and save out summed result
    
    
    # calculate SYNERGY effects {N, mutM, task, M}
    # for each mutation, for each task, calculate permutation test
    # and plot significant ones



    

    # apply a test on the difference between two distributions (logit vs mutate logit)
    # ie shuffle the labels (keeping the pairs) and then recalc difference
    # and also plot out scatter plots for the sig mutants (sig from mask vector)

    # also apply this to delta motif scores?
    # here the shuffle labels is basically flip the sign
    
    visualize = True
    visualize_task_indices = [args.inference_task_indices] + args.visualize_task_indices
    if visualize:
        _visualize_mut_results(
            results_h5_file,
            "pwm-scores.agg",
            "delta_logits.agg",
            "dmim-scores.agg.mut_only",
            visualize_task_indices,
            "pwm_names",
            "mut_pwm_names")
    
    return None
