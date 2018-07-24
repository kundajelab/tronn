# description: scan for grammar scores

import os
import h5py
import glob
import logging

from tronn.graphs import ModelManager
from tronn.datalayer import H5DataLoader
from tronn.datalayer import BedDataLoader
from tronn.nets.nets import net_fns

from tronn.interpretation.clustering import visualize_clustered_features_R
from tronn.interpretation.clustering import visualize_clustered_outputs_R
from tronn.interpretation.clustering import get_cluster_bed_files

from tronn.interpretation.motifs import extract_significant_pwms
from tronn.interpretation.motifs import visualize_significant_pwms_R

from tronn.visualization import visualize_agg_pwm_results
from tronn.visualization import visualize_agg_delta_logit_results
from tronn.visualization import visualize_agg_dmim_adjacency_results

from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.utils import DataKeys


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
        
        # TODO move to h5 utils
        # save out additional useful information
        with h5py.File(results_h5_file, "a") as hf:

            # save master pwm vector
            with h5py.File(args.manifold_file, "r") as manifold:
                #del hf["master_pwm_vector"]
                hf.create_dataset("master_pwm_vector", data=manifold["master_pwm_vector"][:])
                mut_indices = np.where(manifold["master_pwm_vector"][:] > 0)[0]
                pwm_mut_names = [args.pwm_list[i].name for i in mut_indices]
                
            # attach to delta logits and mutated scores
            hf["delta_logits"].attrs["pwm_mut_names"] = pwm_mut_names
            for task_idx in args.inference_task_indices:
                hf["dmim-scores.taskidx-{}".format(task_idx)].attrs["pwm_mut_names"] = pwm_mut_names


    # TODO - all the below stuff is basically
    # get bed files
    # get sig pwms (which aggregates results)
    # visualize R all
    check_manifold = False
    if check_manifold:
        manifold_file_prefix = "{0}/{1}.{2}".format(
            args.out_dir, args.prefix, DataKeys.MANIFOLD_ROOT)
        get_cluster_bed_files(
            results_h5_file,
            manifold_file_prefix)
        extract_significant_pwms(
            results_h5_file,
            args.pwm_list,
            clusters_key=DataKeys.MANIFOLD_CLUST,
            pwm_sig_global_key=DataKeys.MANIFOLD_PWM_SIG_GLOBAL,
            pwm_scores_agg_global_key=DataKeys.MANIFOLD_PWM_SCORES_AGG_GLOBAL,
            pwm_sig_clusters_key=DataKeys.MANIFOLD_PWM_SIG_CLUST,
            pwm_sig_clusters_all_key=DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL,
            pwm_scores_agg_clusters_key=DataKeys.MANIFOLD_PWM_SCORES_AGG_CLUST)

    visualize_R = False
    if visualize_R:
        visualize_clustered_features_R(
            results_h5_file,
            DataKeys.CLUSTERS)
        visualize_clustered_outputs_R(
            results_h5_file,
            DataKeys.CLUSTERS,
            args.visualize_tasks,
            args.visualize_signals)
        visualize_significant_pwms_R(
            results_h5_file)

    
        
    # TODO - here is dmim specific analyses
    # think about how to get significance on mutational shift.
    # {N, mut_motif, task, M}
    # {N, mut_motif, task} delta logits (save as logits?)
    # apply a test on the difference between two distributions (logit vs mutate logit)
    # ie shuffle the labels (keeping the pairs) and then recalc difference
    # and also plot out scatter plots for the sig mutants (sig from mask vector)

    # also apply this to delta motif scores?
    # here the shuffle labels is basically flip the sign
    # normalization here is still an issue. think about how to solve this
    # normalization - fit the noise?
    # normalize such that the drop in prediction is tied to how much importance was lost
    # in mut motif?
    
    # aggregate results
    dmim_keys = ["dmim-scores.taskidx-{}".format(i) for i in args.inference_task_indices]
    pwm_score_keys = ["pwm-scores.taskidx-{}".format(i) for i in args.inference_task_indices]
    if True:
        aggregate_dmim_results(
            results_h5_file,
            "manifold_clusters",
            args.inference_task_indices,
            dmim_keys,
            pwm_score_keys,
            args.pwm_list)

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
