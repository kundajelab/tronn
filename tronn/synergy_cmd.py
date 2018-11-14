# description: scan for grammar scores

import os
import h5py
import glob
import logging

import numpy as np
import networkx as nx

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager

from tronn.interpretation.clustering import get_cluster_bed_files
from tronn.interpretation.clustering import visualize_clustered_features_R
from tronn.interpretation.clustering import visualize_clustered_outputs_R

from tronn.interpretation.motifs import extract_significant_pwms
from tronn.interpretation.motifs import visualize_significant_pwms_R

from tronn.interpretation.networks import get_motif_hierarchies

from tronn.interpretation.variants import get_significant_delta_logit_responses
from tronn.interpretation.variants import get_interacting_motifs
from tronn.interpretation.variants import visualize_interacting_motifs_R

#from tronn.nets.nets import net_fns

from tronn.stats.nonparametric import run_delta_permutation_test

from tronn.util.h5_utils import AttrKeys
from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.h5_utils import copy_h5_datasets

from tronn.util.utils import DataKeys

# TODO clean this up
from tronn.visualization import visualize_agg_pwm_results
from tronn.visualization import visualize_agg_delta_logit_results
from tronn.visualization import visualize_agg_dmim_adjacency_results


def run(args):
    """run synergy analysis
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running synergy scan")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir

    # load in gml, extract example subset and pass to input fn
    grammar = nx.read_gml(args.grammar_file)
    grammar_examples = grammar.graph["examples"].split(",")
    
    # set up dataloader and input fn
    data_loader = setup_data_loader(args)
    data_loader = data_loader.setup_positives_only_dataloader()
    input_fn = data_loader.build_input_fn(
        args.batch_size,
        targets=args.targets,
        target_indices=args.target_indices,
        filter_targets=args.filter_targets,
        singleton_filter_targets=args.singleton_filter_targets,
        examples_subset=grammar_examples,
        use_queues=True,
        shuffle=False,
        skip_keys=[
            DataKeys.ORIG_SEQ_SHUF,
            DataKeys.ORIG_SEQ_ACTIVE_SHUF,
            #DataKeys.ORIG_SEQ_PWM_HITS,
            DataKeys.ORIG_SEQ_PWM_SCORES,
            DataKeys.ORIG_SEQ_PWM_SCORES_THRESH,
            DataKeys.ORIG_SEQ_SHUF_PWM_SCORES,
            DataKeys.WEIGHTED_SEQ_SHUF,
            DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
            DataKeys.WEIGHTED_SEQ_PWM_HITS,
            DataKeys.WEIGHTED_SEQ_PWM_SCORES,
            DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH,
            DataKeys.WEIGHTED_SEQ_SHUF_PWM_SCORES
        ]) # reduce the things being pulled out

    # set up model
    model_manager = setup_model_manager(args)
    args.inference_params.update({"model": model_manager})

    # check if processed inputs
    if args.processed_inputs:
        args.model["name"] = "empty_net"
        args.inference_params.update({"model_reuse": False})
    else:
        args.inference_params.update({"model_reuse": True})
    input_model_manager = setup_model_manager(args)

    # set up sig pwms
    sig_pwms = np.zeros((len(args.pwm_list)))
    sig_indices = nx.get_node_attributes(grammar, "responderidx")
    for pwm_key in sig_indices.keys():
        sig_pwms[sig_indices[pwm_key]] = 1
    args.inference_params.update({"sig_pwms": sig_pwms})
    logging.info("Loaded {} pwms to perturb".format(np.sum(sig_pwms)))

    # set up inference generator
    inference_generator = input_model_manager.infer(
        input_fn,
        args.out_dir,
        args.inference_params,
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

    quit()

    # TODO figure out how to save out paths (as vectors? adjacency?)
    # to be able to run the synergy calculations
    
    # TODO write another function to extract the top, to do MPRA
    
    
    # TODO still do this to show that all motifs have sig effects
    #get_significant_delta_logit_responses(
    #    results_h5_file, DataKeys.MANIFOLD_CLUST)
    
    return None
