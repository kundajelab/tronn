# description: scan for grammar scores

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import networkx as nx

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager

from tronn.interpretation.combinatorial import setup_combinations

from tronn.util.h5_utils import AttrKeys
from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.h5_utils import copy_h5_datasets

from tronn.util.utils import DataKeys


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

    # set up pwm names
    sig_indices = np.where(sig_pwms != 0)[0].tolist()
    sig_pwms_names = []
    for sig_index in sig_indices:
        sig_pwms_names.append(args.pwm_names[sig_index])
        
    # save out names
    sig_pwms_ordered_file = "{}/{}.synergy.pwms.order.txt".format(
        args.out_dir, args.prefix)
    with open(sig_pwms_ordered_file, "w") as fp:
        fp.write("# ordered list of pwms used\n")
        sig_indices = np.where(sig_pwms != 0)[0].tolist()
        for sig_pwm_name in sig_pwms_names:
            fp.write("{}\n".format(sig_pwm_name))

    # set up combinatorial matrix and save out
    num_sig_pwms = int(np.sum(sig_pwms != 0))
    combinations = setup_combinations(num_sig_pwms)
    args.inference_params.update({"combinations": combinations})
    combinations_file = "{}/{}.synergy.combinations.txt".format(
        args.out_dir, args.prefix)
    combinations_df = pd.DataFrame(np.transpose(1 - combinations).astype(int), columns=sig_pwms_names)
    combinations_df.to_csv(combinations_file, sep="\t")

    # set up inference generator
    inference_generator = input_model_manager.infer(
        input_fn,
        args.out_dir,
        args.inference_params,
        checkpoint=model_manager.model_checkpoint,
        yield_single_examples=True)
    
    # run inference and save out
    results_h5_file = "{0}/{1}.synergy.h5".format(
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

    # and also attach the sig pwm names to the features
    with h5py.File(results_h5_file, "a") as hf:
        hf[DataKeys.FEATURES].attrs[AttrKeys.PWM_NAMES] = sig_pwms_names
        hf[DataKeys.MUT_MOTIF_LOGITS].attrs[AttrKeys.PWM_NAMES] = sig_pwms_names
    
    return None
