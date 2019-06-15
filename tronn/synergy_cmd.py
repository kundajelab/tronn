# description: scan for grammar scores

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import networkx as nx

from tronn.datalayer import setup_data_loader
from tronn.interpretation.inference import run_inference

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
    args.dataset_examples = grammar.graph["examples"].split(",")
    logging.info("Running {} examples from grammar".format(len(args.dataset_examples)))
    
    # set up sig pwms
    sig_pwms = np.zeros((len(args.pwm_list)))
    sig_indices = nx.get_node_attributes(grammar, "pwmidx")
    for pwm_key in sig_indices.keys():
        sig_pwms[sig_indices[pwm_key]] = 1
    args.inference_params.update({"sig_pwms": sig_pwms})
    logging.info("Loaded {} pwms to perturb".format(np.sum(sig_pwms)))

    # set up sig pwm names
    sig_indices = np.where(sig_pwms != 0)[0].tolist()
    sig_pwms_names = []
    for sig_index in sig_indices:
        sig_pwms_names.append(args.pwm_names[sig_index])

    # adjustments if dataloader is simulated data
    if args.data_format == "pwm_sims":
        args.grammar_pwms = [args.pwm_list[i] for i in sig_indices]
        args.embedded_only = True
        
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

    # collect a prediction sample if ensemble (for cross model quantile norm)
    # always need to do this if you're repeating backprop
    if args.model["name"] == "ensemble":
        true_sample_size = args.sample_size
        args.sample_size = 1000
        run_inference(args, warm_start=True)
        args.sample_size = true_sample_size
    
    # run inference
    inference_files = run_inference(args)

    # add in PWM names to the datasets
    for inference_file in inference_files:
        add_pwm_names_to_h5(
            inference_file,
            [pwm.name for pwm in args.pwm_list],
            other_keys=[DataKeys.FEATURES])

    # attach sig pwm names
    with h5py.File(inference_file, "a") as hf:
        hf[DataKeys.FEATURES].attrs[AttrKeys.PWM_NAMES] = sig_pwms_names
        hf[DataKeys.MUT_MOTIF_LOGITS].attrs[AttrKeys.PWM_NAMES] = sig_pwms_names
    
    return None
