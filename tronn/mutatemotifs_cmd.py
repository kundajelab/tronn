# description: scan for grammar scores

import os
import h5py
import glob
import logging

import numpy as np

from tronn.datalayer import H5DataLoader
from tronn.interpretation.inference import run_inference
from tronn.interpretation.motifs import get_sig_pwm_vector

from tronn.nets.preprocess_nets import mutate_sequences_single_motif
from tronn.nets.preprocess_nets import postprocess_mutate

from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.formats import write_to_json
from tronn.util.scripts import parse_multi_target_selection_strings
from tronn.util.utils import DataKeys


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

    # set up inference params
    args.inference_params = {
        "cmd_name": "mutatemotifs",
        "premodel_params": {
            "premodel_fn": mutate_sequences_single_motif},
        "mutate_type": args.mutate_type,
        "inference_fn_name": "postprocess_mutate",
        "use_filtering": True}
    args.debug = False
        
    # get a sig pwms vector
    sig_pwms = get_sig_pwm_vector(
        args.sig_pwms_file,
        args.sig_pwms_key,
        args.foreground_targets,
        reduce_type="any")
    args.inference_params.update({"sig_pwms": sig_pwms})
    logging.info("Loaded {} pwms to perturb".format(np.sum(sig_pwms)))
    
    # adjust filter targets based on foreground
    filter_targets = parse_multi_target_selection_strings(
        args.foreground_targets)
    new_filter_targets = []
    for keys_and_indices, params in filter_targets:
        new_filter_targets += keys_and_indices
    args.filter_targets += [(new_filter_targets, {"reduce_type": "any"})]

    # collect a prediction sample if ensemble (for cross model quantile norm)
    # always need to do this if you're repeating backprop
    if args.model["name"] == "ensemble":
        true_sample_size = args.sample_size
        args.sample_size = 1000
        run_inference(args, warm_start=True)
        args.sample_size = true_sample_size

        # attach prediction sample to model
        args.model["params"]["prediction_sample"] = args.prediction_sample
    
    # run inference
    inference_files = run_inference(args)
    
    # save out dataset json
    results_data_log = "{}/dataset.{}.json".format(args.out_dir, args.subcommand_name)
    results_data_loader = H5DataLoader(
        data_dir=args.out_dir, data_files=predictions_files, fasta=args.fasta)
    dataset = results_data_loader.describe()
    dataset.update({
        "targets": args.targets,
        "target_indices": args.target_indices})
    write_to_json(dataset, results_data_log)

    # save out inference json
    infer_log = "{}/infer.{}.json".format(args.out_dir, args.subcommand_name)
    infer_vals = {
        "infer_dir": args.out_dir,
        "model_json": args.model_json,
        "targets": args.targets,
        "inference_targets": args.inference_targets,
        "target_indices": args.target_indices,
        "pwm_file": args.pwm_file}
    write_to_json(infer_vals, infer_log)
    
    return None
