# description: scan for grammar scores

import os
import h5py
import glob
import logging

import numpy as np

from tronn.interpretation.inference import run_inference
from tronn.interpretation.inference import run_multi_model_inference
from tronn.interpretation.motifs import get_sig_pwm_vector
from tronn.util.h5_utils import add_pwm_names_to_h5
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
        
    # get a sig pwms vector
    sig_pwms = get_sig_pwm_vector(
        args.sig_pwms_file,
        args.sig_pwms_key,
        args.foreground_targets,
        reduce_type="any")

    # adjust filter targets based on foreground
    filter_targets = parse_multi_target_selection_strings(
        args.foreground_targets)
    new_filter_targets = []
    for keys_and_indices, params in filter_targets:
        new_filter_targets += keys_and_indices
    args.filter_targets = [(new_filter_targets, {"reduce_type": "any"})]

    
    # TODO add option to ignore long PWMs (later)
    args.inference_params.update({"sig_pwms": sig_pwms})
    logging.info("Loaded {} pwms to perturb".format(np.sum(sig_pwms)))
    
    # run all files together or run rotation of models
    if args.model["name"] == "kfold_models":
        run_multi_model_inference(args, positives_only=True)
    else:
        run_inference(args, positives_only=True)

    # add in PWM names to the datasets
    add_pwm_names_to_h5(
        results_h5_file,
        [pwm.name for pwm in args.pwm_list],
        other_keys=[DataKeys.FEATURES])

    return None
