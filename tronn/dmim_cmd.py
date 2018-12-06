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
from tronn.interpretation.motifs import copy_sig_pwm_vectors_to_h5
from tronn.interpretation.motifs import extract_significant_pwms
from tronn.interpretation.motifs import visualize_significant_pwms_R

from tronn.interpretation.networks import get_motif_hierarchies

from tronn.interpretation.variants import get_significant_delta_logit_responses
from tronn.interpretation.variants import get_interacting_motifs
from tronn.interpretation.variants import run_permutation_dmim_score_test
from tronn.interpretation.variants import visualize_interacting_motifs_R

from tronn.stats.nonparametric import run_delta_permutation_test

from tronn.util.h5_utils import AttrKeys
from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.h5_utils import copy_h5_datasets
from tronn.util.scripts import parse_multi_target_selection_strings
from tronn.util.utils import DataKeys


# TODO clean this up
from tronn.visualization import visualize_agg_pwm_results
from tronn.visualization import visualize_agg_delta_logit_results
from tronn.visualization import visualize_agg_dmim_adjacency_results


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
    args.filter_targets = parse_multi_target_selection_strings(
        args.foreground_targets)
    
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
