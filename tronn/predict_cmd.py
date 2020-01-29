"""Contains code to run predictions
"""

import os
import json
import h5py
import logging

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager
from tronn.interpretation.inference import run_inference
from tronn.util.utils import DataKeys

def _setup_input_skip_keys():
    """
    """
    skip_keys = [
        "features",
        DataKeys.ORIG_SEQ,
        DataKeys.ORIG_SEQ_SHUF,
        DataKeys.ORIG_SEQ_ACTIVE,
        DataKeys.ORIG_SEQ_ACTIVE_SHUF,
        DataKeys.ORIG_SEQ_PWM_HITS,
        DataKeys.ORIG_SEQ_PWM_SCORES,
        DataKeys.ORIG_SEQ_PWM_SCORES_SUM,
        DataKeys.ORIG_SEQ_PWM_SCORES_THRESH,
        DataKeys.ORIG_SEQ_SHUF_PWM_SCORES,
        DataKeys.ORIG_SEQ_PWM_DENSITIES,
        DataKeys.ORIG_SEQ_PWM_MAX_DENSITIES,
        DataKeys.IMPORTANCE_GRADIENTS,
        DataKeys.WEIGHTED_SEQ,
        DataKeys.WEIGHTED_SEQ_SHUF,
        DataKeys.WEIGHTED_SEQ_ACTIVE,
        DataKeys.WEIGHTED_SEQ_ACTIVE_CI,
        DataKeys.WEIGHTED_SEQ_ACTIVE_CI_THRESH,
        DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
        DataKeys.WEIGHTED_SEQ_PWM_HITS,
        DataKeys.WEIGHTED_SEQ_PWM_SCORES,
        DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH,
        DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX,
        DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL,
        DataKeys.WEIGHTED_SEQ_SHUF_PWM_SCORES,
        DataKeys.MUT_MOTIF_ORIG_SEQ,
        "{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ),
        DataKeys.MUT_MOTIF_WEIGHTED_SEQ,
        DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT,
        DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT,
        DataKeys.MUT_MOTIF_POS,
        DataKeys.MUT_MOTIF_PRESENT,
        DataKeys.MUT_MOTIF_WEIGHTED_SEQ_CI,
        DataKeys.MUT_MOTIF_WEIGHTED_SEQ_CI_THRESH,
        DataKeys.MUT_MOTIF_LOGITS,
        DataKeys.MUT_MOTIF_LOGITS_SIG,
        DataKeys.MUT_MOTIF_LOGITS_MULTIMODEL,
        DataKeys.DFIM_SCORES,
        DataKeys.DFIM_SCORES_DX,
        DataKeys.DMIM_SCORES,
        DataKeys.DMIM_SCORES_SIG,
        DataKeys.FEATURES,
        "final_hidden",
        "logits.multimodel"]
    
    return skip_keys


def _setup_skip_output_keys():
    """
    """
    skip_keys = [
        "features",
        "final_hidden"]
    
    return skip_keys


def run(args):
    """cmd to run predictions
    """

    # FOR AITAC
    # setup args.inference_params
    args.inference_params = {
        "ablate_filter_idx": args.ablate_filter_idx,
        "skip_outputs": _setup_skip_output_keys()}
    
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running predictions")
    os.system("mkdir -p {}".format(args.out_dir))
    
    # collect a prediction sample if ensemble (for cross model quantile norm)
    # always need to do this if you're repeating backprop
    if args.model["name"] == "ensemble":
        if args.prediction_sample is None:
            true_sample_size = args.sample_size
            args.sample_size = 1000
            run_inference(args, warm_start=True)
            args.sample_size = true_sample_size

        # attach prediction sample to model
        args.model["params"]["prediction_sample"] = args.prediction_sample
    
    # set up model
    model_manager = setup_model_manager(args)
    
    # set up data loader
    data_loader = setup_data_loader(args)
    input_fn = data_loader.build_input_fn(
        args.batch_size,
        shuffle=not args.fifo if args.fifo is not None else True,
        targets=args.targets,
        target_indices=args.target_indices,
        filter_targets=args.filter_targets,
        use_queues=True,
        skip_keys=_setup_input_skip_keys())

    # predict
    predictor = model_manager.predict(
        input_fn,
        args.out_dir,
        inference_params=args.inference_params,
        checkpoint=model_manager.model_checkpoint)

    # run predictions and save to h5
    predictions_h5_file = "{}/{}.predictions.h5".format(args.out_dir, args.prefix)
    if not os.path.isfile(predictions_h5_file):
        model_manager.infer_and_save_to_h5(
            predictor, predictions_h5_file, args.num_evals)

    return None
