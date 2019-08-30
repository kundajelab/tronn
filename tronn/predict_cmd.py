"""Contains code to run predictions
"""

import os
import json
import h5py
import logging

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager
from tronn.interpretation.inference import run_inference


def run_prediction(args):
    """run prediction
    """


    return




def run(args):
    """cmd to run predictions
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running predictions")
    os.system("mkdir -p {}".format(args.out_dir))

    # TODO load data file to get pwm name
    
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
        targets=args.targets,
        target_indices=args.target_indices,
        filter_targets=args.filter_targets,
        use_queues=True)

    # predict
    predictor = model_manager.predict(
        input_fn,
        args.out_dir,
        checkpoint=model_manager.model_checkpoint)

    # run predictions and save to h5
    predictions_h5_file = "{}/{}.predictions.h5".format(args.out_dir, args.prefix)
    if not os.path.isfile(predictions_h5_file):
        model_manager.infer_and_save_to_h5(
            predictor, predictions_h5_file, args.num_evals)

    return None
