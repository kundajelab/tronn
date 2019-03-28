"""Contains code to run predictions
"""

import os
import json
import h5py
import logging

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager


def run(args):
    """cmd to run predictions
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running predictions")
    os.system("mkdir -p {}".format(args.out_dir))

    # if ensemble, make sure prediction sample present
    
    
    # set up model
    model_manager = setup_model_manager(args)
    
    # set up data loader
    data_loader = setup_data_loader(args)
    input_fn = data_loader.build_input_fn(
        args.batch_size,
        targets=args.targets,
        target_indices=args.target_indices,
        filter_targets=args.filter_targets)

    # evaluate
    predictor = model_manager.predict(
        test_input_fn,
        args.out_dir,
        checkpoint=model_manager.model_checkpoint)

    # run predictions and save to h5
    predictions_h5_file = "{}/{}.predictions.h5".format(args.out_dir, args.prefix)
    if not os.path.isfile(predictions_h5_file):
        model_manager.infer_and_save_to_h5(
            predictor, predictions_h5_file, args.num_evals)

    return None
