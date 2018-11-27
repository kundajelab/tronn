# description: some code to extract params to share with other frameworks

import os
import json
import logging

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager

from tronn.util.formats import write_to_json


def run(args):
    """Setup things and extract params
    """
    # set up
    logger = logging.getLogger(__name__)
    logger.info("Exporting model variables...")

    _BATCH_SIZE = 32 # placeholder

    # resave the model json
    model_log = "{}/model.json".format(args.out_dir)
    write_to_json(args.model, model_log)
    
    # set up dataloader
    data_loader = setup_data_loader(args)
    data_loader.data_files = [data_loader.data_files[0]]
    input_fn = data_loader.build_input_fn(
        _BATCH_SIZE,
        targets=args.targets,
        target_indices=args.target_indices)
    
    # set up model manager
    model_manager = setup_model_manager(args)

    # extract params
    model_manager.extract_model_variables(
        input_fn,
        args.out_dir,
        args.prefix,
        skip=args.skip)
    
    return
