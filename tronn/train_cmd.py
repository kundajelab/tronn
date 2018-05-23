"""Contains the run function for training a model
"""

import glob
import logging

import tensorflow as tf

from tronn.datalayer import H5DataLoader
from tronn.nets.nets import net_fns
from tronn.graphs import ModelManager
from tronn.learn.cross_validation import setup_cv


def run(args):
    """Runs training pipeline
    """
    logging.info("Running training...")

    # find data files and set up folds
    # TODO make this more flexible for user
    data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))
    logging.info('Finding data: found {} chrom files'.format(len(data_files)))
    train_files, valid_files, test_files = setup_cv(data_files, cvfold=args.cvfold)

    # set up dataloader and buid the input functions needed to serve tensor batches
    train_dataloader = H5DataLoader(train_files)
    train_input_fn = train_dataloader.build_input_fn(args.batch_size)
    
    validation_dataloader = H5DataLoader(valid_files)
    validation_input_fn = validation_dataloader.build_input_fn(args.batch_size)
    
    # set up model
    model_manager = ModelManager(
        net_fns[args.model["name"]],
        args.model)
    
    # train and evaluate
    model_manager.train_and_evaluate_with_early_stopping(
        train_input_fn,
        validation_input_fn,
        args.out_dir,
        warm_start=args.transfer_model_checkpoint,
        warm_start_params={
            "skip":["logit"],
            "scope_change": ["", "basset/"]})

    return None
