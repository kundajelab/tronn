"""Contains the run function for training a model
"""

import json
import logging

from tronn.datalayer import setup_h5_files
from tronn.datalayer import H5DataLoader

from tronn.graphs import ModelManager

from tronn.nets.nets import net_fns

from tronn.learn.cross_validation import setup_train_valid_test


def run(args):
    """command to run training
    """
    logging.info("Training...")

    # set up dataset
    h5_files = setup_h5_files(args.data_dir)
    train_files, valid_files, test_files = setup_train_valid_test(
        h5_files, 10) # TODO provide folds as param
    
    # set up dataloader and buid the input functions needed to serve tensor batches
    train_dataloader = H5DataLoader(train_files)
    train_input_fn = train_dataloader.build_input_fn(
        args.batch_size, label_keys=args.label_keys)
    validation_dataloader = H5DataLoader(valid_files)
    validation_input_fn = validation_dataloader.build_input_fn(
        args.batch_size, label_keys=args.label_keys)
    
    # set up model
    model_manager = ModelManager(
        net_fns[args.model["name"]],
        args.model)
    
    # train and evaluate
    best_checkpoint = model_manager.train_and_evaluate_with_early_stopping(
        train_input_fn,
        validation_input_fn,
        args.out_dir,
        warm_start=args.transfer_model_checkpoint,
        warm_start_params={
            "skip":["logit"],
            "scope_change": ["", "basset/"]}) # <- this is for larger model - adjust this
    
    # save out training info into a json
    # need: model, model checkpoint (best), label sets.
    model_info = {
        "name": args.model["name"],
        "params": args.model,
        "checkpoint": best_checkpoint,
        "label_keys": args.label_keys,
        "tasks": args.tasks,
        "num_tasks": train_dataloader.get_num_tasks(train_files),
        "train_files": train_files,
        "valid_files": valid_files,
        "test_files": test_files}
    with open("{}/model_info.json".format(args.out_dir), "w") as fp:
        json.dump(model_info, fp, sort_keys=True, indent=4)
    
    return None
