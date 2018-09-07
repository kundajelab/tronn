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
        h5_files,
        args.kfolds,
        valid_folds=args.valid_folds,
        test_folds=args.test_folds,
        regression=args.regression) # TODO provide folds as param
    
    # set up dataloader and buid the input functions needed to serve tensor batches
    train_dataloader = H5DataLoader(train_files, fasta=args.fasta)
    train_input_fn = train_dataloader.build_input_fn(
        args.batch_size,
        label_keys=args.label_keys,
        filter_tasks=args.filter_keys)
    args.model["num_tasks"] = H5DataLoader.get_num_tasks(
        train_files, args.label_keys, args.label_key_dict)
    print args.model["num_tasks"]
    validation_dataloader = H5DataLoader(valid_files, fasta=args.fasta)
    validation_input_fn = validation_dataloader.build_input_fn(
        args.batch_size,
        label_keys=args.label_keys,
        filter_tasks=args.filter_keys)
    
    # set up model
    model_manager = ModelManager(
        net_fns[args.model["name"]],
        args.model)

    # save out training info into a json
    # need: model, model checkpoint (best), label sets.
    model_info = {
        "name": args.model["name"],
        "params": args.model,
        "checkpoint": None,
        "label_keys": args.label_keys,
        "filter_keys": args.filter_keys,
        "tasks": args.tasks,
        "train_files": train_files,
        "valid_files": valid_files,
        "test_files": test_files}
    
    # train and evaluate
    best_checkpoint = model_manager.train_and_evaluate_with_early_stopping(
        train_input_fn,
        validation_input_fn,
        args.out_dir,
        warm_start=args.transfer_model_checkpoint,
        warm_start_params={
            "skip":["logit"]},
            #"scope_change": ["", "basset/"]},
        regression=args.regression,
        model_info=model_info) # <- this is for larger model - adjust this
    
    # save out training info into a json
    # need: model, model checkpoint (best), label sets.
    model_info["checkpoint"] = best_checkpoint
    with open("{}/model_info.json".format(args.out_dir), "w") as fp:
        json.dump(model_info, fp, sort_keys=True, indent=4)
            
    return None
