"""Contains the run function for training a model
"""

import json
import logging

from tronn.datalayer import H5DataLoader
from tronn.models import ModelManager
from tronn.models import KerasModelManager
from tronn.nets.nets import net_fns
from tronn.learn.cross_validation import setup_train_valid_test


def run(args):
    """command to run training
    """
    logging.info("Training...")
    
    # set up dataset: lists of h5 files
    h5_files = H5DataLoader.setup_h5_files(args.data_dir)
    if args.full_train:
        # put all files into train, test, and valid
        train_files = []
        valid_files = []
        for chrom in sorted(h5_files.keys()):
            train_files += h5_files[chrom][0]
            train_files += h5_files[chrom][1]
            valid_files += h5_files[chrom][0]
            valid_files += h5_files[chrom][2]
        test_files = valid_files
    else:
        # if cross validation, split by chromosomes
        train_files, valid_files, test_files = setup_train_valid_test(
            h5_files,
            args.kfolds,
            valid_folds=args.valid_folds,
            test_folds=args.test_folds,
            regression=args.regression)

    # get num tasks from dataset
    args.model["num_tasks"] = H5DataLoader.get_num_tasks(
        train_files, args.label_keys, args.label_key_dict)
        
    # set up train dataloader and input fn
    train_dataloader = H5DataLoader(train_files, fasta=args.fasta)
    train_input_fn = train_dataloader.build_input_fn(
        args.batch_size,
        label_keys=args.label_keys,
        filter_tasks=args.filter_keys)
    
    # set up valid dataloader and input fn
    validation_dataloader = H5DataLoader(valid_files, fasta=args.fasta)
    validation_input_fn = validation_dataloader.build_input_fn(
        args.batch_size,
        label_keys=args.label_keys,
        filter_tasks=args.filter_keys)
    
    # if requested, get dataset metrics
    if args.get_dataset_metrics:
        if not args.regression:
            train_dataloader.get_classification_metrics(
                "{}/dataset.train".format(args.out_dir),
                label_keys=args.label_keys)
            validation_dataloader.get_classification_metrics(
                "{}/dataset.validation".format(args.out_dir),
                label_keys=args.label_keys)
            test_dataloader = H5DataLoader(
                test_files, fasta=args.fasta)
            test_dataloader.get_classification_metrics(
                "{}/dataset.test".format(args.out_dir),
                label_keys=args.label_keys)
        else:
            train_dataloader.get_regression_metrics(
                "{}/dataset.train".format(args.out_dir),
                label_keys=args.label_keys)
            validation_dataloader.get_regression_metrics(
                "{}/dataset.validation".format(args.out_dir),
                label_keys=args.label_keys)
            test_dataloader = H5DataLoader(
                test_files, fasta=args.fasta)
            test_dataloader.get_regression_metrics(
                "{}/dataset.test".format(args.out_dir),
                label_keys=args.label_keys)
        
    # set up model (either a keras transfer or tensorflow model)
    if args.transfer_keras is not None:
        args.model["name"] = "keras_transfer"
        with open(args.transfer_keras) as fp:
            args.model_info = json.load(fp)
        model_manager = KerasModelManager(
            keras_model_path=args.model_info["checkpoint"],
            model_params=args.model_info.get("params", {}),
            model_dir=args.out_dir)
        args.transfer_model_checkpoint = model_manager.model_checkpoint
        warm_start_params = {}
    else:
        model_manager = ModelManager(
            net_fns[args.model["name"]],
            args.model,
            name=args.model["name"])
        warm_start_params = {"skip": ["logit"]}

    # create a model info json and save out initial state
    # TODO - remove other logs and condense to this log
    model_info = {
        "name": model_manager.name,
        "params": model_manager.model_params,
        "checkpoint": model_manager.model_checkpoint,
        "label_keys": args.label_keys,
        "filter_keys": args.filter_keys,
        "tasks": args.tasks,
        "train_files": train_files,
        "valid_files": valid_files,
        "test_files": test_files}
    with open("{}/model_info.json".format(args.out_dir), "w") as fp:
        json.dump(model_info, fp, sort_keys=True, indent=4)
    
    # train and evaluate
    best_checkpoint = model_manager.train_and_evaluate_with_early_stopping(
        train_input_fn,
        validation_input_fn,
        args.out_dir,
        max_epochs=args.epochs,
        early_stopping_metric=args.metric,
        eval_steps=int(1000. * 512 / args.batch_size),
        warm_start=args.transfer_model_checkpoint,
        warm_start_params=warm_start_params,
        regression=args.regression,
        model_info=model_info,
        early_stopping=not args.full_train,
        multi_gpu=args.distributed)

    # do a final save (in case no epochs were run)
    model_info["checkpoint"] = best_checkpoint
    with open("{}/model_info.json".format(args.out_dir), "w") as fp:
        json.dump(model_info, fp, sort_keys=True, indent=4)
            
    return None
