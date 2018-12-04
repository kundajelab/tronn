"""Contains the run function for training a model
"""

import logging

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager
from tronn.util.formats import write_to_json
from tronn.util.utils import DataKeys


def run(args):
    """command to run training
    """
    # set up
    logger = logging.getLogger(__name__)
    logger.info("Training...")

    # define the logs
    train_dataset_log = "{}/dataset.train_split.json".format(args.out_dir)
    valid_dataset_log = "{}/dataset.validation_split.json".format(args.out_dir)
    test_dataset_log = "{}/dataset.test_split.json".format(args.out_dir)
    model_log = "{}/model.json".format(args.out_dir)
    train_log = "{}/train_summary.json".format(args.out_dir)
    
    # set up dataloader
    data_loader = setup_data_loader(args)

    # set up train/valid/test, default is generate new folds
    if args.full_train:
        logging.info("dataset: setting up full train")
        train_data_loader = data_loader.remove_genomewide_negatives()
        validation_data_loader = data_loader.remove_genomewide_negatives()
        test_data_loader = data_loader.remove_training_negatives()
    elif args.use_transfer_splits:
        logging.info("dataset: using train/valid/test splits from transfer model")
        train_data_loader = data_loader.filter_for_chromosomes(args.transfer_model["dataset"]["train"])
        validation_data_loader = data_loader.filter_for_chromosomes(args.transfer_model["dataset"]["valid"])
        test_data_loader = data_loader.filter_for_chromosomes(args.transfer_model["dataset"]["test"])
    else:
        logging.info("dataset: generating new folds for train/valid/test")
        split_data_loaders = data_loader.setup_cross_validation_dataloaders(
            kfolds=args.kfolds,
            valid_folds=args.valid_folds,
            test_folds=args.test_folds,
            regression=args.regression)
        train_data_loader = split_data_loaders[0]
        validation_data_loader = split_data_loaders[1]
        test_data_loader = split_data_loaders[2]

    # adjust for regression - just run on positives, adjust early stopping criterion
    if args.regression:
        train_data_loader = train_data_loader.setup_positives_only_dataloader()
        validation_data_loader = validation_data_loader.setup_positives_only_dataloader()
        test_data_loader = test_data_loader.setup_positives_only_dataloader()
        logging.info("regression - switching metric to MSE")
        args.early_stopping_metric = "mse"

    # save the chromosome splits into model info
    args.model["dataset"] = {
        "train": train_data_loader.get_chromosomes(),
        "validation": validation_data_loader.get_chromosomes(),
        "test": test_data_loader.get_chromosomes()}
    
    # save out dataset summaries (use the test json in eval)
    dataset_summary = {
        "targets": args.targets,
        "target_indices": args.target_indices,
        "filter_targets": args.filter_targets}
    write_to_json(
        dict(train_data_loader.describe(), **dataset_summary), train_dataset_log)
    write_to_json(
        dict(validation_data_loader.describe(), **dataset_summary), valid_dataset_log)
    write_to_json(
        dict(test_data_loader.describe(), **dataset_summary), test_dataset_log)
        
    # set up train input fn
    train_input_fn = train_data_loader.build_input_fn(
        args.batch_size,
        targets=args.targets,
        target_indices=args.target_indices,
        filter_targets=args.filter_targets,
        use_queues=True)
    
    # set up validation input fn
    validation_input_fn = validation_data_loader.build_input_fn(
        args.batch_size,
        targets=args.targets,
        target_indices=args.target_indices,
        filter_targets=args.filter_targets,
        use_queues=True)
    
    # if requested, get dataset metrics
    # TODO move this out
    if args.get_dataset_metrics:
        split_names = ["train", "validation", "test"]
        if not args.regression:
            for i in xrange(len(split_data_loaders)):
                split_data_loaders[i].get_classification_metrics(
                    "{}/dataset.{}".format(args.out_dir, split_names[i]),
                    targets=args.targets)
        else:
            for i in xrange(len(split_data_loaders)):
                split_data_loaders[i].get_regression_metrics(
                    "{}/dataset.{}".format(args.out_dir, split_names[i]),
                    targets=args.targets)

    # if model does not have num_tasks, infer from dataset
    if args.model["params"].get("num_tasks") is None:
        num_targets = train_data_loader.get_num_targets(
            targets=args.targets,
            target_indices=args.target_indices)
        args.model["params"]["num_tasks"] = num_targets

    # set up model and save summary
    model_manager = setup_model_manager(args)
    write_to_json(model_manager.describe(), model_log)
        
    # train and evaluate
    best_checkpoint = model_manager.train_and_evaluate(
        train_input_fn,
        validation_input_fn,
        args.out_dir,
        max_epochs=args.epochs,
        early_stopping_metric=args.early_stopping_metric,
        train_steps=None,
        eval_steps=int(1000. * 512 / args.batch_size),
        warm_start=args.transfer_checkpoint,
        warm_start_params=args.transfer_params,
        regression=args.regression,
        model_summary_file=model_log,
        train_summary_file=train_log,
        early_stopping=not args.full_train,
        multi_gpu=args.distributed,
        logit_indices=args.logit_indices)
            
    return None
