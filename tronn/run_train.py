"""Contains the run function for training a model
"""

import glob
import logging

import tensorflow as tf

from tronn.datalayer import get_total_num_examples
from tronn.datalayer import load_data_from_filename_list
from tronn.architectures import models
from tronn.graphs import TronnNeuralNetGraph

from tronn.learn.learning import train_and_evaluate
from tronn.learn.evaluation import get_global_avg_metrics


def run(args):
    """Runs training pipeline
    """
    logging.info("Running training...")

    # find data files
    data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))
    logging.info('Finding data: found {} chrom files'.format(len(data_files)))
    train_files = data_files[0:20]
    valid_files = data_files[20:22]
    # TODO(dk) set up test set of files too

    # Get number of train and validation steps
    args.num_train_examples = get_total_num_examples(train_files)
    args.train_steps = args.num_train_examples / args.batch_size - 100
    args.num_valid_examples = get_total_num_examples(valid_files)
    args.valid_steps = args.num_valid_examples / args.batch_size - 100
    
    logging.info("Num train examples: %d" % args.num_train_examples)
    logging.info("Num valid examples: %d" % args.num_valid_examples)
    logging.info("Train_steps/epoch: %d" % args.train_steps)


    # Set up neural net graph
    tronn_graph = TronnNeuralNetGraph(
        {"train": train_files, "valid": valid_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size,
        models[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        loss_fn=tf.losses.sigmoid_cross_entropy,
        optimizer_fn=tf.train.RMSPropOptimizer,
        optimizer_params={'learning_rate': 0.002, 'decay': 0.98, 'momentum': 0.0},
        metrics_fn=get_global_avg_metrics)
    
    # Train and evaluate for some number of epochs
    train_and_evaluate(
        tronn_graph,
        args.out_dir,
        args.train_steps,
        args.metric,
        args.patience,
        args.epochs,
        restore_model_dir=args.restore_model_dir,
        transfer_model_dir=args.transfer_model_dir)

    return None
