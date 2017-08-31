"""Contains the run function for training a model
"""

import glob
import logging

import tensorflow as tf

from tronn.datalayer import get_total_num_examples
from tronn.datalayer import load_data_from_filename_list
from tronn.nets.nets import model_fns
from tronn.graphs import TronnNeuralNetGraph

from tronn.learn.learning import train_and_evaluate
from tronn.learn.evaluation import get_global_avg_metrics

def finetune_tasks(args, tronn_graph, trained_model_dir):
    """Allows fine tuning of individual tasks (just final layer)
    """
    finetune_dir = "{}/finetune".format(args.out_dir)
    for i in range(len(args.finetune_tasks)):
        task = int(args.finetune_tasks[i])
        print "finetuning", task
        if i == 0:
            restore_dir = args.restore_model_dir
        else:
            restore_dir = finetune_dir

        # adjust graph to finetune
        tronn_graph.finetune = True
        tronn_graph.finetune_tasks = [task]

        train_and_evaluate(
            tronn_graph,
            finetune_dir,
            args.train_steps,
            args.metric,
            args.patience,
            epoch_limit=5,
            restore_model_dir=trained_model_dir)

    return



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
        model_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        loss_fn=tf.losses.sigmoid_cross_entropy,
        #loss_fn=tf.nn.weighted_cross_entropy_with_logits,
        #positives_focused_loss=True,
        #class_weighted_loss=True,
        optimizer_fn=tf.train.RMSPropOptimizer,
        optimizer_params={'learning_rate': 0.002, 'decay': 0.98, 'momentum': 0.0},
        metrics_fn=get_global_avg_metrics)

    # make finetune training and training mutually exclusive for now
    if len(args.finetune_tasks) == 0:
        # Train and evaluate for some number of epochs
        trained_model_dir = train_and_evaluate(
            tronn_graph,
            args.out_dir,
            args.train_steps,
            args.metric,
            args.patience,
            epoch_limit=args.epochs,
            restore_model_dir=args.restore_model_dir,
            transfer_model_dir=args.transfer_model_dir)
    else:
        # add in fine-tuning option on tasks
        finetune_tasks(args, tronn_graph, args.restore_model_dir)

    return None
