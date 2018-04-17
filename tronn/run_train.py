"""Contains the run function for training a model
"""

import glob
import logging

import tensorflow as tf

#from tronn.datalayer import get_total_num_examples
#from tronn.datalayer import load_data_from_filename_list
from tronn.datalayer import H5DataLoader

from tronn.nets.nets import net_fns
#from tronn.graphs import TronnNeuralNetGraph
from tronn.graphs import TronnGraphV2

from tronn.learn.cross_validation import setup_cv
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

        # change loss to weighted loss
        #tronn_graph.loss_fn = tf.nn.weighted_cross_entropy_with_logits
        #tronn_graph.class_weighted_loss = True
        
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
    train_files, valid_files, test_files = setup_cv(data_files, cvfold=args.cvfold)

    # set up dataloader
    dataloader = H5DataLoader(
        {"train": train_files, "valid": valid_files},
        tasks=args.tasks)
    
    # Get number of train and validation steps
    # TODO - this can be moved to metrics now
    args.num_train_examples = dataloader.get_num_total_examples("train")
    args.train_steps = args.num_train_examples / args.batch_size - 100
    args.num_valid_examples = dataloader.get_num_total_examples("valid")
    args.valid_steps = args.num_valid_examples / args.batch_size - 100
    
    logging.info("Num train examples: %d" % args.num_train_examples)
    logging.info("Num valid examples: %d" % args.num_valid_examples)
    logging.info("Train_steps/epoch: %d" % args.train_steps)
    
    # extract checkpoint paths
    restore_model_checkpoint = None
    checkpoints = []
    if args.restore_model_dir is not None:
        restore_model_checkpoint = tf.train.latest_checkpoint(args.restore_model_dir)
        checkpoints.append(restore_model_checkpoint)
    if args.restore_model_checkpoint is not None:
        restore_model_checkpoint = args.restore_model_checkpoint
        checkpoints.append(restore_model_checkpoint)
        
    transfer_model_checkpoint = None
    if args.transfer_model_dir is not None:
        transfer_model_checkpoint = tf.train.latest_checkpoint(args.transfer_model_dir)
        checkpoints.append(transfer_model_checkpoint)
    if args.transfer_model_checkpoint is not None:
        transfer_model_checkpoint = args.transfer_model_checkpoint
        checkpoints.append(transfer_model_checkpoint)
        
    assert not ((restore_model_checkpoint is not None)
                and (transfer_model_checkpoint is not None))

    if False:
        tronn_graph = TronnNeuralNetGraph(
            {"train": train_files, "valid": valid_files}, # TODO move from graph
            args.tasks,
            dataloader, # TODO move from graph
            #load_data_from_filename_list,
            args.batch_size,
            net_fns[args.model['name']], # model
            args.model, # model params
            tf.nn.sigmoid,
            loss_fn=tf.losses.sigmoid_cross_entropy,
            #loss_fn=tf.losses.hinge_loss,
            #loss_fn=tf.nn.weighted_cross_entropy_with_logits,
            #positives_focused_loss=True,
            #class_weighted_loss=True,
            optimizer_fn=tf.train.RMSPropOptimizer,
            optimizer_params={'learning_rate': 0.002, 'decay': 0.98, 'momentum': 0.0},
            metrics_fn=get_global_avg_metrics,
            checkpoints=checkpoints)

    
    # Set up neural net graph
    tronn_graph = TronnGraphV2(
        dataloader,
        net_fns[args.model['name']], # model
        args.model, # model params
        args.batch_size,
        final_activation_fn=tf.nn.sigmoid,
        loss_fn=tf.losses.sigmoid_cross_entropy,
        optimizer_fn=tf.train.RMSPropOptimizer,
        optimizer_params={'learning_rate': 0.002, 'decay': 0.98, 'momentum': 0.0},
        metrics_fn=get_global_avg_metrics,
        checkpoints=checkpoints)

    
    # make finetune training and training mutually exclusive for now
    if len(args.finetune_tasks) == 0:
        # Train and evaluate for some number of epochs
        trained_model_dir = train_and_evaluate(
            tronn_graph,
            args.out_dir,
            args.train_steps, # TODO factor this out
            args.metric,
            args.patience,
            epoch_limit=args.epochs,
            restore_model_checkpoint=restore_model_checkpoint, # TODO factor this out
            transfer_model_checkpoint=transfer_model_checkpoint) # TODO factor this out
    else:
        # add in fine-tuning option on tasks
        finetune_tasks(args, tronn_graph, args.restore_model_dir)

    return None
