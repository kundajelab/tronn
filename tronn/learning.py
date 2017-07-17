""" Contains light wrappers for learning and evaluation

The wrappers follow the tf-slim structure for setting up and running a model

"""

import os
import tf_utils

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.datalayer import get_total_num_examples
from tronn.datalayer import load_data_from_filename_list
from tronn.evaluation import get_global_avg_metrics
from tronn.models import models


def train(
        data_loader,
        model_builder,
        final_activation_fn,
        loss_fn,
        optimizer_fn,
        optimizer_params,
        metrics_fn,
        restore,
        stopping_criterion,
        args,
        data_file_list,
        OUT_DIR,
        target_global_step,
        transfer=False,
        transfer_dir='./',
        weighted_cross_entropy=False):
    """ Wraps the routines needed for tf-slim training
    """

    with tf.Graph().as_default():

        # data loader
        features, labels, metadata = data_loader(data_file_list, args.batch_size, args.tasks)

        # model
        logits = model_builder(features, labels, args.model, is_training=True)
        probabilities = final_activation_fn(logits)

        if not weighted_cross_entropy:
            loss = loss_fn(labels, logits)
        else:
            print "NOTE: using weighted loss!"
            pos_weights = nn_utils.get_positive_weights_per_task(data_file_list)
            task_losses = []
            for task_num in range(labels.get_shape()[1]):
                # somehow need to calculate task imbalance...
                task_losses.append(loss_fn(predictions[:,task_num], labels[:,task_num], pos_weights[task_num], loss_collection=None))
            task_loss_tensor = tf.stack(task_losses, axis=1)
            loss = tf.reduce_sum(task_loss_tensor)
            # to later get total_loss
            tf.add_to_collection(ops.GraphKeys.LOSSES, loss)
        total_loss = tf.losses.get_total_loss()

        # metrics
        metric_value, updates = metrics_fn(labels, probabilities, args.tasks)
        for update in updates: tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)
        mean_loss, _ = tf.metrics.mean(loss, updates_collections=tf.GraphKeys.UPDATE_OPS)
        metric_value.update({
            'loss': loss,
            'total_loss': total_loss,
            'mean_loss': mean_loss
            })
        
        # train op
        optimizer = optimizer_fn(**optimizer_params)
        train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

        # summaries
        tf_utils.add_summaries(metric_value)
        for var in tf.model_variables(): tf_utils.add_var_summaries(var)
        values_for_printing = [tf.train.get_global_step(),
                               metric_value['mean_loss'],
                               metric_value['mean_accuracy'],
                               metric_value['mean_auprc'],
                               metric_value['mean_auroc'],
                               loss,
                               total_loss]
        summary_op = tf.Print(tf.summary.merge_all(), values_for_printing)
        
        # print parameter count
        if not restore:
            print 'Created new model:'
            model_params = sum(v.get_shape().num_elements() for v in tf.model_variables())
            trainable_params = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
            total_params = sum(v.get_shape().num_elements() for v in tf.global_variables())
            for var in sorted(tf.trainable_variables(), key=lambda var: (var.name, var.get_shape().num_elements())):
                num_elems = var.get_shape().num_elements()
                if num_elems>500:
                    print var.name, var.get_shape().as_list(), num_elems
            print 'Num params (model/trainable/global): %d/%d/%d' % (model_params, trainable_params, total_params)

        # TODO consider changing the logic here
        if restore:
            checkpoint_path = tf.train.latest_checkpoint(OUT_DIR)
            variables_to_restore = slim.get_model_variables()
            variables_to_restore.append(slim.get_global_step())
            
            # TODO if pretrained on different dataset, remove final layer variables
            if args.transfer_dir:
                print "transferring model"
                checkpoint_path = tf.train.latest_checkpoint(args.transfer_dir)
                variables_to_restore = slim.get_model_variables()
                variables_to_restore.append(slim.get_global_step())
                
                #variables_to_restore_tmp = [ var for var in variables_to_restore if ('out' not in var.name) ]
                variables_to_restore_tmp = [ var for var in variables_to_restore if (('logit' not in var.name) and ('out' not in var.name)) ]
                variables_to_restore = variables_to_restore_tmp
                checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.transfer_dir))
                
            print variables_to_restore
            print checkpoint_path
            
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                checkpoint_path,
                variables_to_restore)
            
            # create init assignment function
            def restoreFn(sess):
                sess.run(init_assign_op, init_feed_dict)

        slim.learning.train(train_op,
                            OUT_DIR,
                            init_fn=restoreFn if restore else None,
                            number_of_steps=target_global_step,
                            summary_op=summary_op,
                            save_summaries_secs=60,
                            save_interval_secs=3600,)

def evaluate(
        data_loader,
        model_builder,
        final_activation_fn,
        loss_fn,
        metrics_fn,
        checkpoint_path,
        args,
        data_file_list,
        out_dir,
        num_evals,
        weighted_cross_entropy=False):
    """
    Wrapper function for doing evaluation (ie getting metrics on a model)
    Note that if you want to reload a model, you must load the same model
    and data loader
    """
    print 'evaluating %s...'%checkpoint_path
    with tf.Graph().as_default():

        # data loader
        features, labels, metadata = data_loader(data_file_list, args.batch_size*4, args.tasks)

        # model - training=False
        logits = model_builder(features, labels, args.model, is_training=False)
        probabilities = final_activation_fn(logits)
        if not weighted_cross_entropy:
            loss = loss_fn(labels, logits)
        else:
            print "NOTE: using weighted loss!"
            pos_weights = nn_utils.get_positive_weights_per_task(data_file_list)
            task_losses = []
            for task_num in range(labels.get_shape()[1]):
                # somehow need to calculate task imbalance...
                task_losses.append(loss_fn(predictions[:,task_num], labels[:,task_num], pos_weights[task_num], loss_collection=None))
            task_loss_tensor = tf.stack(task_losses, axis=1)
            loss = tf.reduce_sum(task_loss_tensor)
            # to later get total_loss
            tf.add_to_collection(ops.GraphKeys.LOSSES, loss)

        total_loss = tf.losses.get_total_loss()

        # metrics
        metric_value, updates = metrics_fn(labels, probabilities, args.tasks)
        for update in updates: tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)

        metric_value['loss'] = loss
        metric_value['total_loss'] = total_loss
        tf_utils.add_summaries(metric_value)

        print "set up done"

        # Evaluate the checkpoint
        metrics_dict = slim.evaluation.evaluate_once(
            None,
            checkpoint_path,
            out_dir,
            num_evals=num_evals,
            summary_op=tf.summary.merge_all(),
            eval_op=updates,
            final_op=metric_value,)
        
        print 'Validation metrics:\n%s'%metrics_dict
    
    return metrics_dict


def train_and_evaluate_once(
        args,
        data_loader, 
        model, 
        final_activation_fn,
        loss_fn,
        optimizer,
        optimizer_params,
        metrics_fn,
        restore,
        train_files,
        validation_files,
        stop_step,
        valid_steps=10000): # 10k
    """Run training for the given number of steps and then evaluate
    """

    # Run training
    train(data_loader, 
          model,
          final_activation_fn,
          loss_fn,
          optimizer, 
          optimizer_params,
          metrics_fn,
          restore,
          'Not yet implemented',
          args,
          train_files,
          '{}/train'.format(args.out_dir),
          stop_step)

    # Get last checkpoint
    checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.out_dir)) 

    # Evaluate after training
    eval_metrics = evaluate(
        data_loader,
        model,
        final_activation_fn,
        loss_fn,
        metrics_fn,
        checkpoint_path,
        args,
        validation_files,
        '{}/valid'.format(args.out_dir),
        num_evals=valid_steps)
    

    return eval_metrics


def train_and_evaluate(
        args,
        data_loader,
        model,
        final_activation_fn,
        loss_fn,
        optimizer,
        optimizer_params,
        metrics_fn,
        restore,
        train_files,
        validation_files,
        epoch_limit):
    """Runs training and evaluation for X epochs"""

    # track metric and bad epochs
    metric_best = None
    consecutive_bad_epochs = 0

    for epoch in xrange(epoch_limit):
        print "CURRENT EPOCH:", str(epoch)

        print restore
        #restore = args.restore is not None

        if epoch > 0:
            restore = True

        if args.transfer_dir:
            checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.transfer_dir))
            curr_step = int(checkpoint_path.split('-')[-1].split('.')[0])
            print curr_step
            target_step = curr_step + args.train_steps
            print target_step
        elif restore:
            checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.out_dir))
            curr_step = int(checkpoint_path.split('-')[-1].split('.')[0])
            target_step = curr_step + args.train_steps
        else:
            target_step = args.train_steps

        eval_metrics = train_and_evaluate_once(args,
                                               data_loader,
                                               model,
                                               final_activation_fn,
                                               loss_fn,
                                               optimizer, optimizer_params,
                                               metrics_fn,
                                               restore,
                                               train_files,
                                               validation_files,
                                               target_step)

        # Early stopping and saving best model
        if metric_best is None or ('loss' in args.metric) != (eval_metrics[args.metric]>metric_best):
            consecutive_bad_epochs = 0
            metric_best = eval_metrics[args.metric]
            with open(os.path.join(args.out_dir, 'best.txt'), 'w') as f:
                f.write('epoch %d\n'%epoch)
                f.write(str(eval_metrics))
        else:
            consecutive_bad_epochs += 1
            if consecutive_bad_epochs>args.patience:
                print 'early stopping triggered'
                break
    
    
    return None


def run(args):
    """Main training and evaluation function
    """

    # find data files
    data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))
    print 'Found {} chrom files'.format(len(data_files))
    train_files = data_files[0:20]
    valid_files = data_files[20:22]

    # Get number of train and validation steps
    args.num_train_examples = get_total_num_examples(train_files)
    args.train_steps = args.num_train_examples / args.batch_size - 100
    args.num_valid_examples = get_total_num_examples(valid_files)
    args.valid_steps = args.num_valid_examples / args.batch_size - 100
    
    print 'Num train examples: %d' % args.num_train_examples
    print 'Num valid examples: %d' % args.num_valid_examples
    print 'train_steps/epoch: %d' % args.train_steps

    # TODO fix transfer args
    train_and_evaluate(args,
                       load_data_from_filename_list,
                       models[args.model['name']],
                       tf.nn.sigmoid,
                       tf.losses.sigmoid_cross_entropy,
                       tf.train.RMSPropOptimizer, {'learning_rate': 0.002, 'decay': 0.98, 'momentum': 0.0},
                       get_global_avg_metrics,
                       args.restore,
                       train_files,
                       valid_files,
                       args.epochs)
    
    return
