""" Contains light wrappers for learning and evaluation

The wrappers follow the tf-slim structure for setting up and running a model

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

import nn_utils


def train(data_loader,
          model_builder,
          loss_fn,
          optimizer_fn,
          optimizer_params,
          metrics_fn,
          restore,
          stopping_criterion,
          args,
          data_file_list,
          OUT_DIR,
          global_step_val,
          transfer=False,
          transfer_dir='./',
          weighted_cross_entropy=False,
          model_has_config=False,
          model_config=None):
    '''
    Wraps the routines needed for tf-slim
    '''

    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader(data_file_list,
                                                 args.batch_size)

        num_tasks = labels.get_shape()
        print num_tasks[1]

        
        # model
        if model_has_config:
            predictions = model_builder(features, int(num_tasks[1]), args.model, is_training=True)
        else:
            predictions = model_builder(features, labels, is_training=True)

        # loss
        # TODO adjust the loss for class imbalance scenario
        if not weighted_cross_entropy:
            total_loss = loss_fn(predictions, labels)
        else:
            print "NOTE: using weighted loss!"
            pos_weights = nn_utils.get_positive_weights_per_task(data_file_list)
            task_losses = []
            for task_num in range(labels.get_shape()[1]):
                # somehow need to calculate task imbalance...
                task_losses.append(loss_fn(predictions[:,task_num], labels[:,task_num], pos_weights[task_num]))
            task_loss_tensor = tf.stack(task_losses, axis=1)
            # stuff for tf-slim
            total_loss = tf.reduce_sum(task_loss_tensor)
            tf.add_to_collection(ops.GraphKeys.LOSSES, total_loss)
            

        # optimizer
        optimizer = optimizer_fn(**optimizer_params)

        # train op
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # build metrics
        summary_op = metrics_fn(total_loss, predictions, labels)

        if restore:
            checkpoint_path = tf.train.latest_checkpoint(OUT_DIR)
            variables_to_restore = slim.get_model_variables()
            variables_to_restore.append(slim.get_global_step())
            
            # TODO if pretrained on different dataset, remove final layer variables
            if transfer:
                #variables_to_restore_tmp = [ var for var in variables_to_restore if ('out' not in var.name) ]
                variables_to_restore_tmp = [ var for var in variables_to_restore if (('logit' not in var.name) and ('out' not in var.name)) ]
                variables_to_restore = variables_to_restore_tmp
                checkpoint_path = tf.train.latest_checkpoint(transfer_dir)
            
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                checkpoint_path,
                variables_to_restore)
            
            # create init assignment function
            def InitAssignFn(sess):
                sess.run(init_assign_op, init_feed_dict)
                
            slim.learning.train(train_op,
                                OUT_DIR,
                                init_fn=InitAssignFn,
                                number_of_steps=global_step_val,
                                summary_op=summary_op,
                                save_summaries_secs=20,
                                saver=tf.train.Saver(max_to_keep=None),
                                save_interval_secs=3600)

        else:
            slim.learning.train(train_op,
                                OUT_DIR,
                                number_of_steps=global_step_val,
                                summary_op=summary_op,
                                save_summaries_secs=20,
                                saver=tf.train.Saver(max_to_keep=None),
                                save_interval_secs=3600)

    return None


def evaluate(data_loader,
             model_builder,
             final_activation_fn,
             metrics_fn,
             checkpoint_path,
             args,
             data_file_list,
             out_dir,
             num_evals=10000,
             model_has_config=False):
    '''
    Wrapper function for doing evaluation (ie getting metrics on a model)
    Note that if you want to reload a model, you must load the same model
    and data loader
    '''

    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader(data_file_list,
                                                 args.batch_size)

        num_tasks = labels.get_shape()
        print num_tasks[1]
        
        # model - training=False
        if model_has_config:
            predictions_prob = final_activation_fn(
                model_builder(features, int(num_tasks[1]), args.model, is_training=False))
        else:
            predictions_prob = final_activation_fn(
                model_builder(features, labels, is_training=False))

        # boolean classification predictions and labels
        labels_bool = tf.cast(labels, tf.bool)
        predictions_bool = tf.greater(predictions_prob, 0.5)
        
        # Choose the metrics to compute
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            "accuracy": slim.metrics.streaming_accuracy(
                predictions_bool, labels_bool),
            "auROC": slim.metrics.streaming_auc(
                predictions_prob, labels, curve="ROC"),
            "auPRC": slim.metrics.streaming_auc(
                predictions_prob, labels, curve="PR"),
            })

        # Define the scalar summaries to write
        for metric_name, metric_value in names_to_values.iteritems():
            tf.summary.scalar(metric_name, metric_value)

        # Evaluate the checkpoint
        metrics_dict = slim.evaluation.evaluate_once(
            None,
            checkpoint_path,
            out_dir,
            num_evals=num_evals,
            summary_op=tf.summary.merge_all(),
            eval_op= names_to_updates.values(),
            final_op=names_to_values)
        
        print metrics_dict
    
    return None
