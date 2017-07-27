""" Contains light wrappers for learning and evaluation

The wrappers follow the tf-slim structure for setting up and running a model

"""

import os
import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_ops import restore_variables_op
from tronn.util.tf_ops import task_weighted_loss_fn
from tronn.util.tf_utils import get_stop_step
from tronn.util.tf_utils import add_summaries
from tronn.util.tf_utils import add_var_summaries


def train(
        data_files,
        tasks,
        data_loader,
        model_fn,
        model_params, # model config
        final_activation_fn,
        loss_fn,
        metrics_fn,
        optimizer_fn,
        optimizer_params,
        stop_step,
        out_dir,
        batch_size=128,
        restore_model_dir=None,
        transfer_model_dir=None,
        weighted_cross_entropy=False):
    """Training routine utilizing tf-slim

    Args:
      data_files: list of data files for data loader
      tasks: list of task indices if using a subset of tasks
      data_loader: datalayer interface to queue and load data
      model_fn: defined architecture to build
      model_params: extra configuration params
      final_activation_fn: final activation function (normally sigmoid)
      loss_fn: loss
      metrics_fn: sets up metrics to calculate
      optimizer_fn: optimizer
      optimizer_params: extra optimizer params
      stop_step: global stepping stop point
      out_dir: where to save the model and events
      batch_size: batch size
      restore_model_dir: restore EXACT same model
      transfer_model_dir: transfer EXACT model minus out nodes
      weighted_cross_entropy: change to a weighted loss

    Returns:
      None
    """
    assert not ((restore_model_dir is not None) and (transfer_model_dir is not None))
    
    with tf.Graph().as_default():

        # data loader
        features, labels, metadata = data_loader(data_files, batch_size, tasks)

        # model and final activation
        logits = model_fn(features, labels, model_params, is_training=True)
        probabilities = final_activation_fn(logits)

        # loss
        if not weighted_cross_entropy:
            loss = loss_fn(labels, logits)
        else:
            loss = task_weighted_loss_fn(data_files, loss_fn, labels, logits)
        total_loss = tf.losses.get_total_loss()

        # metrics
        metric_value, updates = metrics_fn(labels, probabilities, tasks)
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
        add_summaries(metric_value)
        for var in tf.model_variables(): add_var_summaries(var)
        values_for_printing = [tf.train.get_global_step(),
                               metric_value['mean_loss'],
                               metric_value['mean_accuracy'],
                               metric_value['mean_auprc'],
                               metric_value['mean_auroc'],
                               loss,
                               total_loss]
        summary_op = tf.Print(tf.summary.merge_all(), values_for_printing)
        
        # print parameter count
        if (restore_model_dir is None) and (transfer_model_dir is None):
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
        if restore_model_dir is not None:
            init_assign_op, init_feed_dict = restore_variables_op(restore_model_dir)
            # create init assignment function
            def restoreFn(sess):
                sess.run(init_assign_op, init_feed_dict)
        elif transfer_model_dir is not None:
            init_assign_op, init_feed_dict = restore_variables_op(restore_transfer_dir,
                                                                  skip=['logit','out'])
            # create init assignment function
            def restoreFn(sess):
                sess.run(init_assign_op, init_feed_dict)
        else:
            restoreFn = None
        
        slim.learning.train(train_op,
                            out_dir,
                            init_fn=restoreFn,
                            number_of_steps=stop_step,
                            summary_op=summary_op,
                            save_summaries_secs=60,
                            save_interval_secs=3600,)

    return


def evaluate(
        data_files,
        tasks,
        data_loader,
        model_fn,
        model_params,
        final_activation_fn,
        loss_fn,
        metrics_fn,
        out_dir,
        model_dir,
        stop_step,
        batch_size=128,
        weighted_cross_entropy=False):
    """Evaluation routine using tf-slim

    Args:
      data_files: list of data files to evaluate
      tasks: list of task indices used in evaluation
      data_loader: datalayer to stream data to graph
      model_fn: model building function
      model_params: extra params for model
      final_activation_fn: final activation
      loss_fn: loss
      metrics_fn: metrics to calculate
      out_dir: output directory
      model_dir: directory with trained model
      stop_step: how many evals to run
      batch_size: batch size
      weighted_cross_entropy: task weighted

    Returns:
      metrics_dict: dictionary of metrics from metrics fn
    """
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    logging.info('evaluating %s...'%checkpoint_path)
    
    with tf.Graph().as_default():

        # data loader
        features, labels, metadata = data_loader(data_files, batch_size*4, tasks)

        # model - training=False
        logits = model_fn(features, labels, model_params, is_training=False)
        probabilities = final_activation_fn(logits)
        if not weighted_cross_entropy:
            loss = loss_fn(labels, logits)
        else:
            loss = task_weighted_loss_fn(data_files, loss_fn, labels, logits)
        total_loss = tf.losses.get_total_loss()

        # metrics
        metric_value, updates = metrics_fn(labels, probabilities, tasks)
        for update in updates: tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)

        metric_value['loss'] = loss
        metric_value['total_loss'] = total_loss
        add_summaries(metric_value)

        print "set up done"

        # Evaluate the checkpoint
        metrics_dict = slim.evaluation.evaluate_once(
            None,
            checkpoint_path,
            out_dir,
            num_evals=stop_step,
            summary_op=tf.summary.merge_all(),
            eval_op=updates,
            final_op=metric_value,)
        
        print 'Validation metrics:\n%s'%metrics_dict
    
    return metrics_dict


def train_and_evaluate_once(
        train_files,
        validation_files,
        tasks,
        data_loader,
        model_fn,
        model_params,
        final_activation_fn,
        loss_fn,
        optimizer_fn,
        optimizer_params,
        metrics_fn,
        train_stop_step,
        train_dir,
        valid_dir,
        batch_size=128,
        restore_model_dir=None,
        transfer_model_dir=None,
        valid_stop_step=10000): # 10k
    """Routine to train and evaluate for some number of steps
    
    Args:
      train_files: list of data files for training
      validation_files: list of data files for validation
      tasks: tasks to train/evaluate on
      data_loader: datalayer to stream data to graph
      model_fn: model builder
      model_params: extra params for model
      final_activation_fn: final activation
      loss_fn: loss
      optimizer_fn: optimizer
      optimizer_params: extra params for optimizer
      metrics_fn: metrics to calculate
      train_stop_step: number of train steps to run
      out_dir: output directory (fn makes train/valid dirs)
      batch_size: batch size
      restore_model_dir: location of a checkpoint that is
        EXACTLY the same as the train model
      transfer_model_dir: location of a checkpoint that is
        exactly the same as the train model EXCEPT for the
        last layer (logits)
      valid_stop_step: number of valid steps to run

    Returns:
      eval_metrics: metric dictionary with stop metric
    """

    # Run training
    train(
        train_files,
        tasks,
        data_loader,
        model_fn,
        model_params,
        final_activation_fn,
        loss_fn,
        metrics_fn,
        optimizer_fn, 
        optimizer_params,
        train_stop_step,
        train_dir,
        batch_size=batch_size,
        restore_model_dir=restore_model_dir,
        transfer_model_dir=transfer_model_dir)

    # Evaluate after training (use for stopping criteria)
    eval_metrics = evaluate(
        validation_files,
        tasks,
        data_loader,
        model_fn,
        model_params,
        final_activation_fn,
        loss_fn,
        metrics_fn,
        valid_dir,
        train_dir,
        valid_stop_step,
        batch_size=batch_size)

    return eval_metrics


def train_and_evaluate(
        train_files,
        validation_files,
        tasks,
        data_loader,
        model_fn,
        model_params,
        final_activation_fn,
        loss_fn,
        optimizer_fn,
        optimizer_params,
        metrics_fn,
        out_dir,
        train_steps,
        stop_metric,
        patience,
        epoch_limit,
        batch_size=128,
        restore_model_dir=None,
        transfer_model_dir=None):
    """Runs training and evaluation for {epoch_limit} epochs

    Args:
      train_files: list of data files for training
      validation_files: list of data files for validation
      tasks: list of tasks to train/evaluate on
      data_loader: datalayer for streaming data to graph
      model_fn: model builder
      model_params: extra params for model
      final_activation_fn: final activation
      loss_fn: loss
      optimizer_fn: optimizer
      optimizer_params: extra params for optimizer
      metrics_fn: metrics to evaluate
      out_dir: where to save outputs
      train_steps: number of train steps to run
      stop_metric: metric used for early stopping
      patience: number of epochs to wait for improvement
      epoch_limit: number of max epochs
      batch_size: batch size
      restore_model_dir: location of a checkpoint that is
        EXACTLY the same as the train model
      transfer_model_dir: location of a checkpoint that is
        exactly the same as the train model EXCEPT for the
        last layer (logits)

    Returns:
      None
    """
    assert not ((restore_model_dir is not None) and (transfer_model_dir is not None))
    
    # track metric and bad epochs
    metric_best = None
    consecutive_bad_epochs = 0

    for epoch in xrange(epoch_limit):
        logging.info("CURRENT EPOCH:", str(epoch))
        
        if epoch > 0:
            # make sure that transfer_model_dir is None and restore_model_dir is set correctly
            transfer_model_dir = None
            restore_model_dir = "{}/train".format(out_dir)

        # determine the global target step if restoring/transferring
        if restore_model_dir is not None:
            stop_step = get_stop_step(restore_model_dir, train_steps)
        elif transfer_model_dir is not None:
            stop_step = get_stop_step(transfer_model_dir, train_steps)
        else:
            stop_step = train_steps

        # train and evaluate one epoch
        eval_metrics = train_and_evaluate_once(
            train_files,
            validation_files,
            tasks,
            data_loader,
            model_fn,
            model_params,
            final_activation_fn,
            loss_fn,
            optimizer_fn,
            optimizer_params,
            metrics_fn,
            stop_step,
            "{}/train".format(out_dir),
            "{}/valid".format(out_dir),
            batch_size,
            restore_model_dir=restore_model_dir,
            transfer_model_dir=transfer_model_dir)

        # Early stopping and saving best model
        if metric_best is None or ('loss' in stop_metric) != (eval_metrics[stop_metric] > metric_best):
            consecutive_bad_epochs = 0
            metric_best = eval_metrics[stop_metric]
            with open(os.path.join(out_dir, 'best.txt'), 'w') as f:
                f.write('epoch %d\n'%epoch)
                f.write(str(eval_metrics))
        else:
            consecutive_bad_epochs += 1
            if consecutive_bad_epochs > patience:
                logging.info("early stopping triggered")
                break
    
    return None
