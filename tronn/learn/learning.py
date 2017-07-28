""" Contains light wrappers for learning and evaluation

The wrappers follow the tf-slim structure for setting up and running a model

"""

import os
import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_ops import restore_variables_op
from tronn.util.tf_utils import setup_tensorflow_session
from tronn.util.tf_utils import close_tensorflow_session
from tronn.util.tf_utils import get_checkpoint_steps
from tronn.util.tf_utils import add_summaries
from tronn.util.tf_utils import add_var_summaries
from tronn.util.tf_utils import make_summary_op
from tronn.util.tf_utils import print_param_count


def train(
        tronn_graph,
        stop_step,
        out_dir,
        restore_model_dir=None,
        transfer_model_dir=None):
    """Training routine utilizing tf-slim

    Args:
      tronn_graph: a TronnNeuralNetGraph instance
      stop_step: global stepping stop point
      out_dir: where to save the model and events
      restore_model_dir: restore EXACT same model
      transfer_model_dir: transfer EXACT model minus output nodes

    Returns:
      None
    """
    logging.info("Training until {} steps".format(str(stop_step)))
    assert not ((restore_model_dir is not None)
                and (transfer_model_dir is not None))
    
    with tf.Graph().as_default():

        # build graph
        train_op = tronn_graph.build_training_graph(data_key="train")
        
        # summaries
        metric_values = tronn_graph.metric_values
        add_summaries(metric_values)
        for var in tf.model_variables(): add_var_summaries(var)
        summary_op = make_summary_op(tronn_graph.metric_values,
                                     print_out=True)
        
        # print parameter count
        if (restore_model_dir is None) and (transfer_model_dir is None):
            print 'Created new model:'
            print_param_count()

        # Generate an initial assign op if restoring/transferring
        if restore_model_dir is not None:
            init_assign_op, init_feed_dict = restore_variables_op(
                restore_model_dir)
            def restoreFn(sess):
                sess.run(init_assign_op, init_feed_dict)
        elif transfer_model_dir is not None:
            init_assign_op, init_feed_dict = restore_variables_op(
                restore_transfer_dir, skip=['logit','out'])
            def restoreFn(sess):
                sess.run(init_assign_op, init_feed_dict)
        else:
            restoreFn = None

        # tf-slim to train
        slim.learning.train(
            train_op,
            out_dir,
            init_fn=restoreFn,
            number_of_steps=stop_step,
            summary_op=summary_op,
            save_summaries_secs=60,
            saver=tf.train.Saver(max_to_keep=None),
            save_interval_secs=3600)

    return


def evaluate(
        tronn_graph,
        out_dir,
        model_dir,
        stop_step):
    """Evaluation routine using tf-slim

    Args:
      tronn_graph: a TronnNeuralNetGraph instance
      out_dir: output directory
      model_dir: directory with trained model
      stop_step: how many evals to run

    Returns:
      metrics_dict: dictionary of metrics from metrics fn
    """
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    logging.info('evaluating %s...'%checkpoint_path)
    
    with tf.Graph().as_default():

        # build graph
        tronn_graph.build_evaluation_graph(data_key="valid")

        # add summaries
        add_summaries(tronn_graph.metric_values)
        summary_op = make_summary_op(tronn_graph.metric_values)

        # evaluate the checkpoint
        metrics_dict = slim.evaluation.evaluate_once(
            None,
            checkpoint_path,
            out_dir,
            num_evals=stop_step,
            summary_op=summary_op,
            eval_op=tronn_graph.metric_updates,
            final_op=tronn_graph.metric_values)
        
        print 'Validation metrics:\n%s'%metrics_dict
    
    return metrics_dict


def train_and_evaluate_once(
        tronn_graph,
        train_stop_step,
        train_dir,
        valid_dir,
        restore_model_dir=None,
        transfer_model_dir=None,
        valid_stop_step=10000): # 10k
    """Routine to train and evaluate for some number of steps
    
    Args:
      tronn_graph: a TronnNeuralNetGraph instance
      train_stop_step: number of train steps to run
      out_dir: output directory (fn makes train/valid dirs)
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
        tronn_graph,
        train_stop_step,
        train_dir,
        restore_model_dir=restore_model_dir,
        transfer_model_dir=transfer_model_dir)

    # Evaluate after training (use for stopping criteria)
    eval_metrics = evaluate(
        tronn_graph,
        valid_dir,
        train_dir,
        valid_stop_step)

    return eval_metrics


def train_and_evaluate(
        tronn_graph,
        out_dir,
        train_steps,
        stop_metric,
        patience,
        epoch_limit,
        restore_model_dir=None,
        transfer_model_dir=None):
    """Runs training and evaluation for {epoch_limit} epochs

    Args:
      tronn_graph: a TronnNeuralNetGraph instance
      out_dir: where to save outputs
      train_steps: number of train steps to run before evaluating
      stop_metric: metric used for early stopping
      patience: number of epochs to wait for improvement
      epoch_limit: number of max epochs
      restore_model_dir: location of a checkpoint that is
        EXACTLY the same as the train model
      transfer_model_dir: location of a checkpoint that is
        exactly the same as the train model EXCEPT for the
        last layer (logits)

    Returns:
      None
    """
    assert not ((restore_model_dir is not None)
                and (transfer_model_dir is not None))
    
    # track metric and bad epochs
    metric_best = None
    consecutive_bad_epochs = 0

    for epoch in xrange(epoch_limit):
        logging.info("CURRENT EPOCH:", str(epoch))

        if epoch > 0:
            # make sure that transfer_model_dir is None
            # and restore_model_dir is set correctly
            transfer_model_dir = None
            restore_model_dir = "{}/train".format(out_dir)

        # set up stop steps and adjust if coming from a transfer
        stop_step = (epoch + 1) * train_steps
        # adjust if coming from a transfer
        if transfer_model_dir is not None:
            stop_step = get_checkpoint_steps(transfer_model_dir)

        # train and evaluate one epoch
        eval_metrics = train_and_evaluate_once(
            tronn_graph,
            stop_step,
            "{}/train".format(out_dir),
            "{}/valid".format(out_dir),
            restore_model_dir=restore_model_dir,
            transfer_model_dir=transfer_model_dir)

        # Early stopping and saving best model
        if metric_best is None or ('loss' in stop_metric) != (eval_metrics[stop_metric] > metric_best):
            consecutive_bad_epochs = 0
            metric_best = eval_metrics[stop_metric]
            with open(os.path.join(out_dir, 'best.txt'), 'w') as fp:
                fp.write('epoch %d\n'%epoch)
                fp.write(str(eval_metrics))
        else:
            consecutive_bad_epochs += 1
            if consecutive_bad_epochs > patience:
                logging.info("early stopping triggered")
                break
    
    return None


def predict(
        tronn_graph,
        model_dir,
        batch_size,
        num_evals=1000):
    """Prediction routine. When called, returns predictions 
    (with labels and metadata) as an array for downstream processing

    Args:
      tronn_graph: a TronnNeuralNetGraph instance
      model_dir: folder containing trained model
      batch_size: batch size
      num_evals: number of examples to run

    Returns:
      label_array
      logit_array
      probs_array
      metadata_array

    """
    # build graph and run
    with tf.Graph().as_default():

        # build graph
        label_tensor, logit_tensor, probs_tensor = tronn_graph.build_graph()
        metadata_tensor = tronn_graph.metadata
        
        # set up session
        sess, coord, threads = setup_tensorflow_session()
        
        # restore
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        
        # set up arrays to hold outputs
        num_examples = num_evals * batch_size
        all_labels_array = np.zeros((num_examples, label_tensor.get_shape()[1]))
        all_logits_array = np.zeros((num_examples, logit_tensor.get_shape()[1]))
        all_probs_array = np.zeros((num_examples, probs_tensor.get_shape()[1]))
        all_metadata = []
        
        batch_start = 0
        batch_end = batch_size
        # TODO turn this into an example producer (that has the option of merging into regions or not)
        for i in range(num_evals):
            labels, logits, probs, metadata = sess.run([label_tensor,
                                                        logit_tensor,
                                                        probs_tensor,
                                                        metadata_tensor])

            # put into numpy arrays
            all_labels_array[batch_start:batch_end,:] = labels
            all_logits_array[batch_start:batch_end,:] = logits
            all_probs_array[batch_start:batch_end,:] = probs

            # metadata: convert to list and add
            metadata_list = metadata.tolist()
            metadata_string_list = [metadata_piece[0].split('(')[0] for metadata_piece in metadata_list]
            all_metadata = all_metadata + metadata_string_list
            
            # move batch pointer
            batch_start = batch_end
            batch_end += batch_size

        close_tensorflow_session(coord, threads)
    
    return all_labels_array, all_logits_array, all_probs_array, all_metadata
