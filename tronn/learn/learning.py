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

from tronn.interpretation.regions import ExampleGenerator


def train(
        tronn_graph,
        stop_step,
        out_dir,
        restore_model_checkpoint=None,
        transfer_model_checkpoint=None):
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
    assert not ((restore_model_checkpoint is not None)
                and (transfer_model_checkpoint is not None))
    
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
        if (restore_model_checkpoint is None) and (transfer_model_checkpoint is None):
            print 'Created new model:'
            print_param_count()

        # Generate an initial assign op if restoring/transferring
        if restore_model_checkpoint is not None:
            init_assign_op, init_feed_dict = restore_variables_op(
                restore_model_checkpoint)
            def restoreFn(sess):
                sess.run(init_assign_op, init_feed_dict)
        elif transfer_model_checkpoint is not None:
            init_assign_op, init_feed_dict = restore_variables_op(
                transfer_model_checkpoint, skip=['logit','out'])
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
            save_interval_secs=0) # change this?

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
        restore_model_checkpoint=None,
        transfer_model_checkpoint=None,
        valid_stop_step=100): # 10k
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
        restore_model_checkpoint=restore_model_checkpoint,
        transfer_model_checkpoint=transfer_model_checkpoint)

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
        epoch_limit=10,
        restore_model_checkpoint=None,
        transfer_model_checkpoint=None):
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
    assert not ((restore_model_checkpoint is not None)
                and (transfer_model_checkpoint is not None))
    
    # track metric and bad epochs and steps
    metric_best = None
    consecutive_bad_epochs = 0
    step_log = "{}/train/step.log".format(out_dir)
    stopping_log = "{}/train/stopping.log".format(out_dir)

    for epoch in xrange(epoch_limit):
        logging.info("CURRENT EPOCH:", str(epoch))
        print "CURRENT EPOCH:", epoch

        # change model checkpoint as needed
        if epoch > 0:
            # make sure that transfer_model_dir is None
            # and restore_model_dir is set correctly
            transfer_model_checkpoint = None
            restore_model_checkpoint = tf.train.latest_checkpoint(
                "{}/train".format(out_dir))

        # if a log file does not exist, you're freshly in the folder! instantiate and set init_steps
        if not os.path.isfile(step_log):
            if transfer_model_checkpoint is not None:
                init_steps = get_checkpoint_steps(transfer_model_checkpoint)
            else:
                init_steps = 0
            with open(step_log, "w") as out:
                out.write("{}\t{}".format(init_steps, init_steps)) # NOTE: may not need to write last step, may not be used
        else:
            with open(step_log, "r") as fp:
                init_steps, last_steps = map(int, fp.readline().strip().split())
            # also if there is a log file, check for a model checkpoint! if so, set and remove transfer
            restore_model_checkpoint = tf.train.latest_checkpoint(
                "{}/train".format(out_dir))
            if restore_model_checkpoint is not None:
                transfer_model_checkpoint = None

        # set stop step for epoch
        stop_step = init_steps + (epoch+1)*train_steps

        # train and evaluate one epoch
        eval_metrics = train_and_evaluate_once(
            tronn_graph,
            stop_step,
            "{}/train".format(out_dir),
            "{}/valid".format(out_dir),
            restore_model_checkpoint=restore_model_checkpoint,
            transfer_model_checkpoint=transfer_model_checkpoint)
        
        # refresh log with latest step count (counting from start of folder, NOT including transfer)
        with open(step_log, "w") as out:
            out.write("{}\t{}".format(init_steps, stop_step)) # may not need to write stop step, may not be used

        # check for stopping log details.
        if os.path.isfile(stopping_log):
            with open(stopping_log, "r") as fp:
                best_epoch, metric_best = map(float, fp.readline().strip().split())
            consecutive_bad_epochs = epoch - best_epoch

        # Early stopping and saving best model
        if metric_best is None or ('loss' in stop_metric) != (eval_metrics[stop_metric] > metric_best):
            consecutive_bad_epochs = 0
            metric_best = eval_metrics[stop_metric]
            checkpoint_path = tf.train.latest_checkpoint("{}/train".format(out_dir))
            with open(os.path.join(out_dir, 'best.txt'), 'w') as fp:
                fp.write('epoch %d\n'%epoch)
                fp.write("checkpoint path: {}\n".format(checkpoint_path))
                fp.write(str(eval_metrics))
            with open(stopping_log, 'w') as out:
                out.write("{}\t{}".format(epoch, metric_best))
        else:
            consecutive_bad_epochs += 1
            if consecutive_bad_epochs > patience:
                print "early stopping triggered"
                logging.info("early stopping triggered")
                break

    return restore_model_checkpoint


def predict_old(
        tronn_graph,
        model_dir,
        batch_size,
        num_evals=1000,
        merge_regions=False):
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
        
        # restore if given model (option to NOT restore because of models
        # that do not use restore, like PWM convolutions)
        if model_dir is not None:
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


def predict(
        tronn_graph,
        model_dir,
        batch_size,
        model_checkpoint=None,
        num_evals=1000,
        reconstruct_regions=False):
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
        
        # restore if given model (option to NOT restore because of models
        # that do not use restore, like PWM convolutions)
        if model_checkpoint is not None:
            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint)
        elif model_dir is not None:
            checkpoint_path = tf.train.latest_checkpoint(model_dir)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            
        # set up arrays to hold outputs
        num_examples = num_evals
        all_labels_array = np.zeros((num_examples, label_tensor.get_shape()[1]))
        all_logits_array = np.zeros((num_examples, logit_tensor.get_shape()[1]))
        all_probs_array = np.zeros((num_examples, probs_tensor.get_shape()[1]))
        all_metadata = []
        
        batch_start = 0
        batch_end = batch_size

        tensor_dict = {
            "labels": label_tensor,
            "logits": logit_tensor,
            "probs": probs_tensor,
            "feature_metadata": metadata_tensor}
        example_generator = ExampleGenerator(
            sess,
            tensor_dict,
            batch_size,
            reconstruct_regions=reconstruct_regions)

        for i in range(num_evals):

            region, region_arrays = example_generator.run()
            labels = region_arrays["labels"]
            logits = region_arrays["logits"]
            probs = region_arrays["probs"]

            # put into numpy arrays
            all_labels_array[i,:] = labels
            all_logits_array[i,:] = logits
            all_probs_array[i,:] = probs

            # metadata: convert to list and add
            #metadata_list = [region]
            #metadata_string_list = [metadata_piece[0].split('(')[0] for metadata_piece in metadata_list]
            #all_metadata = all_metadata + metadata_string_list
            all_metadata.append(region)

        close_tensorflow_session(coord, threads)
    
    return all_labels_array, all_logits_array, all_probs_array, all_metadata




