""" Contains light wrappers for learning and evaluation

The wrappers follow the tf-slim structure for setting up and running a model

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim


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
          global_step_val):
    '''
    Wraps the routines needed for tf-slim
    '''

    with tf.Graph().as_default() as g:

        # data loader
        features, labels = data_loader(data_file_list,
                                       args.batch_size)

        # model
        predictions = model_builder(features, labels, is_training=True)

        # loss
        total_loss = loss_fn(predictions, labels)

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
                                save_summaries_secs=20)

        else:
            slim.learning.train(train_op,
                                OUT_DIR,
                                number_of_steps=global_step_val,
                                summary_op=summary_op,
                                save_summaries_secs=20)

    return None


def evaluate(data_loader,
             model_builder,
             metrics_fn,
             checkpoint_path,
             args,
             data_file_list,
             out_dir,
             num_evals=10000):
    '''
    Wrapper function for doing evaluation (ie getting metrics on a model)
    Note that if you want to reload a model, you must load the same model
    and data loader
    '''

    with tf.Graph().as_default() as g:

        # data loader
        features, labels = data_loader(data_file_list,
                                       args.batch_size)
        labels_bool = tf.cast(labels, tf.bool)

        # model - training=False
        prediction_probs = tf.nn.sigmoid(model_builder(features, labels, is_training=False))

        # classification predictions - note that this assumes a model
        # with sigmoid NOT softmax.
        # TODO factor this out
        prediction_bools = tf.greater(prediction_probs, 0.5)
        
        # Choose the metrics to compute
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            "accuracy": slim.metrics.streaming_accuracy(prediction_bools, labels_bool),
            "auROC": slim.metrics.streaming_auc(prediction_probs, labels, 
                                                curve="ROC"),
            "auPRC": slim.metrics.streaming_auc(prediction_probs, labels, 
                                                curve="PR"),
            })

        # Define the scalar summaries to write
        for metric_name, metric_value in names_to_values.iteritems():
            tf.scalar_summary(metric_name, metric_value)

        # Evaluate the checkpoint
        metrics_dict = slim.evaluation.evaluate_once('',
                                                     checkpoint_path,
                                                     out_dir,
                                                     num_evals=num_evals,
                                                     summary_op=tf.merge_all_summaries(),
                                                     eval_op= names_to_updates.values(),
                                                     final_op=names_to_values)
        
        print metrics_dict
    
    return None
