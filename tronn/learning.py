# basic wrappers for learning routines

import tensorflow as tf
import tensorflow.contrib.slim as slim


def train(data_loader, model_builder, loss_fn, optimizer_fn, optimizer_params,
          metrics_fn, restore, stopping_criterion, args, seq_length, num_tasks, OUT_DIR):
    '''
    Wraps the routines needed for tf-slim
    '''

    with tf.Graph().as_default() as training_graph:

        # data loader
        features, labels = data_loader(args.data_file, args.batch_size, seq_length, num_tasks)

        # model
        predictions = model_builder(features, labels, True) # Training = True

        # loss
        total_loss = loss_fn(predictions, labels)

        # optimizer
        optimizer = optimizer_fn(**optimizer_params)

        # train op
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # build metrics
        summary_op = metrics_fn(total_loss)

        if restore:
            checkpoint_path = tf.train.latest_checkpoint(OUT_DIR)
            variables_to_restore = slim.get_model_variables()
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, variables_to_restore)
            
            # create init assignment function
            def InitAssignFn(sess):
                sess.run(init_assign_op, init_feed_dict)
                
            slim.learning.train(train_op, OUT_DIR, init_fn=InitAssignFn, number_of_steps=100,
                                summary_op=summary_op, save_summaries_secs=20)

        else:
            slim.learning.train(train_op, OUT_DIR, number_of_steps=100,
                                summary_op=summary_op, save_summaries_secs=20)

    return None


def evaluate(data_loader, model_builder, metrics_fn, checkpoint_path, args, seq_length, num_tasks, out_dir):
    '''
    Note that if you want to reload a model, you must load the same model
    and data loader
    '''

    with tf.Graph().as_default() as evaluation_graph:

        # data loader
        features, labels = data_loader(args.data_file, args.batch_size, seq_length, num_tasks)
        labels = tf.cast(labels, tf.bool)

        # model
        predictions = tf.greater(tf.sigmoid(model_builder(features, labels, False)), 0.5) # Training = False

        # Choose the metrics to compute
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            "accuracy": slim.metrics.streaming_accuracy(predictions, labels),
            })

        for metric_name, metric_value in names_to_values.iteritems():
            tf.summary.scalar(metric_name, metric_value)

        # num batches to evaluate
        num_evals = 100

        # Evaluate the checkpoint
        slim.evaluation.evaluate_once('',
            checkpoint_path,
            out_dir,
            num_evals=num_evals,
            eval_op= names_to_updates.values())

    
    return None
