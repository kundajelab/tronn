""" Contains light wrappers for learning and evaluation

The wrappers follow the tf-slim structure for setting up and running a model

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops

#from tensorflow.contrib.training.python.training import evaluation

# def init_train(data_loader, model_builder, loss_fn, optimizer_fn, optimizer_params,
#           metrics_fn, stopping_criterion, args, seq_length, num_tasks, OUT_DIR, training_graph):

#     with training_graph.as_default() as g:

#         # data loader
#         features, labels = data_loader(args.data_file, args.batch_size, seq_length, num_tasks)

#         # model
#         predictions = model_builder(features, labels, True) # Training = True

#         # loss
#         total_loss = loss_fn(predictions, labels)

#         # optimizer
#         optimizer = optimizer_fn(**optimizer_params)

#         # train op
#         train_op = slim.learning.create_train_op(total_loss, optimizer)

#         # build metrics
#         summary_op = metrics_fn(total_loss)


#     return training_graph, train_op, summary_op


# def train_on_graph(training_graph, train_op, summary_op, restore, OUT_DIR):

#     if restore:
#         checkpoint_path = tf.train.latest_checkpoint(OUT_DIR)
#         variables_to_restore = slim.get_model_variables()
#         init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, variables_to_restore)
        
#         # create init assignment function
#         def InitAssignFn(sess):
#             sess.run(init_assign_op, init_feed_dict)
            
#         slim.learning.train(train_op, OUT_DIR, graph=training_graph, init_fn=InitAssignFn, number_of_steps=100,
#                             summary_op=summary_op, save_summaries_secs=20)

#     else:
#         slim.learning.train(train_op, OUT_DIR, graph=training_graph, number_of_steps=100,
#                             summary_op=summary_op, save_summaries_secs=20)

#     return None


# def make_global_step(start_val, graph=None):
#     '''
#     Taken from slim variables but modified to start at another step value
#     '''

#     graph = ops.get_default_graph() if graph is None else graph
#     # Create in proper graph and base name_scope.
#     with graph.as_default() as g, g.name_scope(None):
#         collections = [ops.GraphKeys.GLOBAL_STEP]
#         return slim.variable(
#             ops.GraphKeys.GLOBAL_STEP,
#             shape=[],
#             dtype=tf.int64,
#             initializer=init_ops.constant_initializer(start_val),
#             trainable=False,
#             collections=collections)



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
          seq_length,
          num_tasks,
          OUT_DIR,
          global_step_val):
    '''
    Wraps the routines needed for tf-slim
    '''

    with tf.Graph().as_default() as g:

        # data loader
        features, labels = data_loader(data_file_list,
                                       args.batch_size,
                                       seq_length,
                                       num_tasks)

        # model
        predictions = model_builder(features, labels, True) # Training = True

        # loss
        # TODO check that this loss is right
        classification_loss = loss_fn(predictions, labels)
        total_loss = slim.losses.get_total_loss()



        # optimizer
        optimizer = optimizer_fn(**optimizer_params)

        # train op
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # build metrics
        summary_op = metrics_fn(total_loss, predictions, labels)

        #tf.Print(slim.get_global_step(), [slim.get_global_step()])

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
             seq_length,
             num_tasks,
             out_dir):
    '''
    Note that if you want to reload a model, you must load the same model
    and data loader
    '''

    with tf.Graph().as_default() as g:

        # data loader
        features, labels = data_loader(data_file_list,
                                       args.batch_size,
                                       seq_length,
                                       num_tasks)
        labels = tf.cast(labels, tf.bool)

        # model
        predictions = tf.greater(tf.sigmoid(model_builder(features,
                                                          labels,
                                                          False)),
                                 0.5) # Training = False
        
        # Choose the metrics to compute
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            "accuracy": slim.metrics.streaming_accuracy(predictions, labels),
            })

        # Define the summaries to write
        for metric_name, metric_value in names_to_values.iteritems():
            tf.scalar_summary(metric_name, metric_value)

        # num batches to evaluate
        num_evals = 1000

        # Evaluate the checkpoint
        a = slim.evaluation.evaluate_once('',
                                          checkpoint_path,
                                          out_dir,
                                          num_evals=num_evals,
                                          summary_op=tf.merge_all_summaries(),
                                          eval_op= names_to_updates.values(),
                                          final_op=names_to_values)
        
        print a
    
    return None
