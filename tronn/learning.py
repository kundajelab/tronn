""" Contains light wrappers for learning and evaluation

The wrappers follow the tf-slim structure for setting up and running a model

"""

import tronn

import tensorflow as tf
import tensorflow.contrib.slim as slim

def train(data_loader,
          model_builder,
          model_config,
          final_activation_fn,
          loss_fn,
          optimizer_fn,
          optimizer_params,
          restore,
          stopping_criterion,
          args,
          data_file_list,
          OUT_DIR,
          target_global_step):
    '''
    Wraps the routines needed for tf-slim
    '''

    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader(data_file_list,
                                                 args.batch_size)

        # model
        logits = model_builder(features, labels, model_config, is_training=True)
        loss = loss_fn(labels, logits)
        names_to_metrics, updates = tronn.evaluation.get_metrics(13, logits, labels, final_activation_fn, loss_fn)
        for update in updates: tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)

        # optimizer
        optimizer = optimizer_fn(**optimizer_params)

        # train op
        total_loss = tf.losses.get_total_loss()
        train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

        # summaries
        tf.summary.scalar('train/loss', loss)
        for name, metric in names_to_metrics.iteritems():
            if metric.get_shape().ndims==0:
                tf.summary.scalar('train/%s'%name, metric)
            else:
                tf.summary.histogram('train/%s'%name, metric)
        for var in tf.model_variables(): tronn.nn_utils.add_var_summaries(var)

        def restoreFn(sess):
            checkpoint_path = tf.train.latest_checkpoint(OUT_DIR)
            restorer = tf.train.Saver()
            print 'Restoring model from %s...'%checkpoint_path
            restorer.restore(sess, checkpoint_path)
        
        # print parameter count
        if not restore:
            print 'Created new model:'
            model_params = sum(v.get_shape().num_elements() for v in tf.model_variables())
            trainable_params = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
            total_params = sum(v.get_shape().num_elements() for v in tf.global_variables())
            for var in sorted(tf.model_variables(), key=lambda var: (var.name, var.get_shape().num_elements())):
                num_elems = var.get_shape().num_elements()
                if num_elems>500:
                    print var.name, var.get_shape().as_list(), num_elems
            print 'Num params (model/trainable/global): %d/%d/%d' % (model_params, trainable_params, total_params)

        imp_metrics = [names_to_metrics[name] for name in ['mean_auPRC', 'mean_auROC', 'mean_accuracy', 'mean_loss']]
        summary_op = tf.Print(tf.summary.merge_all(), [tf.train.get_global_step()] + imp_metrics + [loss, total_loss])
        slim.learning.train(train_op,
                            OUT_DIR,
                            init_fn=restoreFn if restore else None,
                            number_of_steps=target_global_step,
                            summary_op=summary_op,
                            save_summaries_secs=60,
                            save_interval_secs=3600)

    return None


def evaluate(data_loader,
             model_builder,
             model_config,
             final_activation_fn,
             loss_fn,
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
        features, labels, metadata = data_loader(data_file_list,
                                                 args.batch_size*2)#increase batch size since we don't need back-prop

        # model - training=False
        logits = model_builder(features, labels, args, model_config, is_training=False)
        
        # Construct metrics to compute
        names_to_metrics, updates = tronn.evaluation.get_metrics(13, logits, labels, final_activation_fn, loss_fn)#13 days(tasks)

        # Define the scalar summaries to write
        for name, metric in names_to_metrics.iteritems():
            if metric.get_shape().ndims==0:
                tf.summary.scalar(name, metric)
            else:
                tf.summary.histogram(name, metric)

        # Evaluate the checkpoint
        metrics_dict = slim.evaluation.evaluate_once(
            None,
            checkpoint_path,
            out_dir,
            num_evals=num_evals,
            summary_op=tf.summary.merge_all(),
            eval_op=updates,
            final_op=names_to_metrics)
        print 'Validation metrics:\n%s'%metrics_dict
    
    return None
