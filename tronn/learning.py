""" Contains light wrappers for learning and evaluation

The wrappers follow the tf-slim structure for setting up and running a model

"""
import evaluation
import tf_utils
import config

import tensorflow as tf
import tensorflow.contrib.slim as slim


def train(data_loader,
          model_builder,
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

    with tf.Graph().as_default():

        # data loader
        features, labels, metadata = data_loader(data_file_list, args.batch_size, args.tasks)

        # model
        logits = model_builder(features, labels, args.model, is_training=True)
        probabilities = final_activation_fn(logits)
        loss = loss_fn(labels, logits)

        # metrics
        metric_value, metric_update = tf.contrib.metrics.aggregate_metric_map({'mean_loss': tf.contrib.metrics.streaming_mean(loss),
                                                                                'auroc': tf.contrib.metrics.streaming_auc(probabilities, labels, curve='ROC'),
                                                                                'auprc': tf.contrib.metrics.streaming_auc(probabilities, labels, curve='PR'),
                                                                                'accuracy': tf.contrib.metrics.streaming_accuracy(tf.cast(tf.greater(probabilities, 0.5), 'float32'), labels)})
        for update in metric_update.values():
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)
        for metric, value in metric_value.iteritems():
            tf.summary.scalar('train/%s'%metric, value)
        
        # train op
        total_loss = tf.losses.get_total_loss()
        optimizer = optimizer_fn(**optimizer_params)
        train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

        # summaries
        tf.summary.scalar('train/loss', loss)
        for metric, value in metric_value.iteritems():
            tf.summary.scalar('train/%s'%metric, value)
        for var in tf.model_variables(): tf_utils.add_var_summaries(var)
        values_for_printing = [tf.train.get_global_step(), metric_value['mean_loss'], metric_value['accuracy'], metric_value['auprc'], metric_value['auroc'], loss, total_loss]
        summary_op = tf.Print(tf.summary.merge_all(), values_for_printing)

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
            for var in sorted(tf.trainable_variables(), key=lambda var: (var.name, var.get_shape().num_elements())):
                num_elems = var.get_shape().num_elements()
                if num_elems>500:
                    print var.name, var.get_shape().as_list(), num_elems
            print 'Num params (model/trainable/global): %d/%d/%d' % (model_params, trainable_params, total_params)

        slim.learning.train(train_op,
                            OUT_DIR,
                            init_fn=restoreFn if restore else None,
                            number_of_steps=target_global_step,
                            summary_op=summary_op,
                            save_summaries_secs=60,
                            save_interval_secs=3600,
                            session_config=config.session_config)

def evaluate(data_loader,
             model_builder,
             final_activation_fn,
             loss_fn,
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
    print 'evaluating %s...'%checkpoint_path
    with tf.Graph().as_default():

        # data loader
        features, labels, metadata = data_loader(data_file_list, args.batch_size*4, args.tasks)

        # model - training=False
        logits = model_builder(features, labels, args.model, is_training=False)
        probabilities = final_activation_fn(logits)
        loss = loss_fn(labels, logits)

        # metrics
        metric_value, metric_update = tf.contrib.metrics.aggregate_metric_map({'mean_loss': tf.contrib.metrics.streaming_mean(loss),
                                                                                'auroc': tf.contrib.metrics.streaming_auc(probabilities, labels, curve='ROC'),
                                                                                'auprc': tf.contrib.metrics.streaming_auc(probabilities, labels, curve='PR'),
                                                                                'accuracy': tf.contrib.metrics.streaming_accuracy(tf.cast(tf.greater(probabilities, 0.5), 'float32'), labels)})
        for update in metric_update.values():
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)
        for metric, value in metric_value.iteritems():
            tf.summary.scalar('train/%s'%metric, value)

        # Evaluate the checkpoint
        metrics_dict = slim.evaluation.evaluate_once(
            None,
            checkpoint_path,
            out_dir,
            num_evals=num_evals,
            summary_op=tf.summary.merge_all(),
            eval_op=metric_update.values(),
            final_op=metric_value,
            session_config=config.session_config)
        print 'Validation metrics:\n%s'%metrics_dict
    
    return metrics_dict

