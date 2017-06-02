""" Contains light wrappers for learning and evaluation

The wrappers follow the tf-slim structure for setting up and running a model

"""
import tf_utils

import tensorflow as tf
import tensorflow.contrib.slim as slim


def train(data_loader,
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
    '''
    Wraps the routines needed for tf-slim
    '''

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
        values_for_printing = [tf.train.get_global_step(), metric_value['mean_loss'], metric_value['mean_accuracy'], metric_value['mean_auprc'], metric_value['mean_auroc'], loss, total_loss]
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
            def restoreFn(sess):
                sess.run(init_assign_op, init_feed_dict)

        slim.learning.train(train_op,
                            OUT_DIR,
                            init_fn=restoreFn if restore else None,
                            number_of_steps=target_global_step,
                            summary_op=summary_op,
                            save_summaries_secs=60,
                            save_interval_secs=3600,)

def evaluate(data_loader,
             model_builder,
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

        # metrics
        metric_value, updates = metrics_fn(labels, probabilities, args.tasks)
        for update in updates: tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)

        metric_value['loss'] = loss
        metric_value['total_loss'] = total_loss
        tf.nn_utils.add_summaries(metric_value)

        # Evaluate the checkpoint
        metrics_dict = slim.evaluation.evaluate_once(
            None,
            checkpoint_path,
            out_dir,
            num_evals=num_evals,
            summary_op=tf.summary.merge_all(),
            eval_op=metric_update.values(),
            final_op=metric_value,)
        print 'Validation metrics:\n%s'%metrics_dict
    
    return metrics_dict

