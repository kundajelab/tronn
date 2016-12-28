""" Contains functions for evaluation and summarization of metrics

"""

import os
import math
import models
import tensorflow as tf
import numpy as np
import h5py

from sklearn import metrics as skmetrics
from random import shuffle


def streaming_metrics_tronn(total_loss, predictions, labels):

    tf.scalar_summary('loss', total_loss)


    # See weights
    weights = [v for v in tf.all_variables()
               if ('weights' in v.name)]

    weight_sum = tf.add_n([ tf.reduce_sum(w) for w in weights ])

    tf.scalar_summary('weight_sum', weight_sum)

    tf.scalar_summary('predictions', tf.reduce_sum(predictions))
        

    summary_op = tf.merge_all_summaries()

    return summary_op



def get_metrics(tasks, predictions, labels):
    '''
    Set up streaming metrics
    '''

    metric_updates = []
    auROC_tensors = []
    auPRC_tensors = []
    accuracy_tensors = []

    with tf.name_scope('metrics') as scope:
        for task_num in range(tasks):
            with tf.name_scope('roc'):
                auROC, update_op_auROC = tf.contrib.metrics.streaming_auc(predictions[:,task_num], labels[:,task_num], curve='ROC', name='roc{}'.format(task_num))
                auROC_tensors.append(auROC)
                metric_updates.append(update_op_auROC)
        
            with tf.name_scope('accuracy'):
                accuracy, update_op_accuracy = tf.contrib.metrics.streaming_accuracy(tf.cast(tf.greater(predictions[:,task_num], 0.5), 'float32'), 
                                                                                     labels[:,task_num], name='acc{}'.format(task_num))
                accuracy_tensors.append(accuracy)
                metric_updates.append(update_op_accuracy)

            with tf.name_scope('pr'):
                auPRC, update_op_auPRC = tf.contrib.metrics.streaming_auc(predictions[:,task_num], labels[:,task_num], curve='PR', name='pr{}'.format(task_num))
                auPRC_tensors.append(auPRC)
                metric_updates.append(update_op_auPRC)

        learning_metrics = {
            'scalar': {
                'mean_accuracy' : tf.reduce_mean(tf.pack(accuracy_tensors), name='mean_accuracy'),
                'mean_auROC' : tf.reduce_mean(tf.pack(auROC_tensors), name='mean_auROC'),
                'mean_auPRC' : tf.reduce_mean(tf.pack(auPRC_tensors), name='mean_auPRC')
            },
            'histogram': {
                'accuracies' : tf.pack(accuracy_tensors),
                'auROCs' : tf.pack(auROC_tensors),
                'auPRCs' : tf.pack(auPRC_tensors)
            },
        }

    return learning_metrics, metric_updates


def streaming_evaluate(sess, current_model_state, summary_writer, metric_updates, loss_sum, merged,
                       label_batch, train_prediction, model_state, tasks, total_examples_run, num_batches_to_eval, out_prefix):
    '''
    Generalized function that runs data for some number of iterations and returns key metrics of interest (auROC avg, per task, etc)
    '''

    # Run total data (same for train and validation, except for the model_state)
    losses = []
    for batch in range(num_batches_to_eval):
        _, summed_loss, summary, labels,  predictions = sess.run([metric_updates, 
                                                                  loss_sum,
                                                                  merged, 
                                                                  label_batch,
                                                                  train_prediction],
                                                                 feed_dict={model_state:current_model_state})
        losses.append(summed_loss)
        try:
            predictions_total = np.vstack([predictions_total, predictions])
            labels_total = np.vstack([labels_total, labels])
        except:
            predictions_total = predictions
            labels_total = labels

    # Calculate various metrics: right now, average auROC and task auROCs, average loss
    task_aurocs = []
    for i in xrange(tasks):
        try:
            task_aurocs.append(skmetrics.roc_auc_score(labels_total[:,i], predictions_total[:,i]))
        except:
            task_aurocs.append(0)

    auroc_mean = np.mean(np.array(task_aurocs)[np.nonzero(task_aurocs)])

    with open('{}_auROC_mean.txt'.format(out_prefix), 'a') as fp:
        fp.write('{}\n'.format(auroc_mean))
    with open('{}_auROC_tasks.txt'.format(out_prefix), 'a') as fp:
        fp.write('{}\n'.format('\t'.join([str(i) for i in task_aurocs])))

    # Also calculate task auPRCs and average auPRC
    task_auprcs = []
    for i in xrange(tasks):
        try:
            precision, recall = skmetrics.precision_recall_curve(labels_total[:,i], predictions_total[:,i])[:2]
            task_auprcs.append(skmetrics.auc(recall, precision))
        except:
            task_auprcs.append(0)

    auprc_mean = np.mean(np.array(task_auprcs)[np.nonzero(task_auprcs)])
    
    with open('{}_auPRC_mean.txt'.format(out_prefix), 'a') as fp:
        fp.write('{}\n'.format(auprc_mean))
    with open('{}_auPRC_tasks.txt'.format(out_prefix), 'a') as fp:
        fp.write('{}\n'.format('\t'.join([str(i) for i in task_auprcs])))
        
    # Now write out to summary writer
    summary_writer.add_summary(summary, total_examples_run)
    summary_writer.flush()
    
    return np.mean(losses), auroc_mean, auprc_mean


def make_tensorboard_metrics(sess, learning_metrics, LOG_DIR):
    '''
    Set up summaries and writers
    '''
    for metric in learning_metrics['scalar'].keys():
        tf.scalar_summary(metric, learning_metrics['scalar'][metric])
    for metric in learning_metrics['histogram'].keys():
        tf.histogram_summary(metric, learning_metrics['histogram'][metric])
    
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(LOG_DIR + '/train', sess.graph)
    valid_writer = tf.train.SummaryWriter(LOG_DIR + '/valid')
    
    return merged, train_writer, valid_writer


def evaluate(out_h5_file, sess, maxnorm_ops, convlayer_relu_1, model_state, evaluation_sample_size, batch_size):
    '''
    Perform final evaluations on test dataset. This is specifically an evaluate to feed into Basset's
    post-processing framework.
    '''
    
    with h5py.File(out_h5_file, 'w') as hf:
        
        # Store weights for first convlayer
        hf.create_dataset('weights', (300, 4, 19)) # TODO: these variables need to be factored out
        weights = hf.get('weights')
        
        model_weights = sess.run(maxnorm_ops[0], feed_dict={model_state:'test'})
        first_layer_weights = np.squeeze(np.moveaxis(model_weights, [1, 2, 3], [3, 2, 1]))
        weights[:,:,:] = first_layer_weights
        
        # Store activations for {sample_size} examples
        hf.create_dataset('outs', (evaluation_sample_size, 300, 582)) # TODO: these variables need to be factored out
        outs = hf.get('outs')
            
        # Run through the sample size worth of samples (plus a little extra)
        batch_start, batch_end = 0, batch_size
        for i in range(int(math.ceil(evaluation_sample_size / batch_size))):
            print "evaluating batch {}".format(str(i))
            first_layer_activations = sess.run(convlayer_relu_1, feed_dict={model_state:'test'})
            reformatted_activations = np.squeeze(np.moveaxis(first_layer_activations, [2, 3], [3, 2]))
            
            # Store into hdf5 file
            # TODO: also save out sequences (if you start doing random) so that sequences exactly match what you expect.
            if batch_end < evaluation_sample_size:
                outs[batch_start:batch_end,:,:] = reformatted_activations
            else:
                outs[batch_start:evaluation_sample_size,:,:] = reformatted_activations[0:(evaluation_sample_size - batch_end),:,:]
                
            batch_start = batch_end
            batch_end += batch_size

    return None


def evaluate_seq_importances(out_h5_file, sess, importance, train_prediction, label_batch, metadata_batch, model_state, evaluation_sample_size, batch_size, seq_length, tasks):
    '''
    Get importance scores in sequence. Currently built as (input)*(activation gradient)
    '''

    with h5py.File(out_h5_file, 'w') as hf:

        # Create datasets
        hf.create_dataset('importances', (evaluation_sample_size, 1, seq_length, 4)) # NHWC
        importances_hf = hf.get('importances')

        hf.create_dataset('predictions', (evaluation_sample_size, tasks))
        predictions_hf = hf.get('predictions')

        hf.create_dataset('labels', (evaluation_sample_size, tasks))
        labels_hf = hf.get('labels')

        regions_hf = hf.create_dataset('regions', (evaluation_sample_size, 1))
        
        # Run through sample size worth of samples (plut a little extra)
        # and save the importance as well as the predictions and labels (because you want to look at things predicted correctly)
        batch_start, batch_end = 0, batch_size
        for i in range(int(math.ceil(evaluation_sample_size / batch_size))):
            print "evaluating batch {}".format(str(i))

            importances, predictions, labels, regions = sess.run([importance, train_prediction, label_batch, metadata_batch], feed_dict={model_state:'test'})

            # TODO: figure out how to grab out the region names also
            
            if batch_end < evaluation_sample_size:
                importances_hf[batch_start:batch_end,:,:,:] = importances
                predictions_hf[batch_start:batch_end,:] = predictions
                labels_hf[batch_start:batch_end,:] = labels
                regions_hf[batch_start:batch_end] = regions
                
            else:
                importances_hf[batch_start:evaluation_sample_size,:,:,:] = importances[0:(evaluation_sample_size - batch_end),:,:,:]
                predictions_hf[batch_start:evaluation_sample_size,:] = predictions[0:(evaluation_sample_size - batch_end),:]
                labels_hf[batch_start:evaluation_sample_size,:] = labels[0:(evaluation_sample_size - batch_end),:]
                regions_hf[batch_start:evaluation_sample_size] = regions[0:(evaluation_sample_size - batch_end)]

            batch_start = batch_end
            batch_end += batch_size
    
    return None
