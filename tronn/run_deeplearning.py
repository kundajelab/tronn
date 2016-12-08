# run Basset

import os
import math

import layers
import models
import metrics
import tensorflow_io

import tensorflow as tf
import numpy as np
import threading
import h5py

from sklearn import metrics as skmetrics
from random import shuffle


def main():

    # ============================================
    # Variables, directories, etc
    # ============================================

    # TODO: add argparse so that this all happens in the command line
    data = 'skin'
    if data == 'roadcode':
        DATA_DIR = '/mnt/lab_data/kundaje/users/dskim89/dnase_modeling/data/roadmap_encode.basset'
        hdf5_file = '{0}/{1}'.format(DATA_DIR, 'encode_roadmap.h5')
        seq_length = 600 # 1000
        tasks = 164 # 13
        num_batches = 18000 #22576
    elif data == 'skin':
        DATA_DIR = '/mnt/lab_data/kundaje/users/dskim89/ggr/chromatin/results/sequence_model.nn.2016-10-21.basset/data'
        DATA_DIR = '/mnt/lab_data/kundaje/users/dskim89/ggr/chromatin/data/nn.atac.idr_regions.2016-11-30.hdf5'
        hdf5_file = '{0}/{1}'.format(DATA_DIR, 'skin_atac_idr_bysplit.h5')
        seq_length = 1000
        tasks = 13
        num_batches = 34544 # TODO change

    LOG_DIR = './log'
    EVAL_DIR = './eval'
    num_epochs = 20
    batch_size = 128
    metric_batch_interval = 50 # How many batches to run before collecting next set of metrics
    evaluation_sample_size = 1000
    model_state = tf.placeholder(tf.string, shape=[])

    restore = False
    training = True
    evaluating = True

    # Start fresh
    if not restore:
        os.system('rm -r {0} {1}'.format(LOG_DIR, EVAL_DIR))
        os.system('mkdir -p {}'.format(EVAL_DIR))
    
    # ============================================
    # Model setup
    # ============================================
    
    # Set up data queues and batches
    queues, inputs, enqueue_ops, feature_batch, label_batch, metadata_batch = tensorflow_io.data_loader(batch_size, seq_length, tasks, model_state) 

    # Set up model (Basset)
    train_op, training_ops, maxnorm_ops, train_prediction, loss_sum, convlayer_relu_1, importance = models.basset(tasks, feature_batch, label_batch, model_state)

    # Set up evaluation
    learning_metrics, metric_updates = metrics.get_metrics(tasks, train_prediction, label_batch)
    learning_metrics['scalar']['loss'] = loss_sum

    
    # ============================================
    # Session setup
    # ============================================
    
    sess = tf.Session()

    # Initialize variables, tensorboard, and saver
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables()) # important to keep for metrics
    merged, train_writer, valid_writer = metrics.make_tensorboard_metrics(sess, learning_metrics, LOG_DIR)
    saver = tf.train.Saver(max_to_keep=num_epochs)

    # Use a separate thread for asynchronous loading of training data
    coord = tf.train.Coordinator()
    t = threading.Thread(target=tensorflow_io.load_and_enqueue, args=(sess, coord, hdf5_file, batch_size, inputs, enqueue_ops, 'NHWC'))
    t.start()

    # If restore
    if restore:
        checkpoint_filename = tf.train.latest_checkpoint('./')
        saver.restore(sess, checkpoint_filename)
    
    # ============================================
    # Training
    # ============================================

    if training:
        for batch_num in range(num_batches*num_epochs):

            # Run a training step (train step + max norm)
            _, _ = sess.run([train_op, training_ops], feed_dict={model_state:'train'})
            _ = sess.run(maxnorm_ops, feed_dict={model_state:'maxnorm'})
            
            # Get training metrics
            if batch_num % metric_batch_interval == 0:
                loss, auroc_mean, auprc_mean = metrics.streaming_evaluate(sess, 'train', train_writer, metric_updates, loss_sum, merged,
                                                                          label_batch, train_prediction, model_state, tasks, batch_num * batch_size,
                                                                          1, '{}/train_eval'.format(EVAL_DIR))
                print '>> TRAINING EX: {0}\tLOSS: {1}\tAUROC: {2}\tAUPRC: {3}'.format(batch_num * batch_size, loss, auroc_mean, auprc_mean)

            # Get validation metrics
            if batch_num % num_batches == 0:
                loss, auroc_mean, auprc_mean = metrics.streaming_evaluate(sess, 'valid', valid_writer, metric_updates, loss_sum, merged,
                                                                          label_batch, train_prediction, model_state, tasks, batch_num * batch_size,
                                                                          10, '{}/valid_eval'.format(EVAL_DIR))
                print '>> VALIDATION LOSS: {0}\tAUROC: {1}\tAUPRC: {2}'.format(loss, auroc_mean, auprc_mean)

            # For now, save a model at each epoch
            if batch_num % num_batches == 0:
                saver.save(sess, './basset_tensorflow', global_step=batch_num)

    # ============================================
    # Evaluation
    # ============================================

    if evaluating:
        #metrics.evaluate('model_out_tensorflow.h5', sess, maxnorm_ops, convlayer_relu_1, model_state, evaluation_sample_size, batch_size)

        # TODO: get gradients at the activation level
        # TODO: i think metadata can just go in here...
        metrics.evaluate_seq_importances('model_importances_tensorflow.h5', sess, importance, train_prediction, label_batch, metadata_batch,
                                         model_state, evaluation_sample_size, batch_size, seq_length, tasks) 
            
    # Clean up and close all threads
    coord.request_stop()
    for queue in queues.keys():
        sess.run(queues[queue].close(cancel_pending_enqueues=True))
    print "STOPPING"
    coord.join([t], stop_grace_period_secs=3)
    print "DONE"

    return None

main()
