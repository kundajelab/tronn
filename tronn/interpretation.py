"""Description: Contains methods and routines for interpreting models
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import h5py
import glob
import gzip
import math
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.stats

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from scipy.signal import fftconvolve
from scipy.signal import convolve2d

import multiprocessing
import Queue

from tronn.models import models
from tronn.models import pwm_convolve
from tronn.visualization import plot_weights
from tronn.datalayer import load_data_from_filename_list

def func_worker(queue):
    """Takes a tuple of (function, args) from queue and runs them

    Args:
      queue: multiprocessing Queue where each elem is (function, args)

    Returns:
      None
    """
    while not queue.empty():
        try:
            [func, args] = queue.get(timeout=0.1)
        except Queue.Empty:
            continue
        func(*args) # run the function with appropriate arguments
    
    return None


def run_in_parallel(queue, parallel=12):
    """Takes a filled queue and runs in parallel
    
    Args:
      queue: multiprocessing Queue where each elem is (function, args)
      parallel: how many to run in parallel

    Returns:
      None
    """
    pids = []
    for i in xrange(parallel):
        pid = os.fork()
        if pid == 0:
            func_worker(queue)
            os._exit(0)
        else:
            pids.append(pid)
            
    for pid in pids:
        os.waitpid(pid,0)
        
    return None


# =======================================================================
# Guided backpropagation - change Relu to guided Relu
# =======================================================================

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad,
                     gen_nn_ops._relu_grad(grad, op.outputs[0]),
                     tf.zeros(grad.get_shape()))


# =======================================================================
# Layerwise relevance propagation (gradient * input)
# =======================================================================

def layerwise_relevance_propagation(loss, features):
    '''
    Layer-wise Relevance Propagation (Batch et al), implemented
    as input * gradient (equivalence is demonstrated in deepLIFT paper,
    Shrikumar et al)
    '''

    [feature_grad] = tf.gradients(loss, [features])
    importances = tf.multiply(features, feature_grad, 'input_mul_grad')

    return importances


# =====================================================================
# Code for calculating importance scores from a model
# =====================================================================

def region_generator(sess,
                     importances,
                     predictions,
                     labels,
                     metadata,
                     stop_idx,
                     num_task):
    '''
    Build a generator to easily extract regions from session run 
    (input data must be ordered)
    '''

    # initialize variables to track progress through generator
    current_chrom = 'NA'
    current_region_start = 0
    current_region_stop = 0
    current_sequences = {}
    for importance_key in importances.keys():
        current_sequences[importance_key] = np.zeros((1, 1))
    current_labels = np.zeros((labels.get_shape()[1],))
    region_idx = 0

    # what's my stop condition? for now just have a stop_idx
    while region_idx < stop_idx:

        # run session to get importance scores etc
        # TODO: eventually convert to parallel
        importances_dict, predictions_np, labels_np, regions_np = sess.run([
            importances,
            predictions,
            labels,
            metadata])
        
        # TODO remove negatives and negative flanks

        # go through the examples in array, yield as you finish a region
        for i in range(regions_np.shape[0]):

            if np.sum(labels_np[i,:]) == 0:
                # ignore this region
                continue
            
            # get the region info
            region = regions_np[i, 0]
            chrom = region.split(':')[0]
            region_start = int(region.split(':')[1].split('-')[0])
            region_stop = int(region.split(':')[1].split('-')[1].split('(')[0])

            # get the sequence importance scores across tasks
            sequence_dict = {}
            for importance_key in importances_dict.keys():
                sequence_dict[importance_key] = np.squeeze(
                    importances_dict[importance_key][i,:,:,:]).transpose(1, 0)
                
            if ((current_chrom == chrom) and
                (region_start < current_region_stop) and
                (region_stop > current_region_stop)):
                
                # add on to current region
                offset = region_start - current_region_start

                # concat zeros to extend sequence array
                for importance_key in importances_dict.keys():
                    current_sequences[importance_key] = np.concatenate(
                        (current_sequences[importance_key],
                         np.zeros((4, region_stop - current_region_stop))),
                        axis=1)

                    # add new data on top
                    current_sequences[importance_key][:,offset:] += sequence_dict[importance_key]
                    
                current_region_stop = region_stop
                current_labels += labels_np[i,:]

            else:
                # we're on a new region
                if current_chrom != 'NA':
                    region_name = '{0}:{1}-{2}'.format(current_chrom,
                                                       current_region_start,
                                                       current_region_stop)
                    if region_idx < stop_idx:
                        # reduce labels to pos v neg
                        current_labels = (current_labels > 0).astype(int)
                        yield current_sequences, region_name, region_idx, current_labels
                        region_idx += 1
                    current_labels = labels_np[i,:]
                    
                # reset current region with the new region info
                current_chrom = chrom
                current_region_start = region_start
                current_region_stop = region_stop
                for importance_key in importances.keys():
                    current_sequences[importance_key] = sequence_dict[importance_key]
                current_labels = labels_np[i,:]

                
def store_importances(padded_sequence, importance_key, idx, out_file):

    with h5py.File(out_file, 'a') as hf:
        hf[importance_key][idx,:,:] = padded_sequence

    return
                

def run_importance_scores(checkpoint_path,
                          features, # NOT USED
                          labels,
                          metadata,
                          predictions,
                          importances,
                          batch_size,
                          out_file,
                          num_task,
                          sample_size=500,
                          width=4096,
                          pos_only=False):
    '''
    Set up the session and then build motif matrix for a specific task
    '''

    # open a session from checkpoint
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # start queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # get model from checkpoint file
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # get importance scores and save out to hdf5 file
    with h5py.File(out_file, 'w') as hf:

        # set up datasets
        importances_datasets = {}
        for importance_key in importances.keys():
            importances_datasets[importance_key] = hf.create_dataset(importance_key, [sample_size, 4, width]) # TODO change this?
        labels_hf = hf.create_dataset('labels', [sample_size, labels.get_shape()[1]])
        regions_hf = hf.create_dataset('regions', [sample_size, 1], dtype='S100')

        # run the region generator
        for sequence, name, idx, labels_np in region_generator(sess,
                                                               importances,
                                                               predictions,
                                                               labels,
                                                               metadata,
                                                               sample_size,
                                                               num_task):

            
            if idx % 100 == 0:
                print idx

            for importance_key in importances.keys():
                # pad sequence so that it all fits
                if sequence[importance_key].shape[1] < width:
                    zero_array = np.zeros((4, width - sequence[importance_key].shape[1]))
                    padded_sequence = np.concatenate((sequence[importance_key], zero_array), axis=1)
                else:
                    trim_len = (sequence[importance_key].shape[1] - width) / 2
                    padded_sequence = sequence[importance_key][:,trim_len:width+trim_len]
                    
                # save into hdf5 files
                importances_datasets[importance_key][idx,:,:] = padded_sequence

            regions_hf[idx,] = name
            labels_hf[idx,] = labels_np
            # TODO save out predictions too

    coord.request_stop()
    coord.join(threads)

    return None


def generate_importance_scores(data_loader,
                               data_file_list,
                               model_builder,
                               loss_fn,
                               checkpoint_path,
                               args,
                               out_file,
                               guided_backprop=True,
                               method='importances',
                               task=0,
                               sample_size=500,
                               pos_only=False):
    '''
    Set up a graph and then run importance score extractor
    and save out to file.
    '''

    batch_size = int(args.batch_size * 3 / 4.0)
    #batch_size = int(args.batch_size / 4.0)
    
    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader(data_file_list,
                                                 batch_size, args.tasks, shuffle=False)
        num_tasks = labels.get_shape()[1]
        print num_tasks
        task_labels = tf.unstack(labels, axis=1)

        # model and predictions
        if guided_backprop:
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                predictions = model_builder(features, labels, args.model, is_training=False)
        else:
            predictions = model_builder(features, labels, args.model, is_training=False)
            
        task_predictions = tf.unstack(predictions, axis=1) # keep this to check whether model doing well on these or not

        # task specific losses (note: total loss is present just for model loading)
        total_loss = loss_fn(predictions, labels)

        task_losses = []
        for task_num in range(num_tasks):
            task_loss = loss_fn(task_predictions[task_num],
                                tf.ones(task_labels[task_num].get_shape()))
            task_losses.append(task_loss)

        # get importance scores
        importances = {}
        for task_num in range(num_tasks):
            #importances['importances_task{}'.format(task_num)] = layerwise_relevance_propagation(task_losses[task_num], features)
            importances['importances_task{}'.format(task_num)] = layerwise_relevance_propagation(task_predictions[task_num], features)

        # run the model to get the importance scores
        run_importance_scores(checkpoint_path,
                              features,
                              labels,
                              metadata,
                              predictions,
                              importances,
                              batch_size,
                              out_file,
                              task,
                              sample_size=sample_size,
                              pos_only=pos_only)

    return None


# =======================================================================
# Create a motif matrix from importance scores
# =======================================================================

def visualize_sample_sequences(h5_file, num_task, out_dir, sample_size=10):
    '''
    Quick check on importance scores. Find a set of positive
    and negative sequences to visualize
    '''

    importances_key = 'importances_task{}'.format(num_task)
    
    with h5py.File(h5_file, 'r') as hf:
        labels = hf['labels'][:,0]

        for label_val in [1, 0]:

            visualized_region_num = 0
            region_idx = 0

            while visualized_region_num < sample_size:

                if hf['labels'][region_idx,0] == label_val:
                    # get sequence and plot it out
                    sequence = np.squeeze(hf[importances_key][region_idx,:,:])
                    name = hf['regions'][region_idx,0]

                    start = int(name.split(':')[1].split('-')[0])
                    stop = int(name.split('-')[1])
                    sequence_len = stop - start
                    sequence = sequence[:,0:sequence_len]

                    print name
                    print sequence.shape
                    out_plot = '{0}/task_{1}.label_{2}.region_{3}.{4}.png'.format(out_dir, num_task, label_val,
                                                                 visualized_region_num, 
                                                                 name.replace(':', '-'))
                    print out_plot
                    plot_weights(sequence, out_plot)

                    visualized_region_num += 1
                    region_idx += 1

                else:
                    region_idx += 1
                
    return None


def visualize_timeseries_sequences(h5_file, num_timepoints, out_dir, sample_size=5):
    '''
    Look at importance scores across time. Assumes timepoints are the first tasks
    in the set
    '''

    with h5py.File(h5_file, 'r') as hf:
        labels = hf['labels'][:]

        visualized_region_num = 0

        while visualized_region_num < sample_size:

            sequence_idx = np.random.randint(hf['regions'].shape[0], size=1)

            if np.sum(labels[sequence_idx,0:num_timepoints]) < 6:
                continue
            else:

                print sequence_idx

                for task_num in range(num_timepoints):

                    importance_key = 'importances_task{}'.format(task_num)

                    # choose random sequence
                    sequence = np.squeeze(hf[importance_key][sequence_idx,:,:])
                    name = hf['regions'][sequence_idx,0]

                    start = int(name.split(':')[1].split('-')[0])
                    stop = int(name.split('-')[1])
                    sequence_len = stop - start
                    sequence = sequence[:,0:sequence_len]

                    print name
                    print sequence.shape
                    out_plot = '{0}/region_{1}.task_{2}.label_{3}.png'.format(out_dir, name.replace(':', '-'), task_num, int(labels[sequence_idx, task_num]))
                    print out_plot
                    # TODO need to set the heights equal across all sequences to plot equal heights
                    plot_weights(sequence, out_plot)

                visualized_region_num += 1
                
    return None


# =======================================================================
# Create a motif matrix from importance scores
# =======================================================================

class PWM(object):
    def __init__(self, weights, name=None, threshold=None):
        self.weights = weights
        self.name = name
        self.threshold = threshold

    @staticmethod
    def from_homer_motif(motif_file):
        with open(motif_file) as fp:
            header = fp.readline().strip().split('\t')
            name = header[1]
            threshold = float(header[2])
            weights = np.loadtxt(fp)

        return PWM(weights, name, threshold)

    @staticmethod
    def get_encode_pwms(motif_file):
        pwms = []

        with open(motif_file) as fp:
            line = fp.readline().strip()
            while True:
                if line == '':
                    break

                header = line.strip('>').strip()
                weights = []
                while True:
                    line = fp.readline()
                    if line == '' or line[0] == '>':
                        break
                    weights.append(map(float, line.split()))
                pwms.append(PWM(np.array(weights).transpose(1,0), header))

        return pwms

    @staticmethod
    def from_cisbp_motif(motif_file):
        name = os.path.basename(motif_file)
        with open(motif_file) as fp:
            _ = fp.readline()
            weights = np.loadtxt(fp)[:, 1:]
        return PWM(weights, name)


def run_pwm_convolution(data_loader,
                        importance_h5,
                        out_h5,
                        batch_size,
                        pwm_file,
                        task_num):
    '''
    Wrapper function where, given an importance matrix, can convert everything
    into a motif matrix
    '''

    importance_key = 'importances_task{}'.format(task_num)
    
    # get basic key stats (to set up output h5 file)
    pwm_list = PWM.get_encode_pwms(pwm_file)
    num_pwms = len(pwm_list)
    with h5py.File(importance_h5, 'r') as hf:
        num_examples = hf[importance_key].shape[0]
        num_tasks = hf['labels'].shape[1]

    # First set up graph and convolutions model
    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader([importance_h5],
                                                 batch_size,
                                                 features_key=importance_key)

        # load the model
        motif_tensor, load_pwm_update = pwm_convolve(features, pwm_list)

        # run the model (set up sessions, etc)
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Run update to load the PWMs
        _ = sess.run(load_pwm_update)

        # set up hdf5 file for saving sequences
        with h5py.File(out_h5, 'w') as out_hf:
            motif_mat = out_hf.create_dataset('motif_scores',
                                              [num_examples, num_pwms])
            labels_mat = out_hf.create_dataset('labels',
                                               [num_examples, num_tasks])
            regions_mat = out_hf.create_dataset('regions',
                                                [num_examples, 1],
                                                dtype='S100')
            motif_names_mat = out_hf.create_dataset('motif_names',
                                                    [num_pwms, 1],
                                                    dtype='S100')

            # save out the motif names
            for i in range(len(pwm_list)):
                motif_names_mat[i] = pwm_list[i].name


            # run through batches worth of sequence
            for batch_idx in range(num_examples / batch_size + 1):

                print batch_idx * batch_size

                batch_motif_mat, batch_regions, batch_labels = sess.run([motif_tensor,
                                                                         metadata,
                                                                         labels])

                batch_start = batch_idx * batch_size
                batch_stop = batch_start + batch_size

                # TODO save out to hdf5 file
                if batch_stop < num_examples:
                    motif_mat[batch_start:batch_stop,:] = batch_motif_mat
                    labels_mat[batch_start:batch_stop,:] = batch_labels
                    regions_mat[batch_start:batch_stop] = batch_regions.astype('S100')
                else:
                    motif_mat[batch_start:num_examples,:] = batch_motif_mat[0:num_examples-batch_start,:]
                    labels_mat[batch_start:num_examples,:] = batch_labels[0:num_examples-batch_start]
                    regions_mat[batch_start:num_examples] = batch_regions[0:num_examples-batch_start].astype('S100')

        coord.request_stop()
        coord.join(threads)

    return None

def run_pwm_convolution_multiple(data_loader,
                        importance_h5,
                        out_h5,
                        batch_size,
                        num_tasks,
                        pwm_file):
    '''
    Wrapper function where, given an importance matrix, can convert everything
    into a motif matrix. Does this across multiple tasks
    '''

    # get basic key stats (to set up output h5 file)
    pwm_list = PWM.get_encode_pwms(pwm_file)
    num_pwms = len(pwm_list)
    with h5py.File(importance_h5, 'r') as hf:
        num_examples = hf['importances_task0'].shape[0]

    # set up hdf5 file for saving sequences
    with h5py.File(out_h5, 'w') as out_hf:
        motif_mat = out_hf.create_dataset('motif_scores',
                                          [num_examples, num_pwms, num_tasks])
        labels_mat = out_hf.create_dataset('labels',
                                           [num_examples, num_tasks])
        regions_mat = out_hf.create_dataset('regions',
                                            [num_examples, 1],
                                            dtype='S100')
        motif_names_mat = out_hf.create_dataset('motif_names',
                                                [num_pwms, 1],
                                                dtype='S100')

        # save out the motif names
        for i in range(len(pwm_list)):
            motif_names_mat[i] = pwm_list[i].name

        # for each task
        for task_num in range(num_tasks):

            # First set up graph and convolutions model
            with tf.Graph().as_default() as g:

                # data loader
                features, labels, metadata = data_loader([importance_h5],
                                                         batch_size,
                                                         'importances_task{}'.format(task_num))

                # load the model
                motif_tensor, load_pwm_update = models.pwm_convolve(features, pwm_list)

                # run the model (set up sessions, etc)
                sess = tf.Session()

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # start queue runners
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # Run update to load the PWMs
                _ = sess.run(load_pwm_update)

                # run through batches worth of sequence
                for batch_idx in range(num_examples / batch_size + 1):

                    print batch_idx * batch_size

                    batch_motif_mat, batch_regions, batch_labels = sess.run([motif_tensor,
                                                                             metadata,
                                                                             labels])

                    batch_start = batch_idx * batch_size
                    batch_stop = batch_start + batch_size

                    # TODO save out to hdf5 file
                    if batch_stop < num_examples:
                        motif_mat[batch_start:batch_stop,:,task_num] = batch_motif_mat
                        labels_mat[batch_start:batch_stop,:] = batch_labels[:,0:num_tasks]
                        regions_mat[batch_start:batch_stop] = batch_regions.astype('S100')
                    else:
                        motif_mat[batch_start:num_examples,:,task_num] = batch_motif_mat[0:num_examples-batch_start,:]
                        labels_mat[batch_start:num_examples,:] = batch_labels[0:num_examples-batch_start,0:num_tasks]
                        regions_mat[batch_start:num_examples] = batch_regions[0:num_examples-batch_start].astype('S100')

                coord.request_stop()
                coord.join(threads)

    return None


def run_motif_distance_extraction(data_loader,
                        importance_h5,
                        out_h5,
                        batch_size,
                        pwm_file,
                        task_num,
                        top_k_val=2):
    '''
    Wrapper function where, given an importance matrix, can convert everything
    into motif scores and motif distances for the top k hits
    Only take positive sequences to build grammars!
    '''

    importance_key = 'importances_task{}'.format(task_num)
    print importance_key
    
    # get basic key stats (to set up output h5 file)
    pwm_list = PWM.get_encode_pwms(pwm_file)
    num_pwms = len(pwm_list)
    with h5py.File(importance_h5, 'r') as hf:
        num_examples = hf[importance_key].shape[0]
        num_tasks = hf['labels'].shape[1]

    # First set up graph and convolutions model
    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader([importance_h5],
                                                 batch_size,
                                                 importance_key)

        # load the model
        motif_scores, motif_distances, load_pwm_update = models.top_motifs_w_distances(features, pwm_list, top_k_val)

        # run the model (set up sessions, etc)
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Run update to load the PWMs
        _ = sess.run(load_pwm_update)

        # set up hdf5 file for saving sequences
        # TODO edit these datasets
        with h5py.File(out_h5, 'w') as out_hf:
            motif_score_mat = out_hf.create_dataset('motif_scores',
                                              [num_examples, num_pwms, num_pwms, top_k_val ** 2])
            motif_dist_mat = out_hf.create_dataset('motif_dists',
                [num_examples, num_pwms, num_pwms, top_k_val ** 2])
            labels_mat = out_hf.create_dataset('labels',
                                               [num_examples, num_tasks])
            regions_mat = out_hf.create_dataset('regions',
                                                [num_examples, 1],
                                                dtype='S100')
            motif_names_mat = out_hf.create_dataset('motif_names',
                                                    [num_pwms, 1],
                                                    dtype='S100')

            # save out the motif names
            for i in range(len(pwm_list)):
                motif_names_mat[i] = pwm_list[i].name

            # run through batches worth of sequence
            for batch_idx in range(num_examples / batch_size + 1):

                print batch_idx * batch_size

                batch_motif_scores, batch_motif_dists, batch_regions, batch_labels = sess.run([motif_scores,
                    motif_distances,
                                                                         metadata,
                                                                         labels])

                batch_start = batch_idx * batch_size
                batch_stop = batch_start + batch_size

                # TODO save out to hdf5 file
                if batch_stop < num_examples:
                    motif_score_mat[batch_start:batch_stop,:,:,:] = batch_motif_scores
                    motif_dist_mat[batch_start:batch_stop,:,:,:] = batch_motif_dists
                    labels_mat[batch_start:batch_stop,:] = batch_labels
                    regions_mat[batch_start:batch_stop] = batch_regions.astype('S100')
                else:
                    motif_score_mat[batch_start:num_examples,:,:,:] = batch_motif_scores[0:num_examples-batch_start,:,:,:]
                    motif_dist_mat[batch_start:num_examples,:,:,:] = batch_motif_dists[0:num_examples-batch_start,:,:,:]
                    labels_mat[batch_start:num_examples,:] = batch_labels[0:num_examples-batch_start]
                    regions_mat[batch_start:num_examples] = batch_regions[0:num_examples-batch_start].astype('S100')

        coord.request_stop()
        coord.join(threads)

    return None


# =======================================================================
# Other useful helper functions
# =======================================================================

def extract_positives_from_motif_mat(h5_file, out_file, task_num):
    '''
    Extract positive set from h5 file to handle in R
    remember to keep track of index positions
    '''

    with h5py.File(h5_file, 'r') as hf:

        # better to pull it all into memory to slice fast
        labels = hf['labels'][:,task_num]
        motif_scores = hf['motif_scores'][:] 
        motif_names = list(hf['motif_names'][:,0])
        regions = list(hf['regions'][:,0])

        pos_array = motif_scores[labels > 0,:]
        pos_regions = list(hf['regions'][labels > 0,0])

        motif_df = pd.DataFrame(data=pos_array[:],
                                index=pos_regions,
                                columns=motif_names)

        # also save out indices
        pos_indices = np.where(labels > 0)
        motif_df['indices'] = pos_indices[0]
        motif_df.to_csv(out_file, sep='\t', compression='gzip')

    return None


def extract_positives_from_motif_topk_mat(h5_file, out_file):
    '''
    Extract positive set from h5 file to handle in R
    remember to keep track of index positions
    '''

    with h5py.File(h5_file, 'r') as hf:

        # better to pull it all into memory to slice fast
        labels = hf['labels'][:,0]
        motif_scores = hf['motif_scores'][:] 
        motif_names = list(hf['motif_names'][:,0])
        regions = list(hf['regions'][:,0])

        pos_array = motif_scores[labels > 0,:]
        pos_regions = list(hf['regions'][labels > 0,0])

        motif_df = pd.DataFrame(data=pos_array[:],
                                index=pos_regions,
                                columns=motif_names)

        # also save out indices
        pos_indices = np.where(labels > 0)
        motif_df['indices'] = pos_indices[0]
        motif_df.to_csv(out_file, sep='\t', compression='gzip')

    return None


def importance_h5_to_txt(h5_file, txt_file):
    '''
    Conversion script for quick viewing in R to plot
    '''

    with h5py.File(h5_file, 'r') as hf:
        motif_mat = hf['motif_scores']
        motif_names_mat = list(hf['motif_names'][:,0])
        regions_mat = list(hf['regions'][:,0])

        motif_df = pd.DataFrame(data=motif_mat[:],
                                index=regions_mat[:],
                                columns=motif_names_mat[:])
        motif_df.to_csv(txt_file, sep='\t', compression='gzip')

    return None

# =======================================================================
# Clustering positives to get subgroups
# =======================================================================





# =======================================================================
# Bootstrap FDR
# =======================================================================

def bootstrap_fdr(motif_mat_h5, out_prefix, task_num, region_set=None,
                  bootstrap_num=9999, fdr=0.005, zscore_cutoff=1.0):
    '''
    Given a motif matrix and labels, calculate a bootstrap FDR
    '''

    # set up numpy array
    with h5py.File(motif_mat_h5, 'r') as hf:

        # better to pull it all into memory to slice fast
        labels = hf['labels'][:,task_num] 
        motif_scores = hf['motif_scores'][:] # TODO only take sequences that are from skin sequences
        motif_names = list(hf['motif_names'][:,0])

        # first calculate the scores for positive set
        if region_set != None:
            # TODO allow passing in an index set which represents your subset of positives
            pos_indices = np.loadtxt(region_set, dtype=int)
            pos_indices_sorted = np.sort(pos_indices)
            pos_array = motif_scores[pos_indices_sorted,:]
            
        else:
            pos_array = motif_scores[labels > 0,:]
        pos_array_z = scipy.stats.mstats.zscore(pos_array, axis=1)
        pos_vector = np.mean(pos_array_z, axis=0) # normalization

        # TODO save out the mean column of positives
        motif_z_avg_df = pd.DataFrame(data=pos_vector, index=motif_names)
        motif_z_avg_df.to_csv('{}.zscores.txt'.format(out_prefix), sep='\t')

        num_pos_examples = pos_array.shape[0]
        num_motifs = pos_array.shape[1]

        # set up results array
        bootstraps = np.zeros((bootstrap_num+1, num_motifs))
        bootstraps[0,:] = pos_vector

        # Now only select bootstraps from regions that are open in skin
        #motif_scores = motif_scores[np.sum(hf['labels'], axis=1) > 0,:]

        for i in range(bootstrap_num):

            if i % 1000 == 0:
                print i

            # randomly select examples 
            bootstrap_indices = np.random.choice(motif_scores.shape[0],
                                                 num_pos_examples,
                                                 replace=False)
            bootstrap_indices.sort()
            bootstrap_array = motif_scores[bootstrap_indices,:]

            bootstrap_array_z = scipy.stats.mstats.zscore(bootstrap_array, axis=1)

            # calculate sum
            bootstrap_sum = np.mean(bootstrap_array_z, axis=0)

            # save into vector
            bootstraps[i+1,:] = bootstrap_sum

    # convert to ranks and save out
    bootstrap_df = pd.DataFrame(data=bootstraps, columns=motif_names)
    bootstrap_ranks_df = bootstrap_df.rank(ascending=False, pct=True)
    pos_fdr = bootstrap_ranks_df.iloc[0,:]
    pos_fdr.to_csv('{}.bootstrap_fdr.txt'.format(out_prefix), sep='\t')

    # also save out a list of those that passed the FDR cutoff
    fdr_cutoff = pos_fdr.ix[pos_fdr < 0.05]
    fdr_cutoff.to_csv('{}.bootstrap_fdr.cutoff.txt'.format(out_prefix), sep='\t')

    # also save out list that pass FDR and also zscore cutoff
    pos_fdr_t = pd.DataFrame(data=pos_fdr, index=pos_fdr.index)
    fdr_w_zscore = motif_z_avg_df.merge(pos_fdr_t, left_index=True, right_index=True)
    fdr_w_zscore.columns = ['zscore', 'FDR']
    fdr_w_zscore_cutoffs = fdr_w_zscore[(fdr_w_zscore['FDR'] < 0.005) & (fdr_w_zscore['zscore'] > zscore_cutoff)]
    fdr_w_zscore_cutoffs_sorted = fdr_w_zscore_cutoffs.sort_values('zscore', ascending=False)
    fdr_w_zscore_cutoffs_sorted.to_csv('{}.fdr_cutoff.zscore_cutoff.txt'.format(out_prefix), sep='\t')
    
    
    return None


def generate_motif_x_motif_mat(motif_mat_h5, out_prefix, region_set=None, score_type='spearman'):
    '''
    With a sequences x motif mat, filter for region set and then get
    correlations of motif scores with other motif scores
    '''

    with h5py.File(motif_mat_h5, 'r') as hf:

        # better to pull it all into memory to slice fast
        labels = hf['labels'][:,0]
        motif_scores = hf['motif_scores'][:] 
        motif_names = list(hf['motif_names'][:,0])

        # select region set if exists, if not just positives
        if region_set != None:
            # TODO allow passing in an index set which represents your subset of positives
            pos_indices = np.loadtxt(region_set, dtype=int)
            pos_indices_sorted = np.sort(pos_indices)
            pos_array = motif_scores[pos_indices_sorted,:]
            
        else:
            pos_array = motif_scores[labels > 0,:]

        pos_array_z = scipy.stats.mstats.zscore(pos_array, axis=1)


        # Now for each motif, calculate the correlation (spearman)
        num_motifs = len(motif_names)
        motif_x_motif_array = np.zeros((num_motifs, num_motifs))

        for i in range(num_motifs):
            if i % 50 == 0:
                print i
            for j in range(num_motifs):
                if score_type == 'spearman':
                    score, pval = scipy.stats.spearmanr(pos_array_z[:,i], pos_array_z[:,j])
                elif score_type == 'mean_score':
                    score = np.mean(pos_array_z[:,i] * pos_array_z[:,j])
                elif score_type == 'mean_x_spearman':
                    rho, pval = scipy.stats.spearmanr(pos_array_z[:,i], pos_array_z[:,j])
                    score = rho * np.mean(pos_array_z[:,i] * pos_array_z[:,j])
                else:
                    score, pval = scipy.stats.spearmanr(pos_array_z[:,i], pos_array_z[:,j])
                motif_x_motif_array[i,j] = score

        motif_x_motif_df = pd.DataFrame(data=motif_x_motif_array, columns=motif_names, index=motif_names)
        motif_x_motif_df.to_csv('{0}.motif_x_motif.{1}.txt'.format(out_prefix, score_type), sep='\t')
    

    return None




def group_motifs_by_sim(motif_list, motif_dist_mat, out_file, cutoff=0.7):
    '''
    Given a motif list and a distance matrix, form
    groups of motifs and put out list
    '''

    # Load the scores into a dictionary
    motif_dist_df = pd.read_table(motif_dist_mat, index_col=0)
    motif_dist_dict = {}
    print motif_dist_df.shape
    motif_names = list(motif_dist_df.index)
    for i in range(motif_dist_df.shape[0]):
        motif_dist_dict[motif_names[i]] = {}
        for j in range(motif_dist_df.shape[1]):
            motif_dist_dict[motif_names[i]][motif_names[j]] = motif_dist_df.iloc[i, j]

    # if first motif, put into motif group dict as seed
    motif_groups = []

    with gzip.open(motif_list, 'r') as fp:
        for line in fp:
            current_motif = line.strip()
            print current_motif
            current_motif_matched = 0

            if len(motif_groups) == 0:
                motif_groups.append([current_motif])
                continue

            for i in range(len(motif_groups)):
                # compare to each motif in group. if at least 1 is above cutoff, join group
                motif_group = list(motif_groups[i])
                for motif in motif_group:
                    similarity = motif_dist_dict[motif][current_motif]
                    if similarity >= cutoff:
                        motif_groups[i].append(current_motif)
                        current_motif_matched = 1

                motif_groups[i] = list(set(motif_groups[i]))

            if current_motif_matched == 0:
                motif_groups.append([current_motif])


    with gzip.open(out_file, 'w') as out:
        for motif_group in motif_groups:
            out.write('#\n')
            for motif in motif_group:
                out.write('{}\n'.format(motif))

    return None


def get_motif_similarities(motif_list, motif_dist_mat, out_file, cutoff=0.5):
    '''
    Given a motif list and a distance matrix, form
    groups of motifs and put out list
    '''

    # Load the scores into a dictionary
    motif_dist_df = pd.read_table(motif_dist_mat, index_col=0)
    motif_dist_dict = {}
    print motif_dist_df.shape
    motif_names = list(motif_dist_df.index)
    for i in range(motif_dist_df.shape[0]):
        motif_dist_dict[motif_names[i]] = {}
        for j in range(motif_dist_df.shape[1]):
            motif_dist_dict[motif_names[i]][motif_names[j]] = motif_dist_df.iloc[i, j]


    # load in motifs
    important_motifs = pd.read_table(motif_list, index_col=0)
    important_motif_list = list(important_motifs.index)

    with open(out_file, 'w') as out:
        with open(motif_list, 'r') as fp:
            for line in fp:

                if 'zscore' in line:
                    continue
                
                current_motif = line.strip().split('\t')[0]
                print current_motif
                for motif in important_motif_list:
                    if motif == current_motif:
                        continue

                    similarity = motif_dist_dict[motif][current_motif]
                    if similarity >= cutoff:
                        out.write('{}\t{}\t{}\n'.format(current_motif, motif, similarity))

    return None


def choose_strongest_motif_from_group(zscore_file, motif_groups_file, out_file):
    '''
    Takes a motif groups file and zscores and chooses strongest one to output
    '''

    # read in zscore file to dictionary
    zscore_dict = {}
    with open(zscore_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')

            if fields[0] == '0':
                continue

            zscore_dict[fields[0]] = float(fields[1])

    # for each motif group, select strongest
    with gzip.open(motif_groups_file, 'r') as fp:
        with gzip.open(out_file, 'w') as out:
            motif = ''
            zscore = 0


            for line in fp:

                if line.startswith('#'):
                    if motif != '':
                        out.write('{0}\t{1}\n'.format(motif, zscore))

                    motif = ''
                    zscore = 0
                    continue

                current_motif = line.strip()
                current_zscore = zscore_dict[current_motif]

                if current_zscore > zscore:
                    motif = current_motif
                    zscore = current_zscore

    return None

def add_zscore(zscore_file, motif_file, out_file):
    '''
    Quick function to put zscore with motif
    '''

    # read in zscore file to dictionary
    zscore_dict = {}
    with open(zscore_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')

            if fields[0] == '0':
                continue

            zscore_dict[fields[0]] = float(fields[1])

    # for each motif add zscore
    with open(motif_file, 'r') as fp:
        with open(out_file, 'w') as out:
            for line in fp:

                motif = line.strip()
                zscore = zscore_dict[motif]
                out.write('{0}\t{1}\n'.format(motif, zscore))

    return None


def reduce_motif_redundancy_by_dist_overlap(motif_dists_mat_h5, motif_offsets_mat_file, motif_list_file):
    '''
    remove motifs if they overlap (ie, their average distance is 0)
    '''

    # read in motif list
    motif_list = []    
    with gzip.open(motif_list_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')
            motif_list.append((fields[0], float(fields[1])))

    final_motif_list = []
    with h5py.File(motif_dists_mat_h5, 'r') as hf:


        # make a motif to index dict
        motif_names = list(hf['motif_names'][:,0])
        name_to_index = {}
        for i in range(len(motif_names)):
            name_to_index[motif_names[i]] = i


        for i in range(len(motif_list)):
            is_best_single_motif = 1
            motif_i = motif_list[i][0]
            motif_i_idx = name_to_index[motif_i]

            for j in range(len(motif_list)):
                motif_j = motif_list[j][0]
                motif_j_idx = name_to_index[motif_j]

                dists = hf['motif_dists'][:,motif_i_idx, motif_j_idx,:]
                dists_flat = dists.flatten()

                dists_mean = np.mean(dists_flat)

                print motif_i, motif_j, dists_mean

            # compare to all others. if no matches stronger than it, put into final list

            # if there is a match, but the other one is higher zscore, do not add




        # for each motif compared to each other motif,
        # check to see their average distance


    return None


def make_score_dist_plot(motif_a, motif_b, motif_dists_mat_h5, out_prefix):
    '''
    Helper function to make plot
    '''

    with h5py.File(motif_dists_mat_h5, 'r') as hf:

        # make a motif to index dict
        motif_names = list(hf['motif_names'][:,0])
        name_to_index = {}
        for i in range(len(motif_names)):
            name_to_index[motif_names[i]] = i

        motif_a_idx = name_to_index[motif_a]
        motif_b_idx = name_to_index[motif_b]

        scores = hf['motif_scores'][:,motif_a_idx,motif_b_idx,:]
        dists = hf['motif_dists'][:,motif_a_idx,motif_b_idx,:]

        # flatten
        scores_flat = scores.flatten()
        dists_flat = dists.flatten()

        # TODO adjust the dists
        
        # make a pandas df and save out to text
        out_table = '{}.scores_w_dists.txt.gz'.format(out_prefix)
        dists_w_scores = np.stack([dists_flat, scores_flat], axis=1)
        dists_w_scores_df = pd.DataFrame(data=dists_w_scores)
        dists_w_scores_df.to_csv(out_table, sep='\t', compression='gzip', header=False, index=False)

    # then plot in R
    plot_script = '/users/dskim89/git/tronn/scripts/make_score_dist_plot.R'
    os.system('Rscript {0} {1} {2}'.format(plot_script, out_table, out_prefix))


    return None


def plot_sig_pairs(motif_pair_file, motif_dists_mat_h5, cutoff=3):
    '''
    Go through sig file and plot sig pairs
    '''

    seen_pairs = []

    with open(motif_pair_file, 'r') as fp:
        for line in fp:

            [motif_a, motif_b, zscore] = line.strip().split('\t')

            if float(zscore) >= cutoff:
                motif_a_hgnc = motif_a.split('_')[0]
                motif_b_hgnc = motif_b.split('_')[0]

                pair = '{0}-{1}'.format(motif_a_hgnc, motif_b_hgnc)

                if pair not in seen_pairs:
                    out_prefix = '{0}.{1}-{2}'.format(motif_pair_file.split('.txt')[0], motif_a_hgnc, motif_b_hgnc)
                    make_score_dist_plot(motif_a, motif_b, motif_dists_mat_h5, out_prefix)

                    seen_pairs.append(pair)
                    seen_pairs.append('{0}-{1}'.format(motif_b_hgnc, motif_a_hgnc))

    return None


def get_significant_motif_pairs(motif_list, motif_x_motif_mat_file, out_file, manual=False, std_cutoff=3):
    '''
    With a motif list, compare all to all and check significance
    '''

    # first load in the motif x motif matrix
    motif_x_motif_df = pd.read_table(motif_x_motif_mat_file, index_col=0)
    motif_names = list(motif_x_motif_df.index)

    # get index dictionary
    motif_to_idx = {}
    for i in range(len(motif_names)):
        motif_to_idx[motif_names[i]] = i

    # calculate mean and std across all values in matrix
    mean = motif_x_motif_df.values.mean()
    std = motif_x_motif_df.values.std()

    print mean
    print std

    # for each motif, compare to each other one. only keep if above 2 std
    if manual:
        important_motifs = pd.read_table(motif_list, header=None)
        important_motif_list = list(important_motifs[0])
    else:
        important_motifs = pd.read_table(motif_list, index_col=0)
        important_motif_list = list(important_motifs.index)

    print important_motif_list

    already_seen = []
    
    with open(out_file, 'w') as out:

        for i in range(len(important_motif_list)):

            mean = motif_x_motif_df.values.mean(axis=0)[i]
            std = motif_x_motif_df.values.std(axis=0)[i]

            print mean, std


            for j in range(len(important_motif_list)):

                name_1 = important_motif_list[i]
                name_2 = important_motif_list[j]

                if name_1 == name_2:
                    continue

                idx_1 = motif_to_idx[name_1]
                idx_2 = motif_to_idx[name_2]

                score = motif_x_motif_df.iloc[idx_1, idx_2]

                if score >= (mean + std_cutoff * std):
                    print name_1, name_2, score
                    out_string = '{0}\t{1}\t{2}\n'.format(name_1, name_2, score)
                    if out_string in already_seen:
                        continue
                    else:
                        out.write(out_string)
                        already_seen.append('{1}\t{0}\t{2}\n'.format(name_1, name_2, score))

    return None



def interpret(
        args,
        data_loader,
        data_files,
        model,
        loss_fn,
        prefix,
        out_dir, 
        task_nums, # manual
        dendro_cutoffs, # manual
        motif_file,
        motif_sim_file,
        motif_offsets_file,
        rna_file,
        rna_conversion_file,
        checkpoint_path,
        scratch_dir='./',
        sample_size=220000):
    """placeholder for now"""

    importances_mat_h5 = '{0}/{1}.importances.h5'.format(scratch_dir, prefix)

    # ---------------------------------------------------
    # generate importance scores across all open sites
    # ---------------------------------------------------

    if not os.path.isfile(importances_mat_h5):
        generate_importance_scores(
            data_loader,
            data_files,
            model,
            loss_fn,
            checkpoint_path,
            args,
            importances_mat_h5,
            guided_backprop=True, 
            method='importances',
            sample_size=sample_size) # TODO change this, it's a larger set than this

    # ---------------------------------------------------
    # for each task, do the following:
    # ---------------------------------------------------

    for task_num_idx in range(len(task_nums)):

        task_num = task_nums[task_num_idx]
        print "Working on task {}".format(task_num)

        if args.plot_importances:
            # visualize a few samples
            sample_seq_dir = 'task_{}.sample_seqs'.format(task_num)
            os.system('mkdir -p {}'.format(sample_seq_dir))
            visualize_sample_sequences(importances_mat_h5, task_num, sample_seq_dir)
            
        # ---------------------------------------------------
        # Run all task-specific importance scores through PWM convolutions
        # IN: sequences x importance scores
        # OUT: sequences x motifs
        # ---------------------------------------------------
        motif_mat_h5 = 'task_{}.motif_mat.h5'.format(task_num)
        if not os.path.isfile(motif_mat_h5):
            run_pwm_convolution(
                data_loader,
                importances_mat_h5,
                motif_mat_h5,
                args.batch_size * 2,
                motif_file,
                task_num)

        # ---------------------------------------------------
        # extract the positives to cluster in R and visualize
        # IN: sequences x motifs
        # OUT: positive sequences x motifs
        # ---------------------------------------------------
        pos_motif_mat = 'task_{}.motif_mat.positives.txt.gz'.format(task_num)
        if not os.path.isfile(pos_motif_mat):
            extract_positives_from_motif_mat(motif_mat_h5, pos_motif_mat, task_num)

        # ---------------------------------------------------
        # Cluster positives in R and output subgroups
        # IN: positive sequences x motifs
        # OUT: subgroups of sequences
        # ---------------------------------------------------
        cluster_dir = 'task_{}.positives.clustered'.format(task_num)
        if not os.path.isdir(cluster_dir):
            os.system('mkdir -p {}'.format(cluster_dir))
            prefix = 'task_{}'.format(task_num)
            os.system('run_region_clustering.R {0} 50 {1} {2}/{3}'.format(pos_motif_mat,
                                                                          dendro_cutoffs[task_num_idx],
                                                                          cluster_dir,
                                                                          prefix))        


        # ---------------------------------------------------
        # Now for each subgroup of sequences, get a grammar back
        # ---------------------------------------------------
        for subgroup_idx in range(dendro_cutoffs[task_num_idx]):
            
            index_group = "{0}/task_{1}.group_{2}.indices.txt.gz".format(cluster_dir, task_num, subgroup_idx+1)
            out_prefix = '{}'.format(index_group.split('.indices')[0])
            
            
            # ---------------------------------------------------
            # Run boostrap FDR to get back significant motifs
            # IN: subgroup x motifs
            # OUT: sig motifs
            # ---------------------------------------------------
            bootstrap_fdr_cutoff_file = '{}.fdr_cutoff.zscore_cutoff.txt'.format(out_prefix)
            if not os.path.isfile(bootstrap_fdr_cutoff_file):
                bootstrap_fdr(motif_mat_h5, out_prefix, task_num, index_group)

            # Filter with RNA evidence
            rna_filtered_file = '{}.rna_filtered.txt'.format(bootstrap_fdr_cutoff_file.split('.txt')[0])
            if not os.path.isfile(rna_filtered_file):
                add_rna_evidence = ("filter_w_rna.R "
                                    "{0} "
                                    "{1} "
                                    "{2} "
                                    "{3}").format(bootstrap_fdr_cutoff_file, rna_conversion_file, rna_file, rna_filtered_file)
                print add_rna_evidence
                os.system(add_rna_evidence)

            # make a network grammar
            # size of node is motif strength, links are motif similarity
            grammar_network_plot = '{}.grammar.network.pdf'.format(rna_filtered_file.split('.txt')[0])
            
            #if not os.path.isfile(grammar_network_plot):
            make_network_plot = 'make_network_grammar_v2.R {0} {1} {2}'.format(rna_filtered_file, motif_sim_file, grammar_network_plot)
            print make_network_plot
            os.system(make_network_plot)


            
                
    return None


def run(args):

    # find data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    print 'Found {} chrom files'.format(len(data_files))

    # checkpoint file
    checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.model_dir))
    print checkpoint_path

    # set up scratch_dir
    os.system('mkdir -p {}'.format(args.scratch_dir))

    # load external data files
    with open(args.annotations, 'r') as fp:
        annotation_files = json.load(fp)

    # current manual choices
    task_nums = [0, 9, 10, 14]
    dendro_cutoffs = [7, 6, 7, 7]
    
    interpret(args,
              load_data_from_filename_list,
              data_files,
              models[args.model['name']],
              tf.losses.sigmoid_cross_entropy,
              args.prefix,
              args.out_dir,
              task_nums, 
              dendro_cutoffs, 
              annotation_files["motif_file"],
              annotation_files["motif_sim_file"],
              annotation_files["motif_offsets_file"],
              annotation_files["rna_file"],
              annotation_files["rna_conversion_file"],
              checkpoint_path,
              scratch_dir=args.scratch_dir,
              sample_size=args.sample_size)

    return
