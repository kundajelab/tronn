"""Description: contains code to generate per-base-pair importance scores
"""

import h5py
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from tronn.util.tf_utils import setup_tensorflow_session, close_tensorflow_session
from tronn.models import stdev_cutoff
from tronn.visualization import plot_weights


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    """Replaces ReLUs with guided ReLUs in a tensorflow graph. Use to 
    allow guided backpropagation in interpretation mode. Generally only 
    turn on for a trained model.
    
    Args:
      op: the op to replace the gradient
      grad: the gradient value
    
    Returns:
      tensorflow operation that removes negative gradients
    """
    return tf.where(0. < grad,
                    gen_nn_ops._relu_grad(grad, op.outputs[0]),
                    tf.zeros(grad.get_shape()))


def layerwise_relevance_propagation(tensor, features):
    """Layer-wise Relevance Propagation (Batch et al), implemented
    as input * gradient (equivalence is demonstrated in deepLIFT paper,
    Shrikumar et al). Generally center the tensor on the logits.
    
    Args:
      tensor: the tensor from which to propagate gradients backwards
      features: the input tensor on which you want the importance scores
    
    Returns:
      Input tensor weighted by gradient backpropagation.
    """
    [feature_grad] = tf.gradients(tensor, [features])
    importances = tf.multiply(features, feature_grad, 'input_mul_grad')
    
    return importances


def region_generator(sess,
                     importances,
                     predictions,
                     labels,
                     metadata,
                     stop_idx,
                     num_task): # TODO check if this arg is needed
    """Build a generator to easily extract regions from session run 
    (input data must be ordered)
    
    Args:
      sess: tensorflow session with graph/model
      importances: a dictionary of tensors that correspond to importances
      predictions: predictions tensor
      labels: labels tensor
      metadata: metadata tensor
      stop_idx: how many regions to generate
      num_task: whic task to focus on

    Returns:
      current_sequences: dictionary of importance scores {task: importances}
      region_name: name of region {chr}:{start}-{stop}
      region_idx: index of region for output file
      current_labels: labels for the region
    """
    
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
        importances_dict, predictions_np, labels_np, regions_np = sess.run([
            importances,
            predictions,
            labels,
            metadata])
        
        # TODO(dk) remove negatives and negative flanks

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


def extract_importances(
        tronn_graph,
        model_dir,
        out_file,
        sample_size,
        method="guided_backprop",
        width=4096,
        pos_only=False):
    """Set up a graph and then run importance score extractor
    and save out to file.

    Args:
      tronn_graph: a TronnNeuralNetGraph instance
      model_dir: directory with trained model
      out_file: hdf5 file to store importances
      method: importance method to use
      sample_size: number of regions to run
      pos_only: only keep positive sequences

    Returns:
      None
    """
    with tf.Graph().as_default() as g:

        # build graph
        if method == "guided_backprop":
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                importances = tronn_graph.build_inference_graph()
        elif method == "simple_gradients":
            importances = tronn_graph.build_inference_graph()

        # set up session
        sess, coord, threads = setup_tensorflow_session()

        # restore
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        # set up hdf5 file to store outputs
        with h5py.File(out_file, 'w') as hf:
            
            # set up datasets
            importances_datasets = {}
            for importance_key in importances.keys():
                importances_datasets[importance_key] = hf.create_dataset(importance_key, [sample_size, 4, width])
            labels_hf = hf.create_dataset('labels', [sample_size, tronn_graph.labels.get_shape()[1]])
            regions_hf = hf.create_dataset('regions', [sample_size, 1], dtype='S100')

            # run the region generator
            for sequence, name, idx, labels_np in region_generator(
                    sess, importances, predictions, labels, metadata, sample_size, num_task):
            
                if idx % 1000 == 0:
                    print idx

                # For importances, pad/trim to make fixed length and store in hdf5
                for importance_key in importances.keys():
                    if sequence[importance_key].shape[1] < width:
                        zero_array = np.zeros((4, width - sequence[importance_key].shape[1]))
                        padded_sequence = np.concatenate((sequence[importance_key], zero_array), axis=1)
                    else:
                        trim_len = (sequence[importance_key].shape[1] - width) / 2
                        padded_sequence = sequence[importance_key][:,trim_len:width+trim_len]
                    importances_datasets[importance_key][idx,:,:] = padded_sequence

                # For other regions save also
                regions_hf[idx,] = name
                labels_hf[idx,] = labels_np
                # TODO(dk) save out predictions too

        close_tensorflow_session(coord, threads)

    return None


def call_importance_peaks(
        data_loader,
        importance_h5,
        out_h5,
        batch_size,
        task_num,
        pval):
    """Calls peaks on importance scores
    
    Currently assumes a poisson distribution of scores. Calculates
    poisson lambda and uses it to get a pval threshold.

    """
    print "calling peaks with pval {}".format(pval)
    importance_key = 'importances_task{}'.format(task_num)
    
    with h5py.File(importance_h5, 'r') as hf:
        num_examples = hf[importance_key].shape[0]
        seq_length = hf[importance_key].shape[2]
        num_tasks = hf['labels'].shape[1]

    # First set up graph and convolutions model
    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader([importance_h5],
                                                 batch_size,
                                                 features_key=importance_key)

        # load the model
        thresholded_tensor = stdev_cutoff(features)

        # run the model (set up sessions, etc)
        sess, coord, threads = setup_tensorflow_session()

        # set up hdf5 file for saving sequences
        with h5py.File(out_h5, 'w') as out_hf:
            importance_mat = out_hf.create_dataset(importance_key,
                                              [num_examples, 4, seq_length])
            labels_mat = out_hf.create_dataset('labels',
                                               [num_examples, num_tasks])
            regions_mat = out_hf.create_dataset('regions',
                                                [num_examples, 1],
                                                dtype='S100')

            # run through batches worth of sequence
            for batch_idx in range(num_examples / batch_size + 1):

                print batch_idx * batch_size

                batch_importances, batch_regions, batch_labels = sess.run([thresholded_tensor,
                                                                           metadata,
                                                                           labels])

                batch_start = batch_idx * batch_size
                batch_stop = batch_start + batch_size

                # TODO save out to hdf5 file
                if batch_stop < num_examples:
                    importance_mat[batch_start:batch_stop,:] = batch_importances
                    labels_mat[batch_start:batch_stop,:] = batch_labels
                    regions_mat[batch_start:batch_stop] = batch_regions.astype('S100')
                else:
                    importance_mat[batch_start:num_examples,:] = batch_importances[0:num_examples-batch_start,:]
                    labels_mat[batch_start:num_examples,:] = batch_labels[0:num_examples-batch_start]
                    regions_mat[batch_start:num_examples] = batch_regions[0:num_examples-batch_start].astype('S100')

        close_tensorflow_session(coord, threads)

    return None




def visualize_sample_sequences(h5_file, num_task, out_dir, sample_size=10):
    """Quick check on importance scores. Find a set of positive
    and negative sequences to visualize

    Args:
      h5_file: hdf5 file of importance scores
      num_task: which task to focus on
      out_dir: where to store these sample sequences
      sample_size: number of regions to visualize

    Returns:
      Plots of visualized sequences
    """
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
