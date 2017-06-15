"""Description: contains code to generate per-base-pair importance scores
"""


import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


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
    """Set up the session and then build motif matrix for a specific task

    Args:
	  checkpoint_path: where the model is stored
	  features: feature tensor
	  labels: label tensor
	  metadata: metadata tensor
	  predictions: predictions tensor
	  importances: importances tensor
	  batch_size: batch size
	  out_file: hdf5 file to store importances
	  num_task: which task to focus on
	  sample_size: number of samples to run
	  width: how wide to keep importances
	  pos_only: only keep regions that have at least 1 positive

	Returns:
	  hdf5 file with importance datasets for each task
    """
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
            importances_datasets[importance_key] = hf.create_dataset(importance_key, [sample_size, 4, width])
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
    """Set up a graph and then run importance score extractor
    and save out to file.

    Args:
      data_loader: datalayer
      data_file_list: list of hdf5 files to run
      model_builder: model
      loss_fn: loss function
      checkpoint_path: where model is
      args: args
      out_file: hdf5 file to store importances
      guided_backprop: importance method to use
      method: importance method to use
      task: which task to run
      sample_size: number of regions to run
      pos_only: only keep positive sequences

    Returns:
      hdf5 file of importances
    """
    batch_size = int(args.batch_size / 2.0)
    
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
                predictions = model_builder(features,
                                            labels,
                                            args.model,
                                            is_training=False)
        else:
            predictions = model_builder(features,
                                        labels,
                                        args.model,
                                        is_training=False)

        # keep this for now to know which examples model does poorly on
        task_predictions = tf.unstack(predictions, axis=1)

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
            importances['importances_task{}'.format(task_num)] = layerwise_relevance_propagation(task_predictions[task_num],
                                                                                                 features)

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
