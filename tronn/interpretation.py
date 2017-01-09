"""Contains methods and routines for interpreting neural nets
"""

import h5py
import math
import tensorflow as tf


def layerwise_relevance_propagation(loss, features):
    '''
    Layer-wise Relevance Propagation (Batch et al), implemented
    as input * gradient (equivalence is demonstrated in deepLIFT paper,
    Shrikumar et al)
    '''

    [feature_grad] = tf.gradients(loss, [features])
    importances = tf.mul(features, feature_grad, 'input_mul_grad')

    return importances


def run_lrp(checkpoint_path,
            features,
            labels,
            metadata,
            predictions,
            importances,
            batch_size,
            out_file,
            sample_size=1000):
    '''
    Wrapper for running LRP and saving all relevant outputs to an hdf5 file.
    Note that you must set up the graph first to run this.
    '''

    # open a session from checkpoint
    sess = tf.Session()

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    # start queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # from here, run evaluation
    with h5py.File(out_file, 'w') as hf:

        # create datasets
        importances_hf = hf.create_dataset(
            'importances',
            [sample_size] + list(features.get_shape()[1:]))
        predictions_hf = hf.create_dataset(
            'predictions',
            [sample_size] + list(predictions.get_shape()[1:]))
        labels_hf = hf.create_dataset(
            'labels',
            [sample_size] + list(labels.get_shape()[1:]))
        regions_hf = hf.create_dataset(
            'regions',
            [sample_size, 1], dtype='S100')

        # run through the sample size
        batch_start, batch_end = 0, batch_size
        for i in range(int(math.ceil(sample_size / batch_size))):
            print "LRP on batch {}".format(str(i))

            importances_np, predictions_np, labels_np, regions_np = sess.run(
                [importances, predictions, labels, metadata])

            if batch_end < sample_size:
                hf_end = batch_end
                np_end = batch_end
            else:
                hf_end = sample_size
                np_end = sample_size - batch_end

            importances_hf[batch_start:hf_end,:,:,:] = importances_np[0:np_end,:,:,:]
            predictions_hf[batch_start:hf_end,:] = predictions_np[0:np_end,:]
            labels_hf[batch_start:hf_end,:] = labels_np[0:np_end,:]
            regions_hf[batch_start:hf_end,:] = regions_np[0:np_end,:].astype('S100')

            batch_start = batch_end
            batch_end += batch_size

    coord.request_stop()
    coord.join(threads)

    return None


def interpret(data_loader,
              data_file_list,
              model_builder,
              loss_fn,
              checkpoint_path,
              args,
              out_file):
    '''
    Set up a graph and then run importance score extractor
    and save out to file.
    '''

    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader(data_file_list,
                                                 args.batch_size)

        # model
        predictions = model_builder(features, labels, is_training=False)

        # loss
        total_loss = loss_fn(predictions, labels)

        # get the importance scores using input * gradient method
        importances = layerwise_relevance_propagation(total_loss, features)

        # run the model to get the importance scores
        run_lrp(checkpoint_path,
                features,
                labels,
                metadata,
                predictions,
                importances,
                args.batch_size,
                out_file)      

    return None
