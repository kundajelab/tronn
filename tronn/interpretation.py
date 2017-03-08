"""Contains methods and routines for interpreting neural nets
"""
import config

import h5py
import gzip
import math
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def layerwise_relevance_propagation(loss, features):
    '''
    Layer-wise Relevance Propagation (Batch et al), implemented
    as input * gradient (equivalence is demonstrated in deepLIFT paper,
    Shrikumar et al)
    '''

    [feature_grad] = tf.gradients(loss, [features])
    importances = tf.multiply(features, feature_grad, 'input_mul_grad')

    return importances


def run_lrp(checkpoint_path,
            features,
            labels,
            metadata,
            logits,
            importances,
            batch_size,
            out_file,
            sample_size=1000,
            ignore_num=142000):
    '''
    Wrapper for running LRP and saving all relevant outputs to an hdf5 file.
    Note that you must set up the graph first to run this.
    '''

    # open a session from checkpoint
    sess = tf.Session(session_config=config.session_config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # start queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # from here, run evaluation
    with h5py.File(out_file, 'w') as hf:

        # create datasets
        # TODO make a different importance dataset per importance requested
        names_to_hf = {}
        for dataset_name in importances.keys():
            names_to_hf[dataset_name] = hf.create_dataset(dataset_name, [sample_size] + list(features.get_shape()[1:]))

        logits_hf = hf.create_dataset(
            'logits',
            [sample_size] + list(logits.get_shape()[1:]))
        labels_hf = hf.create_dataset(
            'labels',
            [sample_size] + list(labels.get_shape()[1:]))
        regions_hf = hf.create_dataset(
            'regions',
            [sample_size, 1], dtype='S100')

        # Ignore num: current hacky way to burn samples to get to genommic region
        # of interest
        for i in xrange(int(math.ceil(ignore_num / batch_size))):
            _ = sess.run([labels])

        # run through the sample size
        batch_start, batch_end = 0, batch_size
        for i in xrange(int(math.ceil(sample_size / float(batch_size)))):
            print "LRP on batch {}".format(str(i))

            importances_dict, logits_np, labels_np, regions_np = sess.run(
                [importances, logits, labels, metadata])

            if batch_end < sample_size:
                hf_end = batch_end
                np_end = batch_end
            else:
                hf_end = sample_size
                np_end = sample_size - batch_start

            for dataset_name in importances.keys():
                names_to_hf[dataset_name][batch_start:hf_end,:,:,:] = importances_dict[dataset_name][0:np_end,:,:,:]
            logits_hf[batch_start:hf_end,:] = logits_np[0:np_end,:]
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
              out_file,
              sample_size=1000):
    '''
    Set up a graph and then run importance score extractor
    and save out to file.
    '''

    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader(data_file_list, args.batch_size, args.days)
        num_tasks = labels.get_shape()[1]
        task_labels = tf.unstack(labels, axis=1)

        # model
        logits = model_builder(features, labels, args.model, is_training=False)
        task_logits = tf.unstack(logits, axis=1)

        # loss, global and task-specific
        total_loss = loss_fn(logits, labels)
        task_losses = []
        for task_num in range(num_tasks):
            task_losses.append(loss_fn(task_logits[task_num], task_labels[task_num]))

        # get importance scores
        importances = {}
        importances['importance_global'] = layerwise_relevance_propagation(total_loss, features)
        for task_num in range(num_tasks):
            importances['importance_{}'.format(task_num)] = layerwise_relevance_propagation(task_losses[task_num], features)

        # run the model to get the importance scores
        run_lrp(checkpoint_path,
                features,
                labels,
                metadata,
                logits,
                importances,
                args.batch_size,
                out_file,
                sample_size)      

    return None


def importance_to_plot(hdf5_file, out_plot, dataset_name='importance_global'):
    '''
    For the regions selected, make an average signal plot
    '''

    with h5py.File(hdf5_file, 'r') as hf:
        importances = hf.get(dataset_name)
        importances_redux = np.squeeze(importances)
        importances_max = np.amax(importances_redux, axis=2) 

        # regions: get start and finish and make a numpy array
        regions = hf.get('regions')
        region_start = int(regions[0][0].split(':')[1].split('-')[0])
        print region_start
        region_end = int(regions[-1][0].split(':')[1].split('-')[1].split('(')[0])
        print region_end

        aggregate_info = np.zeros((region_end - region_start,))
        normalizer = 1e-8 * np.ones((region_end - region_start,))

        # get indices of where labels were positive
        labels = hf.get('labels')
        pos_indices = np.where(np.sum(labels, axis=1))

        # then filter importances through the labels
        importances_poslabel = np.squeeze(importances_max[pos_indices,:])

        for i in xrange(importances_poslabel.shape[0]):

            # global index
            example_idx = pos_indices[0][i]

            # Get the region name
            region_name = regions[example_idx][0]
            local_region_start = int(region_name.split(':')[1].split('-')[0]) - region_start
            local_region_end = int(region_name.split(':')[1].split('-')[1].split('(')[0]) - region_start
            #print local_region_start, local_region_end

            aggregate_info[local_region_start:local_region_end] += importances_poslabel[i,:]
            normalizer[local_region_start:local_region_end] += 1

            #print 'aggregate: {}'.format(str(np.sum(aggregate_info)))

            #print 'norm: {}'.format(str(np.sum(normalizer)))

        # normalize things
        normalized_signal = np.divide(aggregate_info, normalizer)
        genome_coord = np.arange(region_start, region_end)
        #print np.sum(normalized_signal)
        #print normalized_signal[0:1000]
        #print normalized_signal[1000:2000]
        #print np.where(normalized_signal > 0)
        #print len(np.where(normalized_signal > 0)[0])

        plt.clf()
        plt.plot(genome_coord, normalized_signal)
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        plt.savefig('topview_{}.pdf'.format(dataset_name))

        # Zoom in
        plt.clf()
        zoom_start = 53250000 - genome_coord[0]
        zoom_end = 53300000 - genome_coord[0]
        plt.plot(genome_coord[zoom_start:zoom_end], normalized_signal[zoom_start:zoom_end])
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        plt.savefig('zoomed_{}.pdf'.format(dataset_name))

        # Zoom some more
        plt.clf()
        zoom_start = 53250000 - genome_coord[0]
        zoom_end = 53255000 - genome_coord[0]
        plt.plot(genome_coord[zoom_start:zoom_end], normalized_signal[zoom_start:zoom_end])
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        plt.savefig('zoomzoom_{}.pdf'.format(dataset_name))

        # Zoom some more some more
        plt.clf()
        zoom_start = 53251400 - genome_coord[0]
        zoom_end = 53251600 - genome_coord[0]
        plt.plot(genome_coord[zoom_start:zoom_end], normalized_signal[zoom_start:zoom_end])
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        plt.savefig('zoomzoomzoom_{}.pdf'.format(dataset_name))


    return None


def importance_to_bed(hdf5_file, out_bed, dataset_name='importance_global', smoothing=4):
    '''
    Given an hdf5 file, this function goes into the importance scores and
    converts to a bed format of high importance regions. Currently very hacky
    '''

    with h5py.File(hdf5_file, 'r') as hf:
        importances = hf.get(dataset_name)
        importances_redux = np.squeeze(importances)
        importances_max = np.amax(importances_redux, axis=2) 
        
        regions = hf.get('regions')

        # get indices of where labels were positive
        labels = hf.get('labels')
        pos_indices = np.where(np.sum(labels, axis=1))

        # then filter importances through the labels
        importances_poslabel = np.squeeze(importances_max[pos_indices,:])

        # now extract bed style regions
        #OUT = gzip.open(out_bed, 'w')
        OUT = open(out_bed, 'w')
        for i in xrange(importances_poslabel.shape[0]):

            if i % 1000 == 0:
                print i

            example_idx = pos_indices[0][i]

            # Smooth importance scores
            smoothing_position_total = len(importances_poslabel[i,:]) - smoothing
            smoothed_importance = np.zeros(smoothing_position_total)
            for pos in xrange(smoothing_position_total):
                smoothed_importance[pos] = np.sum(importances_poslabel[i, pos:(pos+smoothing)]) / float(smoothing)

            # Set a threshold as global mean of sequence and set everything below to zero
            seq_mean = np.mean(smoothed_importance)
            smoothed_importance[ smoothed_importance < seq_mean ] = 0

            # Then from each starting position, look forward {lookforward} bases, +/- 2. If there is still info content, push out until there is no info content.
            pos = 0
            total_regions = 0
            while True:
                # Break if you've come to end of sequence
                if pos >= smoothing_position_total:
                    break
                
                # Calculate total info
                current_total_info = np.sum(smoothed_importance[pos:(pos+smoothing)])
                
                # First check if there is any info content
                if current_total_info <= 0:
                    pos += 1
                    continue

                # If so, start checking further out to get a window of importance
                # heuristic - this region should also contain at least 4 informative sites?
                start = pos
                stop = pos+smoothing
                while True:
                    new_total_info = np.sum(smoothed_importance[start:(stop+1)])
                    if new_total_info == current_total_info:
                        break
                    else:
                        current_total_info = new_total_info
                        stop += 1
                    
                # With start and stop, save out this piece to BED file
                region_name = regions[example_idx][0]
                chrom = region_name.split(':')[0]
                region_start = int(region_name.split(':')[1].split('-')[0])
                region_stop = int(region_name.split('-')[-1].split('(')[0])

                impt_start = region_start + start
                impt_stop = region_start + stop

                OUT.write('{0}\t{1}\t{2}\tregion{3}\t{3}\t+\n'.format(chrom, impt_start, impt_stop, total_regions))
                total_regions += 1
                
                if total_regions % 1000 == 0:
                    print total_regions
                
                pos = stop
                
        OUT.close()



    return None
