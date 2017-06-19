"""Description: Contains functions/tools for analyzing weighted kmers
extracted from importance weighted sequences.
"""

import os
import glob
import h5py
import json

import tensorflow as tf

from tronn.datalayer import load_data_from_filename_list
from tronn.interpretation.importances import generate_importance_scores
from tronn.interpretation.importances import visualize_sample_sequences
from tronn.models import stdev_cutoff
from tronn.models import models

# TODO move region generator to a utils group?
# TODO move bootstrap FDR to utils group?


def call_importance_peaks(data_loader,
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
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # set up hdf5 file for saving sequences
        with h5py.File(out_h5, 'w') as out_hf:
            importance_mat = out_hf.create_dataset(importance_key,
                                              [num_examples, 4, seq_length])
            labels_mat = out_hf.create_dataset('labels',
                                               labels.get_shape())
            regions_mat = out_hf.create_dataset('regions',
                                                metadata.get_shape(),
                                                dtype='S100')

            # TODO remove this
            num_examples = 12800
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

        coord.request_stop()
        coord.join(threads)

    return None


def kmerize():
    """Convert weighted sequence into weighted kmers
    """
    
    return None


def cluster_kmers():
    """Given a distance matrix of kmers, performs Louvain clustering to get
    communities of kmers that can then be merged to make motifs
    """

    # consider jaccard distance?    

    return None


def make_one_motif():
    """Given a group of kmers, forms a PWM motif
    """
    
    return None


def make_motifs():
    """Given clusters of kmers, make motifs
    """
    
    return None


def interpret_wkm(
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
        sample_size=220000,
        pval=0.05):
    """placeholder for now"""

    importances_mat_h5 = '{0}/{1}.importances.h5'.format(scratch_dir, prefix)
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

		# ---------------------------------------------------
        # Generate thresholded importance scores
        # IN: sequences x importance scores
        # OUT: sequences x importance scores
        # ---------------------------------------------------           
        thresholded_importances_mat_h5 = '{}.thresholded.h5'.format(importances_mat_h5.split('.h5')[0])
        if not os.path.isfile(thresholded_importances_mat_h5):
        	call_importance_peaks(data_loader, 
        		importances_mat_h5, 
        		thresholded_importances_mat_h5,
        		args.batch_size * 4,
        		task_num,
        		pval=pval)

        if args.plot_importances:
            # visualize a few samples
            sample_seq_dir = 'task_{}.sample_seqs.thresholded'.format(task_num)
            os.system('mkdir -p {}'.format(sample_seq_dir))
            visualize_sample_sequences(thresholded_importances_mat_h5, task_num, sample_seq_dir)
            
                
    return None


def run(args):
    """Run all functions to go from importance scores to de novo motifs
    """

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
    
    interpret_wkm(args,
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
    
    return None
