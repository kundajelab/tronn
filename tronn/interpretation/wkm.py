"""Description: Contains functions/tools for analyzing weighted kmers
extracted from importance weighted sequences.
"""

import os
import glob
import h5py
import json
import numpy as np
import pandas as pd

import tensorflow as tf

from tronn.datalayer import load_data_from_filename_list
from tronn.interpretation.importances import generate_importance_scores
from tronn.interpretation.importances import visualize_sample_sequences
from tronn.models import stdev_cutoff
from tronn.models import models

from tronn.preprocess import one_hot_encode
from scipy.signal import correlate2d

from tronn.visualization import plot_weights

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

        coord.request_stop()
        coord.join(threads)

    return None


def kmer_to_idx(kmer_array, num_bases=5):
    """Get unique identifying position of kmer
    """
    kmer_idx = 0
    for i in range(kmer_array.shape[1]):

        if kmer_array[0,i] > 0:
            num = 1
        elif kmer_array[1,i] > 0:
            num = 2
        elif kmer_array[2,i] > 0:
            num = 3
        elif kmer_array[3,i] > 0:
            num = 4
        else:
            num = 0
        kmer_idx += num * (num_bases**i)
        
    return kmer_idx

def kmer_to_string(kmer_array, num_bases=5):
    """Get unique identifying position of kmer
    """
    kmer_idx = 0
    base_list = []
    # TODO check reversal
    for i in range(kmer_array.shape[1]):

        if kmer_array[0,i] > 0:
            num = 'A'
        elif kmer_array[1,i] > 0:
            num = 'C'
        elif kmer_array[2,i] > 0:
            num = 'G'
        elif kmer_array[3,i] > 0:
            num = 'T'
        else:
            num = 'N'
        base_list.append(num)
        
    return ''.join(base_list)


def idx_to_kmer(idx, kmer_len=6, num_bases=5):
    """From unique index val, get kmer string back
    """
    num_to_base = {0: "N", 1:"A", 2:"C", 3:"G", 4:"T"}

    idx_tmp = idx
    kmer_string = []
    for pos in reversed(range(kmer_len)):
        num = int(idx_tmp / num_bases**pos)
        try:
            kmer_string.append(num_to_base[num])
        except:
            import pdb
            pdb.set_trace()
        idx_tmp -= num * (num_bases**pos)
    
    return ''.join(kmer_string)


def kmerize(importances_h5, task_num, kmer_lens=[6, 8, 10], min_pos=4, num_bases=5):
    """Convert weighted sequence into weighted kmers

    Args:
      importances_h5: file with importances h5 file

    Returns:
      sparse matrix of kmer scores
    """
    importances_key = 'importances_task{}'.format(task_num)

    # determine total cols of matrix
    total_cols = 0
    for kmer_len in kmer_lens:
        total_cols += num_bases**kmer_len
    print total_cols
        
    # determine number of sequences and generate sparse matrix
    with h5py.File(importances_h5, 'r') as hf:
        num_pos_examples = np.sum(hf['labels'][:,task_num] > 0)
        num_examples = hf['regions'].shape[0]

        pos_indices = np.where(hf['labels'][:,task_num] > 0)

    wkm_mat = np.zeros((num_pos_examples, total_cols))
        
    # now start reading into file
    with h5py.File(importances_h5, 'r') as hf:

        # for each kmer:
        for kmer_len_idx in range(len(kmer_lens)):
            print 'kmer len', kmer_lens[kmer_len_idx]
            kmer_len = kmer_lens[kmer_len_idx]
            
            if kmer_len_idx == 0:
                start_idx = 0
            else:
                start_idx = kmer_len_idx * (num_bases**kmer_lens[kmer_len_idx-1])

            current_idx = 0
            for example_idx in pos_indices[0]:

                if current_idx % 100 == 0:
                    print current_idx

                # go through the sequence
                sequence = hf[importances_key][example_idx,:,:]

                for i in range(sequence.shape[1] - kmer_len):
                    weighted_kmer = sequence[:,i:(i+kmer_len)]
                    kmer = (weighted_kmer > 0).astype(int)

                    if np.sum(kmer) < min_pos:
                        continue

                    kmer_idx = kmer_to_idx(kmer)
                    wkm_score = np.sum(weighted_kmer)
                    wkm_mat[current_idx, start_idx+kmer_idx] += wkm_score
                current_idx += 1

        # TODO save out a copy
                
    return wkm_mat


def reduce_kmer_mat(wkm_mat, cutoff=100, kmer_len=6):
    """Remove zeros and things below cutoff 
    """
    #kmer_mask = np.any(wkm_mat > cutoff, axis=0)

    kmer_indices = np.arange(wkm_mat.shape[1])[np.any(wkm_mat > cutoff, axis=0)]
    kmer_strings = [idx_to_kmer(kmer_idx, kmer_len=kmer_len)
                    for kmer_idx in kmer_indices]
    reduced_wkm_mat = wkm_mat[:,np.any(wkm_mat > cutoff, axis=0)]

    # make pandas dataframe
    wkm_df = pd.DataFrame(data=reduced_wkm_mat, columns=kmer_strings)
    wkm_df.to_csv('test.txt', sep='\t')
    
    return wkm_df


def cluster_kmers():
    """Given a distance matrix of kmers, performs Louvain clustering to get
    communities of kmers that can then be merged to make motifs
    """

    # compute distance metric
    # consider jaccard distance



    # and then cluster based on distances
    # use phenograph (wrap in python3 script?)

    

    return None


def generate_offsets(array_1, array_2, offset):
    '''
    This script sets up two sequences with the offset 
    intended. Offset is set by first sequence
    '''
    bases_len = array_1.shape[0]
    seq_len1 = array_1.shape[1]
    seq_len2 = array_2.shape[1]
    
    if offset > 0:
        total_len = np.maximum(seq_len1, offset + seq_len2)
        array_1_padded = np.concatenate((array_1,
                                         np.zeros((bases_len, total_len - seq_len1))),
                                        axis=1)
        array_2_padded = np.concatenate((np.zeros((bases_len, offset)),
                                         array_2,
                                         np.zeros((bases_len, total_len - seq_len2 - offset))),
                                        axis=1)
    elif offset < 0:
        total_len = np.maximum(seq_len2, -offset + seq_len1)
        array_1_padded = np.concatenate((np.zeros((bases_len, -offset)),
                                         array_1,
                                         np.zeros((bases_len, total_len - seq_len1 + offset))),
                                        axis=1)
        array_2_padded = np.concatenate((array_2,
                                         np.zeros((bases_len, total_len - seq_len2))),
                                        axis=1)
    else:
        if seq_len1 > seq_len2:
            total_len = seq_len1
            array_1_padded = array_1
            array_2_padded = np.concatenate((array_2,
                                             np.zeros((bases_len, total_len - seq_len2))),
                                            axis=1)
        elif seq_len1 < seq_len2:
            total_len = seq_len2
            array_2_padded = array_2
            array_1_padded = np.concatenate((array_1,
                                             np.zeros((bases_len, total_len - seq_len1))),
                                            axis=1)
        else:
            array_1_padded = array_1
            array_2_padded = array_2

    return array_1_padded, array_2_padded


def merge_pwms(pwm1, pwm2, offset):
    """Merge pwms by offset
    """

    pwm1_padded, pwm2_padded = generate_offsets(pwm1, pwm2, offset)
    try:
        merged_pwm = pwm1_padded + pwm2_padded
    except:
        import pdb
        pdb.set_trace()
        
    # renormalize by columns
    col_sums = merged_pwm.sum(axis=0)
    normalized_pwm = merged_pwm / col_sums[np.newaxis,:]
    final_pwm = np.nan_to_num(normalized_pwm)
        
    return final_pwm


def xcor_pwms(pwm1, pwm2, allowed_offset=2):
    """cross correlation of pwms
    """

    xcor_vals = correlate2d(pwm1, pwm2, mode='same')

    score = np.max(xcor_vals[1,:])
    offset = np.argmax(xcor_vals[1,:]) - (pwm1.shape[1] / 2 - 1)

    #ordered_offsets = np.argsort(xcor_vals[1,:])

    #offset_index = -1
    #while True:
    #    if (xcor_vals[1,ordered_offsets[offset_index]] - (pwm1.shape[1] / 2 -1)) > allowed_offset:
    #        offset_index -=1
    #    else:
    #        break

    #offset = ordered_offsets[offset_index]
    
    return score, offset


def make_motifs(kmer_list, cor_cutoff=3):
    """Given a group of kmers, forms a PWM motif(s)
    """
    motifs = [] # list of PWMs

    tmp_kmer_list = list(kmer_list)

    while len(tmp_kmer_list) > 0:
#        import pdb
#        pdb.set_trace()

        
        current_kmer = tmp_kmer_list[0]
        
        print "current_kmer:", current_kmer
        print 'kmers:', tmp_kmer_list
        print 'motifs:', [kmer_to_string(motif) for motif in motifs]

        used = 0
        kmer_len = len(current_kmer)
        # TODO one hot encode kmer
        current_pwm = np.squeeze(one_hot_encode(current_kmer)).transpose(1,0)

        #import pdb
        #pdb.set_trace()
        
        # first compare to motifs to greedily add
        for i in range(len(motifs)):
            motif = motifs[i]
            score, offset = xcor_pwms(motif, current_pwm)

            if score > cor_cutoff:
                pwm = merge_pwms(motif, current_pwm, offset)
                del motifs[i]
                motifs.append(pwm)
                tmp_kmer_list.remove(current_kmer)
                used = True
                break

        if used:
            continue
            
        # then check other kmers
        if len(tmp_kmer_list) != 1:
        
            for other_kmer in tmp_kmer_list[1:]:
                other_pwm = np.squeeze(one_hot_encode(other_kmer)).transpose(1,0)
                
                
                score, offset = xcor_pwms(current_pwm, other_pwm)

                if score > cor_cutoff:
                    print "used", other_kmer
                    pwm = merge_pwms(current_pwm, other_pwm, offset)
                    motifs.append(pwm)
                    tmp_kmer_list.remove(current_kmer)
                    tmp_kmer_list.remove(other_kmer)
                    used = True
                    break

        if used:
            continue

        # if nothing else, just put into motif list
        tmp_kmer_list.remove(current_kmer)
        motifs.append(current_pwm)
        
        
    return motifs


def agglom_motifs(kmer_list):
    """
    go through hierarchical until no more combinations
    """

    all_motif_lists = []
    motif_list = [np.squeeze(one_hot_encode(kmer)).transpose(1,0) for kmer in kmer_list]
    
    while True:

        if len(motif_list) == 1:
            break
        
        # start with current motif list
        num_motifs = len(motif_list)
        
        xcor_mat = np.zeros((num_motifs, num_motifs))
        
        # calculate every pair of xcor
        for i in range(len(motif_list)):
            for j in range(len(motif_list)):
                score, offset = xcor_pwms(motif_list[i], motif_list[j])
                xcor_mat[i,j] = score

        # take the best xcor (above cutoff) and merge
        np.fill_diagonal(xcor_mat, 0)
        pwm1_idx, pwm2_idx = np.unravel_index(np.argmax(xcor_mat), dims=xcor_mat.shape)

        print "chose", kmer_to_string(motif_list[pwm1_idx]), kmer_to_string(motif_list[pwm2_idx]), offset
        
        score, offset = xcor_pwms(motif_list[pwm1_idx], motif_list[pwm2_idx])
        merged_pwm = merge_pwms(motif_list[pwm1_idx], motif_list[pwm2_idx], offset)
        motif_list.append(merged_pwm)
        del motif_list[pwm1_idx]
        del motif_list[pwm2_idx]


        all_motif_lists.append(list(motif_list))

        print [kmer_to_string(motif) for motif in all_motif_lists[-1]]


    return all_motif_lists


def make_motif_sets(clustered_file):
    """Given clusters of kmers, make motifs
    """

    # For now just merge by kmers (don't worry about exact heights yet)
    data = pd.read_table(clustered_file, index_col=0)
    communities = list(set(data['community']))
    communities.remove(-1) # remove ones not in a community

    # for each community group:
    for community in communities:
        print "community:", community
        # get the kmers
        community_df = data.loc[data['community'] == community]
        kmers = community_df.index.tolist()
        if 'Unnamed: 0' in kmers:
            kmers.remove('Unnamed: 0')
        
        #motifs = make_motifs(kmers)
        motifs = agglom_motifs(kmers)

        print len(motifs)
        #motif_strings = [kmer_to_string(motif) for motif in motifs]
        #print motif_strings

        prefix = 'test'
        for motif_idx in range(len(motifs[-3])):
            motif = motifs[-3][motif_idx]
            plot_weights(motif, '{0}.community{1}.motif{2}.png'.format(prefix, community, motif_idx), figsize=(20,2))
        
        import pdb
        pdb.set_trace()
        
    import pdb
    pdb.set_trace()
        
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
        thresholded_importances_mat_h5 = 'task_{}.importances.thresholded.h5'.format(task_num)
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

	# ---------------------------------------------------
        # Convert into weighted kmers
        # IN: sequences x importance scores
        # OUT: sequences x importance scores
        # ---------------------------------------------------
        wkm_h5 = 'task_{}.wkm.h5'.format(task_num)
        if not os.path.isfile(wkm_h5):
            # first convert to wkm
            if False:
                wkm_sparse_mat = kmerize(thresholded_importances_mat_h5, task_num_idx, kmer_lens=[6])

                # then remove zero columns
                # TODO need to keep the kmer position names
                wkm_sparse_mat_redux = reduce_kmer_mat(wkm_sparse_mat, cutoff=100)

                # GO OUT TO PHENOGRAPH HERE
            
            # make motifs at this point?
            make_motif_sets('/srv/scratch/shared/indra/dskim89/home/git/tronn/scripts/communities.txt')

            


            
            # and then cluster
            #clusters = cluster_kmers(wkm_sparse_mat_redux)

            quit()
        
        
            
                
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
