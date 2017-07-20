"""Description: Contains functions/tools for analyzing weighted kmers
extracted from importance weighted sequences.
"""

import os
import glob
import h5py
import json
import math
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

from tronn.interpretation.motifs import run_pwm_convolution
from tronn.interpretation.motifs import extract_positives_from_motif_mat

from tronn.util.parallelize import *

import phenograph

import scipy.stats


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


def kmer_to_string2(kmer_array, num_bases=5):
    """Get unique identifying position of kmer
    """
    kmer_idx = 0
    base_list = []
    # TODO check reversal
    for i in range(kmer_array.shape[1]):

        if np.sum(kmer_array[:,i]) == 0:
            bp = 'N'
        else:
            idx = np.argmax(kmer_array[:,i])
        
            if idx == 0:
                bp = 'A'
            elif idx == 1:
                bp = 'C'
            elif idx == 2:
                bp = 'G'
            elif idx == 3:
                bp = 'T'

        base_list.append(bp)
        
    return ''.join(base_list)


def idx_to_kmer(idx, kmer_len=7, num_bases=5):
    """From unique index val, get kmer string back
    """
    num_to_base = {0: "N", 1:"A", 2:"C", 3:"G", 4:"T"}

    idx_tmp = idx
    reverse_kmer_string = []
    for pos in reversed(range(kmer_len)):
        num = int(idx_tmp / num_bases**pos)
        try:
            reverse_kmer_string.append(num_to_base[num])
        except:
            import pdb
            pdb.set_trace()
        idx_tmp -= num * (num_bases**pos)
    kmer_string = reversed(reverse_kmer_string)
    
    return ''.join(kmer_string)


def kmerize(importances_h5, task_num, kmer_lens=[6, 8, 10], num_bases=5):
    """Convert weighted sequence into weighted kmers

    Args:
      importances_h5: file with importances h5 file

    Returns:
      matrix of kmer scores
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
    onehot_wkm_mat = np.zeros((num_pos_examples, total_cols, 4, max(kmer_lens)))
    
    # now start reading into file
    with h5py.File(importances_h5, 'r') as hf:

        # for each kmer:
        for kmer_len_idx in range(len(kmer_lens)):
            print 'kmer len', kmer_lens[kmer_len_idx]
            kmer_len = kmer_lens[kmer_len_idx]

            min_pos = kmer_len - 2
            
            if kmer_len_idx == 0:
                start_idx = 0
            else:
                start_idx = kmer_len_idx * (num_bases**kmer_lens[kmer_len_idx-1])

            current_idx = 0
            for example_idx in pos_indices[0]:

                if current_idx % 500 == 0:
                    print current_idx

                # go through the sequence
                sequence = hf[importances_key][example_idx,:,:]

                for i in range(sequence.shape[1] - kmer_len):
                    weighted_kmer = sequence[:,i:(i+kmer_len)]
                    kmer = (weighted_kmer > 0).astype(int)

                    if np.sum(kmer) < min_pos:
                        continue

                    kmer_idx = kmer_to_idx(kmer)
                    wkm_score = np.sum(weighted_kmer) # bp adjusted score?
                    wkm_mat[current_idx, start_idx+kmer_idx] += wkm_score

                    # TODO - adjust kmer as needed
                    # TODO consider adjusting for base pair importances - ex CNNGT for p63 is only two bp of importance
                    onehot_wkm_mat[current_idx, start_idx+kmer_idx,:,0:kmer_len] += weighted_kmer
                    
                current_idx += 1

    onehot_wkm_avg = np.mean(onehot_wkm_mat, axis=0)

    return wkm_mat, onehot_wkm_avg

def kmerize2(
        h5_file,
        kmer_lens=[6, 8, 10],
        num_bases=5,
        is_importances=False,
        task_num=0,
        save=False,
        save_file='tmp.h5'):
    """Convert weighted sequence into weighted kmers

    Args:
      importances_h5: file with importances h5 file

    Returns:
      matrix of kmer scores
    """

    if is_importances:
        feature_key = 'importances_task{}'.format(task_num)
    else:
        feature_key = 'features'
    
    # determine total cols of matrix (number of kmer features)
    total_cols = 0
    for kmer_len in kmer_lens:
        total_cols += num_bases**kmer_len
    print 'total kmer features:', total_cols
        
    # determine number of sequences and generate matrix
    with h5py.File(h5_file, 'r') as hf:
        if is_importances:
            num_examples = np.sum(hf['labels'][:,task_num] > 0)
            example_indices = np.where(hf['labels'][:,task_num] > 0)
        else:
            num_examples = hf['regions'].shape[0]
            example_indices = [np.array(range(num_examples))]
        num_tasks = hf['labels'].shape[1]
    print 'total examples:', num_examples
            

    if is_importances:
        wkm_mat = np.zeros((num_examples, total_cols))
        onehot_wkm_mat = np.zeros((num_examples, total_cols, 4, max(kmer_lens)))

    with h5py.File(save_file, 'w') as out:

        # set up datasets
        importance_mat = out.create_dataset(feature_key,
                                            [num_examples, total_cols])
        labels_mat = out.create_dataset('labels',
                                        [num_examples, num_tasks])
        regions_mat = out.create_dataset('regions',
                                         [num_examples, 1],
                                         dtype='S100')
        
        
        # now start reading into file
        with h5py.File(h5_file, 'r') as hf:
            
            # for each kmer, figure out start position
            for kmer_len_idx in range(len(kmer_lens)):
                print 'kmer len', kmer_lens[kmer_len_idx]
                kmer_len = kmer_lens[kmer_len_idx]
                min_pos = kmer_len - 2
                if kmer_len_idx == 0:
                    start_idx = 0
                else:
                    start_idx = kmer_len_idx * (num_bases**kmer_lens[kmer_len_idx-1])

                # run through all the examples
                current_idx = 0
                for example_idx in example_indices[0]:
                    
                    if current_idx % 500 == 0:
                        print current_idx

                    # go through the sequence

                    if not is_importances:
                        wkm_vec = np.zeros((total_cols))
                        sequence = np.squeeze(hf[feature_key][example_idx,:,:,:]).transpose(1,0)
                    else:
                        sequence = hf[feature_key][example_idx,:,:]
                    
                    for i in range(sequence.shape[1] - kmer_len):
                        weighted_kmer = sequence[:,i:(i+kmer_len)]
                        kmer = (weighted_kmer > 0).astype(int)
                        
                        if np.sum(kmer) < min_pos:
                            continue

                        kmer_idx = kmer_to_idx(kmer)

                        if is_importances:
                            wkm_score = np.sum(weighted_kmer) # bp adjusted score?
                            wkm_mat[current_idx, start_idx+kmer_idx] += wkm_score
                            onehot_wkm_mat[current_idx, start_idx+kmer_idx,:,0:kmer_len] += weighted_kmer
                        else:
                            wkm_vec[start_idx+kmer_idx] += 1

                    # and then save if necessary
                    if save:
                        importance_mat[current_idx,:] = wkm_vec
                        labels_mat[current_idx,:] = hf['labels'][example_idx,:]
                        regions_mat[current_idx] = hf['regions'][example_idx]
                    
                    current_idx += 1

    if not save:
        os.system('rm {}'.format(save_file))
                    
    if is_importances:
        onehot_wkm_avg = np.mean(onehot_wkm_mat, axis=0)
        return wkm_mat, onehot_wkm_avg
    else:
        return None

def kmerize_parallel(in_dir, out_dir, parallel=24):
    """Utilize func_worker to run on multiple files
    """
    
    kmerize_queue = setup_multiprocessing_queue()

    # find h5 files
    h5_files = glob.glob('{}/*.h5'.format(in_dir))
    print "Found {} h5 files".format(len(h5_files))

    for h5_file in h5_files:
        out_file = '{0}/{1}.kmers.h5'.format(out_dir,
                                             os.path.basename(h5_file).split('.h5')[0])
        kmerize_args = [h5_file, [6], 5, False, 0, True, out_file]

        if not os.path.isfile(out_file):
            kmerize_queue.put([kmerize2, kmerize_args])

    run_in_parallel(kmerize_queue, parallel=parallel, wait=True)

    return None


def reduce_kmer_mat(wkm_mat, kmer_len, cutoff=100, topk=200):
    """Remove zeros and things below cutoff 
    """

    if False: # ie old code
        #kmer_mask = np.any(wkm_mat > cutoff, axis=0)
        
        kmer_indices = np.arange(wkm_mat.shape[1])[np.any(wkm_mat > cutoff, axis=0)]
        kmer_strings = [idx_to_kmer(kmer_idx, kmer_len=kmer_len)
                        for kmer_idx in kmer_indices]
        #reduced_wkm_mat = wkm_mat[:,np.any(wkm_mat > cutoff, axis=0)]

    # using top k kmer mode for now
    wkm_colsums = np.sum(wkm_mat, axis=0)
    kmer_indices = np.arange(wkm_mat.shape[1])[np.argpartition(wkm_colsums, -topk)[-topk:]]
    kmer_strings = [idx_to_kmer(kmer_idx, kmer_len=kmer_len)
                        for kmer_idx in kmer_indices]
    reduced_wkm_mat = wkm_mat[:,np.argpartition(wkm_colsums, -topk)[-topk:]]
    
    # make pandas dataframe
    wkm_df = pd.DataFrame(data=reduced_wkm_mat, columns=kmer_strings)
    wkm_df.to_csv('test.txt', sep='\t')
    
    return wkm_df


def cluster_kmers():
    """Given a distance matrix of kmers, performs Louvain clustering to get
    communities of kmers that can then be merged to make motifs
    """

    # TODO phenograph

    
    # compute distance metric
    # consider jaccard distance



    # and then cluster based on distances
    # use phenograph (wrap in python3 script?)

    

    return None


def generate_offsets(array_1, array_2, offset):
    '''This script sets up two sequences with the offset 
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


def normalize_pwm(pwm):
    """normalize columns so each column adds up to 1
    """
    col_sums = pwm.sum(axis=0)
    normalized_pwm = pwm / np.amax(col_sums[np.newaxis,:])
    final_pwm = np.nan_to_num(normalized_pwm)

    return final_pwm


def normalize_pwm2(pwm):
    """normalize so that it matches N(0,1)
    ie subtract mean and divide by standard dev
    """

    mean = np.mean(pwm)
    std = np.std(pwm)

    normalized_pwm = (pwm - mean) / std
    final_pwm = np.nan_to_num(normalized_pwm)

    return final_pwm
    
def chomp_pwm(pwm):
    """chomp off trailing/leading Ns
    """
    chomped_pwm = pwm[:,~np.all(pwm == 0, axis=0)]

    assert(pwm.shape[0] == 4)

    return chomped_pwm


def merge_pwms(pwm1, pwm2, offset, normalize=False):
    """Merge pwms by offset
    """

    pwm1_padded, pwm2_padded = generate_offsets(pwm1, pwm2, offset)
    try:
        merged_pwm = pwm1_padded + pwm2_padded
    except:
        import pdb
        pdb.set_trace()

    if normalize:
        final_pwm = normalize_pwm(merged_pwm)
    else:
        final_pwm = merged_pwm
        
    return final_pwm


def xcor_pwms(pwm1, pwm2, normalize=True):
    """cross correlation of pwms
    """

    if normalize:
        pwm1_norm = normalize_pwm2(pwm1)
        pwm2_norm = normalize_pwm2(pwm2)
    else:
        pwm1_norm = pwm1
        pwm2_norm = pwm2

    # TODO log first?
    xcor_vals = correlate2d(pwm1_norm, pwm2_norm, mode='same')
    #xcor_vals = correlate2d(pwm1_norm, pwm2_norm)

    if False:
        # and divide by length of shorter one to get max val be 1
        # TODO divide by max possible score (sum of each val squared) to get max
        max_attainable_val = np.sum(np.power(pwm1_norm, 2))
        #max_attainable_val = np.sum(np.power((pwm1_norm > 0).astype(int), 2))
        
        #xcor_norm = xcor_vals / min(pwm1_norm.shape[1], pwm2_norm.shape[1])
        xcor_norm = xcor_vals / max_attainable_val
        
        score = np.max(xcor_norm[1,:]) # NOTE - this depends on correlate2d mode
        offset = np.argmax(xcor_norm[1,:]) - int(math.ceil(pwm2_norm.shape[1] / 2.) - 1)
        #offset = np.argmax(xcor_norm[3,:]) - pwm2.shape[1] # NOTE this depends on correlate2d mode

    elif False:
        perfect_match_score = np.sum(np.power(pwm1_norm, 2)) # exact match is like multiplying it by itself
        pwm_diff = np.absolute(xcor_vals[1,:] - perfect_match_score) / pwm1_norm.shape[1] # get absolute diff between PWMs and normalize by length

        score = np.min(pwm_diff)
        offset = np.argmin(pwm_diff) - int(math.ceil(pwm2_norm.shape[1] / 2.) - 1)

    else:
        xcor_norm = xcor_vals / (pwm1_norm.shape[0]*pwm1_norm.shape[1])
        score = np.max(xcor_norm[1,:])
        offset = np.argmax(xcor_norm[1,:]) - int(math.ceil(pwm2_norm.shape[1] / 2.) - 1)
        
    #ordered_offsets = np.argsort(xcor_vals[1,:])

    #offset_index = -1
    #while True:
    #    if (xcor_vals[1,ordered_offsets[offset_index]] - (pwm1.shape[1] / 2 -1)) > allowed_offset:
    #        offset_index -=1
    #    else:
    #        break

    #offset = ordered_offsets[offset_index]

    #import pdb
    #pdb.set_trace()
    
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


def agglom_motifs(starting_motif_list, cut_fract):
    """
    go through hierarchical until no more combinations
    also set up score tracking to choose when to stop merging
    """
    motif_list = list(starting_motif_list)
    all_motif_lists = []
    scores = np.zeros((len(motif_list)))
    score_idx = 0
    
    while True:

        motif_list = [chomp_pwm(motif) for motif in motif_list]

        print [kmer_to_string2(normalize_pwm(motif)) for motif in motif_list]
        
        if len(motif_list) == 1:
            break
        
        # start with current motif list
        num_motifs = len(motif_list)
        
        xcor_mat = np.zeros((num_motifs, num_motifs))
        
        # calculate every pair of xcor
        for i in range(len(motif_list)):
            for j in range(len(motif_list)):
                # for this calculation, normalize (but otherwise, keep counts)
                score, offset = xcor_pwms(motif_list[i], motif_list[j])
                xcor_mat[i,j] = score

        if True:
            # take the best xcor (above cutoff) and merge
            np.fill_diagonal(xcor_mat, 0)
            pwm1_idx, pwm2_idx = np.unravel_index(np.argmax(xcor_mat), dims=xcor_mat.shape)
            
            print "chose", kmer_to_string(motif_list[pwm1_idx]), kmer_to_string(motif_list[pwm2_idx])
            
            score, offset = xcor_pwms(motif_list[pwm1_idx], motif_list[pwm2_idx])
            print score, offset
            merged_pwm = merge_pwms(motif_list[pwm1_idx], motif_list[pwm2_idx], offset)

            motif_list.append(merged_pwm)
            del motif_list[pwm1_idx]
            del motif_list[pwm2_idx]
            
            scores[score_idx] = score
            all_motif_lists.append(list(motif_list))
            
            score_idx += 1

        else:
            # take the best xcor (above cutoff) and merge
            np.fill_diagonal(xcor_mat, 1000)
            pwm1_idx, pwm2_idx = np.unravel_index(np.argmin(xcor_mat), dims=xcor_mat.shape)
            
            print "chose", kmer_to_string(motif_list[pwm1_idx]), kmer_to_string(motif_list[pwm2_idx])

            score, offset = xcor_pwms(motif_list[pwm1_idx], motif_list[pwm2_idx])
            print score, offset
            merged_pwm = merge_pwms(motif_list[pwm1_idx], motif_list[pwm2_idx], offset)

            motif_list.append(merged_pwm)
            del motif_list[pwm1_idx]
            del motif_list[pwm2_idx]
            
            scores[score_idx] = score
            all_motif_lists.append(list(motif_list))
            
            score_idx += 1

            #print [kmer_to_string(motif) for motif in all_motif_lists[-1]]
            
        #scores_fract = scores / scores[0]
    scores_fract = scores
    #stop_idx = np.argmin(scores_fract < 0.15) - 2
    stop_idx = np.argmin(scores_fract > cut_fract) - 2

    return all_motif_lists[stop_idx]


def write_pwm(pwm_file, pwm, pwm_name):
    """Append a PWM (normalized to center at zero) to a motif file
    """
    normalized_pwm = normalize_pwm2(pwm)
    
    with open(pwm_file, 'a') as fp:
        fp.write('>{}\n'.format(pwm_name))
        for i in range(normalized_pwm.shape[1]):
            vals = normalized_pwm[:,i].tolist()
            val_strings = [str(val) for val in vals]
            fp.write('{}\n'.format('\t'.join(val_strings)))

    return None
    

def make_motif_sets(clustered_df, wkm_array, prefix, cut_fract=0.7):
    """Given clusters of kmers, make motifs
    """

    # For now just merge by kmers (don't worry about exact heights yet)
    #data = pd.read_table(clustered_file, index_col=0)
    data = clustered_df
    communities = list(set(data['community']))
    if (-1 in communities):
        communities.remove(-1) # remove ones not in a community
    community_motif_sets = []
    
    # for each community group:
    for community in communities:
        print "community:", community
        # get the kmers
        community_df = data.loc[data['community'] == community]
        kmers = community_df.index.tolist() # sort to make deterministic (for now)
        if 'Unnamed: 0' in kmers:
            kmers.remove('Unnamed: 0')
        kmers_scores = community_df.sum(axis=1).tolist()

        #kmer_indices = [kmer_to_idx(kmer) for kmer in kmers]
        
        #motif_list = [kmers_scores[i] * np.squeeze(one_hot_encode(kmers[i])).transpose(1,0) for i in range(len(kmers))]
        # motifs: you want to take the kmer PWMs (as identified by the NN) but also weight by number of sequences
        motif_list = [wkm_array[
            kmer_to_idx(
                np.squeeze(
                    one_hot_encode(kmers[i])).transpose(1,0)
            ),:,:] for i in range(len(kmers))]
        #motif_list = [kmers_scores[i] * wkm_array[kmer_indices[i],:,:] for i in range(len(kmers))]
        # and sort
        kmers_sort_indices = np.argsort([kmer_to_string2(motif) for motif in motif_list])
        motif_list_sorted = [motif_list[i] for i in kmers_sort_indices]

        print "kmers: ", kmers
        
        motifs = agglom_motifs(motif_list_sorted, cut_fract)
        
        normalized_motifs = [normalize_pwm(motif) for motif in motifs]
        motif_strings = [kmer_to_string(motif) for motif in normalized_motifs]
        print "motifs ({0}): {1}".format(len(motifs), motif_strings)

        for motif_idx in range(len(motifs)):
            motif = normalized_motifs[motif_idx]
            plot_weights(motif, '{0}.community{1}.motif{2}.png'.format(prefix, community, motif_idx), figsize=(motif.shape[1],2))

        community_motif_sets.append(motifs)

    # TODO: save out to PWM file
    # set up PWM (use PWM formula)
    flat_motif_list = [motif for community_list in community_motif_sets for motif in community_list]
    # sort motifs by kmer name
    flat_motif_ordered_indices = np.argsort([kmer_to_string2(motif) for motif in flat_motif_list])
    flat_motifs_ordered = [flat_motif_list[i] for i in flat_motif_ordered_indices]
    print [kmer_to_string2(motif) for motif in flat_motifs_ordered]
    master_motifs = agglom_motifs(flat_motifs_ordered, cut_fract=0.8)

    # write out to PWM file
    # TODO put in optional MEME tool to name the motifs by closest hit(s)
    for motif_idx in range(len(master_motifs)):
        motif_name = '{0}.motif_{1}'.format(prefix, motif_idx)
        write_pwm('{}.motif_file.txt'.format(prefix), master_motifs[motif_idx], motif_name)

    # and plot it out so you have a representation of the motif
    normalized_master_motifs = [normalize_pwm(motif) for motif in master_motifs]
    print "master_list:", [kmer_to_string2(motif) for motif in normalized_master_motifs]
    for motif_idx in range(len(master_motifs)):
        motif = normalized_master_motifs[motif_idx]
        plot_weights(motif, '{0}.master.motif{1}.png'.format(prefix, motif_idx), figsize=(motif.shape[1],2))
    
    return None


def get_sequence_communities(text_mat_file, prefix):
    """Cluster sequences by which motifs they have. Uses phenograph - this is good
    because it has a way of ignoring things that don't really belong in clusters
    """

    data = pd.read_table(text_mat_file)

    if 'Unnamed: 0' in data.columns:
        del data['Unnamed: 0']

    # TODO change this later to keep locations
    del data['indices']

    # normalize first
    data_norm = data.apply(scipy.stats.zscore, axis=1)

    
    data_npy = data_norm.as_matrix()
    communities, graph, Q = phenograph.cluster(data_npy)
    data_norm['community'] = communities

    # sort by community
    data_sorted = data_norm.sort_values('community')


    # For each community, save out significant motifs (ie, greater than 1 stdev?)
    communities = list(np.unique(communities))

    # TODO save out motif grammar lists
    grammar_file = '{}.grammars.txt'.format(prefix)
    with open(grammar_file, 'w') as out:
        for community in communities:
            community_data = data_sorted.loc[data_sorted['community'] == community]
            print community_data.shape
            
            del community_data['community']
            community_motif_avg = community_data.mean(axis=0)
            
            community_motifs = community_data.loc[:,community_motif_avg > 0.5] #TODO move parameter

            print community_motifs.columns.tolist()
            out.write('{0}.{1}\t{2}\n'.format(prefix, community,'\t'.join(community_motifs.columns.tolist())))
        
    print communities
    
    seq_communities_file = '{}.seq_communities.txt'.format(prefix)
    data_sorted.to_csv(seq_communities_file, sep='\t')
    
    return grammar_file, seq_communities_file


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
                kmer_len = 7
                
                wkm_full, onehot_wkm_full = kmerize(thresholded_importances_mat_h5, task_num, kmer_lens=[kmer_len])

                # then remove zero columns
                # TODO need to keep the kmer position names
                wkm_reduced = reduce_kmer_mat(wkm_full, kmer_len, cutoff=120) # for 6, 100 was good

                print "number kmers kept", wkm_reduced.shape[1]
                
                # import and run phenograph here
                phenograph_results_file = 'task_{}.phenograph.txt'.format(task_num)

                wkm_reduced_transposed = wkm_reduced.transpose().as_matrix()
                communities, graph, Q = phenograph.cluster(wkm_reduced_transposed)
                                                           #k=15,
                                                           #min_cluster_size=10)
                
                # and save results out to various matrices
                sort_indices = np.argsort(communities)
                data_sorted = wkm_reduced_transposed[sort_indices,:]
                communities_sorted = communities[sort_indices]
                columns_sorted = wkm_reduced.columns[sort_indices]
                
                out_df = pd.DataFrame(data=data_sorted, index=columns_sorted)
                out_df['community'] = communities_sorted
                out_df.to_csv(phenograph_results_file, sep='\t')
                
                # make motifs at this point
                # and save out as a PWM file
                make_motif_sets(out_df, onehot_wkm_full, 'task_{}'.format(task_num))

                # and here also make some educated guesses for what PWM this matches (tomtom?)
                # TODO convert PWM to meme format (see basset code)
                # also convert HOCOMOCO to meme format

            motif_mat_h5 = 'task_{}.wkm.motif_mat.h5'.format(task_num)
            if not os.path.isfile(motif_mat_h5):
                run_pwm_convolution(
                    data_loader,
                    importances_mat_h5,
                    motif_mat_h5,
                    args.batch_size * 4,
                    'task_{}.motif_file.txt'.format(task_num),
                    task_num)

            # ---------------------------------------------------
            # extract the positives to cluster in R and visualize
            # IN: sequences x motifs
            # OUT: positive sequences x motifs
            # ---------------------------------------------------
            pos_motif_mat = 'task_{}.wkm_mat.positives.txt.gz'.format(task_num)
            if not os.path.isfile(pos_motif_mat):
                extract_positives_from_motif_mat(motif_mat_h5, pos_motif_mat, task_num)


            # TODO isolate this bit here for now
            #os.system('mkdir -p ')
            
                
            #  phenograph here again for the clustering
            grammar_file, seq_communities_file = get_sequence_communities(pos_motif_mat, 'task_{}'.format(task_num))

            # TODO fix seq communities file to keep chrom info. take that and make a BED? use preprocess code
            
            # TODO from there for every pair of motifs, set up model and get back dependencies.
            # TODO take phenograph clusters and use ISM to induce the dependencies
            

            quit()
            
            continue
            
            
                
            # ---------------------------------------------------
            # Cluster positives in R and output subgroups
            # IN: positive sequences x motifs
            # OUT: subgroups of sequences
            # ---------------------------------------------------
            cluster_dir = 'task_{}.positives.wkm.clustered'.format(task_num)
            if not os.path.isdir(cluster_dir):
                os.system('mkdir -p {}'.format(cluster_dir))
                prefix = 'task_{}'.format(task_num)
                os.system('run_region_clustering.R {0} 50 {1} {2}/{3}'.format(pos_motif_mat,
                                                                              dendro_cutoffs[task_num_idx],
                                                                              cluster_dir,
                                                                              prefix))

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
