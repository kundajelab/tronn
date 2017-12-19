"""Description: code to analyze motif matrices
"""

import h5py
import math
import numpy as np
import pandas as pd
import scipy.stats

import tensorflow as tf

from scipy.signal import correlate2d

from tronn.visualization import plot_weights


def get_encode_pwms(motif_file, as_dict=False):
    """Extracts motifs into PWM class format
    """
    # option to set up as dict or list
    if as_dict:
        pwms = {}
    else:
        pwms = []

    # open motif file and read
    with open(motif_file) as fp:
        line = fp.readline().strip()
        while True:
            if line == '':
                break
            
            header = line.strip('>').strip()
            weights = []
            
            while True:
                line = fp.readline()
                if line == '' or line[0] == '>': break
                weights.append(map(float, line.split()))

            pwm = PWM(np.array(weights).transpose(1,0), header)

            # store into dict or list
            if as_dict:
                pwms[header] = pwm
            else:
                pwms.append(pwm)
                
    return pwms


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
                                                                


class PWM(object):
    """PWM class for PWM operations"""
    
    def __init__(self, weights, name=None, threshold=None):
        self.weights = weights
        self.name = name
        self.threshold = threshold

        
    def normalize(self, style="gaussian", in_place=True):
        """Normalize pwm
        """
        if style == "gaussian":
            mean = np.mean(self.weights)
            std = np.std(self.weights)
            normalized_weights = (self.weights - mean) / std
        elif style == "probabilities":
            col_sums = self.weights.sum(axis=0)
            normalized_pwm_tmp = self.weights / np.amax(col_sums[np.newaxis,:])
            normalized_weights = np.nan_to_num(normalized_pwm_tmp)
        elif style == "log_odds":
            print "Not yet implemented"
        else:
            print "Style not recognized"
            
        if in_place:
            self.weights = normalized_weights
            return self
        else:
            new_pwm = PWM(normalized_weights, "{}.norm".format(self.name))
            return new_pwm

        
    def xcor(self, pwm, normalize=True):
        """Compute xcor score with other motif, return score and offset relative to first pwm
        """
        if normalize:
            pwm1_norm = self.normalize(in_place=False)
            pwm2_norm = pwm.normalize(in_place=False)
        else:
            pwm1_norm = pwm1
            pwm2_norm = pwm2

        # calculate xcor
        xcor_vals = correlate2d(pwm1_norm.weights, pwm2_norm.weights, mode='same')
        xcor_norm = xcor_vals / (pwm1_norm.weights.shape[0]*pwm1_norm.weights.shape[1])
        score = np.max(xcor_norm[1,:])
        offset = np.argmax(xcor_norm[1,:]) - int(math.ceil(pwm2_norm.weights.shape[1] / 2.) - 1)

        return score, offset

    
    def chomp(self):
        """Remove leading/trailing Ns. In place.
        """
        chomped_weights = self.weights[:,~np.all(self.weights == 0, axis=0)]
        self.weights = chomped_weights
        assert(self.weights.shape[0] == 4)
        
        return self
    

    def merge(self, pwm, offset, new_name=None, normalize=False):
        """Merge in another PWM and output a new PWM
        """
        weights1_padded, weights2_padded = generate_offsets(self.weights, pwm.weights, offset)
        try:
            merged_pwm = weights1_padded + weights2_padded
        except:
            import pdb
            pdb.set_trace()

        new_pwm = PWM(merged_pwm, new_name)

        if normalize:
            new_pwm.normalize()

        return new_pwm

    
    def to_motif_file(self, motif_file):
        """Write PWM out to file
        """
        with open(motif_file, 'a') as fp:
            fp.write('>{}\n'.format(self.name))
            for i in range(self.weights.shape[1]):
                vals = self.weights[:,i].tolist()
                val_strings = [str(val) for val in vals]
                fp.write('{}\n'.format('\t'.join(val_strings)))
        
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


def bootstrap_fdr_v2(
        pwm_counts_mat_h5,
        counts_key, # which pwm counts table to use
        pwm_names,
        out_prefix,
        task_idx, # which labels to pull (by idx)
        global_importances=False,
        region_set=None,
        bootstrap_num=99,
        fdr=0.05):
    """Given a motif matrix and labels, calculate a bootstrap FDR
    """
    with h5py.File(pwm_counts_mat_h5, 'r') as hf:
        
        at_least_one_pos = (hf["negative"][:,0] == 0).astype(int)
        #at_least_one_pos = (np.sum(hf["labels"][:], axis=1) > 0).astype(int)
        
        # better to pull it all into memory to slice fast 
        if global_importances:
            labels = at_least_one_pos
            pwm_scores = hf[counts_key][:]
        else:
            labels = hf['labels'][:][at_least_one_pos > 0, task_idx]
            pwm_scores = hf[counts_key][:][at_least_one_pos > 0,:]
        
    # first calculate ranks w just positives
    pos_array = pwm_scores[labels > 0,:]
    pos_array_summed = np.sum(pos_array, axis=0)
    
    # extract top k for spot check
    top_counts_k = 100
    top_counts_indices = np.argpartition(pos_array_summed, -top_counts_k)[-top_counts_k:]
    most_seen_pwms = [pwm_names[i] for i in xrange(len(pwm_names)) if i in top_counts_indices]
    print most_seen_pwms
        
    # now calculate the true diff between pos and neg sets for motifs
    neg_array = pwm_scores[labels < 1,:]
    neg_array_summed = np.sum(neg_array, axis=0)
        
    true_diff = pos_array_summed - neg_array_summed

    # set up results array 
    bootstraps = np.zeros((bootstrap_num+1, pwm_scores.shape[1])) 
    bootstraps[0,:] = true_diff

    # now do bootstraps
    for i in range(bootstrap_num):
        
        if i % 10 == 0: print i
        
        # randomly select pos and neg examples
        # NOTE: this is as unbalanced as the dataset is...
        # TODO - dk, balance this?
        shuffled_labels = np.random.permutation(labels)
        pos_array = pwm_scores[shuffled_labels > 0,:]
        pos_array_summed = np.sum(pos_array, axis=0)

        neg_array = pwm_scores[shuffled_labels < 1,:]
        neg_array_summed = np.sum(neg_array, axis=0)

        bootstrap_diff = pos_array_summed - neg_array_summed
        
        # save into vector 
        bootstraps[i+1,:] = bootstrap_diff
        
    # convert to ranks and save out 
    bootstrap_df = pd.DataFrame(data=bootstraps, columns=pwm_names)
    #bootstrap_df = bootstrap_df.loc[:,(bootstrap_df != 0).any(axis=0)] # remove all zero columns
    bootstrap_df.to_csv("{}.permutations.txt".format(out_prefix), sep='\t')
    bootstrap_ranks_df = bootstrap_df.rank(ascending=False, pct=True)
    
    pos_fdr = bootstrap_ranks_df.iloc[0,:]
    pos_fdr_file = "{}.txt".format(out_prefix)
    pos_fdr.to_csv(pos_fdr_file, sep='\t')

    # also save out a list of those that passed the FDR cutoff
    fdr_cutoff = pos_fdr.ix[pos_fdr < fdr]
    fdr_cutoff_file = "{}.cutoff.txt".format(out_prefix)
    fdr_cutoff.to_csv(fdr_cutoff_file, sep='\t')

    # also save one that is adjusted for most seen
    fdr_cutoff = fdr_cutoff[fdr_cutoff.index.isin(most_seen_pwms)]
    fdr_cutoff_mostseen_file = "{}.cutoff.most_seen.txt".format(out_prefix)
    fdr_cutoff.to_csv(fdr_cutoff_mostseen_file, sep='\t')

    # save a vector of counts (for rank importance)
    final_pwms = fdr_cutoff.index
    pwm_name_to_index = dict(zip(pwm_names, range(len(pwm_names))))
    counts = [pos_array_summed[pwm_name_to_index[pwm_name]] for pwm_name in final_pwms]

    pwm_names_clean = [pwm_name.split("_")[0] for pwm_name in final_pwms]
    most_seen_df = pd.DataFrame(data=counts, index=pwm_names_clean)
    most_seen_file = "{}.most_seen.txt".format(out_prefix)
    most_seen_df.to_csv(most_seen_file, sep='\t', header=False)

    return fdr_cutoff_file


def make_motif_x_timepoint_mat(pwm_counts_mat_h5, key_list, task_indices_list, pwm_indices, pwm_names, prefix=""):
    """Make a motif x timepoints matrix and normalize the columns
    """
    # set up output matrix
    pwm_x_timepoint = np.zeros((len(pwm_indices), len(key_list))) 

    with h5py.File(pwm_counts_mat_h5, "r") as hf:
        # for each task (ie timepoint)
        for i in xrange(len(key_list)):
            key = key_list[i]
            print key
            task_idx = task_indices_list[i]
            print task_idx
            labels = hf['labels'][:, task_idx]
            pwm_scores = hf[key][:][labels > 0,:]
            pwm_scores_selected = pwm_scores[:, np.array(pwm_indices)]
            
            # sum up
            # convert to counts first?
            pwm_scores_summed = np.divide(np.sum(pwm_scores_selected, axis=0), pwm_scores_selected.shape[0])
            #pwm_scores_summed = np.sum((np.abs(pwm_scores_selected) > 0), axis=0)
            #pwm_scores_summed = np.mean(pwm_scores_selected, axis=0)
            
            # and save into matrix
            pwm_x_timepoint[:,i] = pwm_scores_summed
            
    # set up as pandas to save out
    columns = ["d0.0", "d0.5", "d1.0", "d1.5", "d2.0", "d2.5", "d3.0", "d4.5", "d5.0", "d6.0"]
    pwm_x_timepoint_df = pd.DataFrame(data=pwm_x_timepoint, index=pwm_names, columns=columns)
    pwm_x_timepoint_file = "{}.pwm_x_timepoint.mat.txt".format(prefix)
    pwm_x_timepoint_df.to_csv(pwm_x_timepoint_file, sep='\t')
    
    return pwm_x_timepoint_file



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
