"""Description: code to analyze motif matrices
"""

import h5py
import math

import numpy as np
import pandas as pd

import tensorflow as tf

#import scipy.stats
from scipy.stats import pearsonr
from scipy.signal import correlate2d
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from tronn.visualization import plot_weights


def read_pwm_file(pwm_file, value_type="log_likelihood", as_dict=False):
    """Extracts motifs into PWM class format
    """
    background_freq = 0.25
    
    # option to set up as dict or list
    if as_dict:
        pwms = {}
    else:
        pwms = []

    # open motif file and read
    with open(pwm_file) as fp:
        line = fp.readline().strip()
        while True:
            if line == '':
                break
            header = line.strip('>').strip()
            weights = []
            
            while True:
                line = fp.readline()
                if line == '' or line[0] == '>': break
                position_weights = map(float, line.split())
                
                if value_type == "log_likelihood":
                    # no need to change anything
                    weights.append(position_weights)
                elif value_type == "probability":
                    # convert to log likelihood
                    weights.append(
                        np.log2(np.array(position_weights) / background_freq).tolist())

            pwm = PWM(np.array(weights).transpose(1,0), header)

            # store into dict or list
            if as_dict:
                pwms[header] = pwm
            else:
                pwms.append(pwm)
                
    return pwms


class PWM(object):
    """PWM class for PWM operations
    """
    
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


    def pearson_xcor(self, pwm, use_probs=True, ic_thresh=0.4, ncor=False):
        """Calculate pearson across offsets, return best score
        and best position
        """
        # get total offset
        offset_total = self.weights.shape[1] + pwm.weights.shape[1] - 1
                
        # set up values
        max_cor_val = -1
        best_offset = 0

        for i in xrange(offset_total):

            # get padded weights
            self_padded_weights, other_padded_weights = self.pad_by_offset(pwm, i)

            # use merge and chomp to get the start and stop to chomp
            start_idx, stop_idx = self.merge(
                pwm, offset=i, chomp=False).chomp_points(ic_thresh=ic_thresh)
            if start_idx == stop_idx:
                continue
            
            #start_idx, stop_idx = PWM(np.maximum(self_padded_weights,other_padded_weights)).chomp_points(ic_thresh=ic_thresh)

            self_padded_weights_chomped = self_padded_weights[:,start_idx:stop_idx]
            other_padded_weights_chomped = other_padded_weights[:,start_idx:stop_idx]

            if use_probs:
                self_padded_weights_chomped = PWM(self_padded_weights_chomped).get_probs()
                other_padded_weights_chomped = PWM(other_padded_weights_chomped).get_probs()
            
            # take both and calculate
            # this is a pearson on the log scale, should it be with the probs?
            cor_val, pval = pearsonr(
                self_padded_weights_chomped.flatten(),
                other_padded_weights_chomped.flatten())

            # normalization (RSAT)
            if ncor:
                width_norm_val = (
                    self.weights.shape[1] + pwm.weights.shape[1] - self_padded_weights_chomped.shape[1]) / float(
                        self_padded_weights_chomped.shape[1])
                cor_val = cor_val * width_norm_val
                
            if cor_val > max_cor_val:
                max_cor_val = cor_val
                best_offset = i

        return max_cor_val, best_offset

    
    def rsat_cor(self, pwm, ncor=False, offset=None):
        """Calculate a pearson correlation across all positions
        """
        # tODO - dont really need this
        # return the pearson
        val, offset = self.pearson_xcor(pwm, ncor=ncor)
        
        return val

    
    def get_probs(self, count_factor=500, epsilon=0.01):
        """Take weights and convert to a PFM
        """
        #pseudo_counts = count_factor * np.exp(self.weights) + epsilon

        probs = np.exp(self.weights) / np.sum(np.exp(self.weights), axis=0)

        return probs

    
    def get_ic(self):
        """Get information content per each position per base pair
        """
        probs = self.get_probs()
        ic = 2 + np.sum(probs * np.log2(probs), axis=0)
        
        return ic
    
    
    def chomp_points(self, ic_thresh=0.4):
        """Remove leading/trailing Ns. In place, but also outputs self
        """
        ic = self.get_ic()
        
        # find starting point
        # iterate through positions unti you find the last
        # position before a high IC position
        start_idx = 0
        while start_idx < self.weights.shape[1]:
            # calculate IC of position
            if ic[start_idx] > ic_thresh:
                break
            start_idx += 1
        if start_idx == self.weights.shape[1]:
            start_idx = self.weights.shape[1]

        # find stop point
        stop_idx = self.weights.shape[1] - 1
        while stop_idx > 0:
            # calculate IC of position
            if ic[stop_idx] > ic_thresh:
                break
            stop_idx -= 1
        if stop_idx == 0:
            stop_idx = self.weights.shape[1]

        return start_idx, stop_idx + 1

    
    def chomp(self, ic_thresh=0.4):
        """Remove leading/trailing Ns. In place, but also outputs self
        """
        start_idx, stop_idx = self.chomp_points(ic_thresh=ic_thresh)

        # chomp
        self.weights = self.weights[:,start_idx:stop_idx+1]
        
        return self


    def pad_weights(self, start_pad, end_pad, in_place=False):
        """Pad weights with start_pad bp in the front
        and end_pad bp in the back
        """
        padded_weights = np.concatenate(
            (np.zeros((4, start_pad)),
             self.weights,
             np.zeros((4, end_pad))),
            axis=1)
        
        return padded_weights


    def pad_by_offset(self, pwm, offset, chomp=False):
        """Pads self and other pwm to be same length
        """
        total_length = self.weights.shape[1] + 2*(pwm.weights.shape[1] - 1) #-offset
        
        # self pwm
        front_pad = pwm.weights.shape[1] - 1
        end_pad = total_length - (front_pad + self.weights.shape[1])
        self_padded_weights = self.pad_weights(front_pad, end_pad)

        # other pwm
        front_pad = offset
        end_pad = total_length - (front_pad + pwm.weights.shape[1])
        other_padded_weights = pwm.pad_weights(front_pad, end_pad)

        return self_padded_weights, other_padded_weights

    
    def merge(
            self,
            pwm,
            offset,
            weights=(1.0, 1.0),
            ic_thresh=0.4,
            background_freq=0.25,
            new_name=None,
            chomp=True,
            prob_space=True,
            normalize=False):
        """Merge in another PWM and output a new PWM
        """
        self_padded_weights, other_padded_weights = self.pad_by_offset(pwm, offset)
        weight_sum = weights[0] + weights[1]
            
        if prob_space:
            self_padded_probs = np.exp(self_padded_weights) / np.sum(np.exp(self_padded_weights), axis=0)
            other_padded_probs = np.exp(other_padded_weights) / np.sum(np.exp(other_padded_weights), axis=0)
        
            # merge
            # merging by first moving back to prob space and then
            # returning to log space
            weighted_summed_probs = weights[0] * self_padded_probs + weights[1] * other_padded_probs
            new_pwm = PWM(
                np.log2(
                    weighted_summed_probs / (weight_sum * background_freq)),
                name=new_name)
        else:
            # do work in the log2 space
            weighted_summed_vals = weights[0] * self_padded_weights + weights[1] * other_padded_weights
            new_pwm = PWM(
                weighted_summed_vals / weight_sum,
                name=new_name)
            
        # chomp
        if chomp:
            new_pwm.chomp(ic_thresh=ic_thresh)

        # normalize if desired
        if normalize:
            new_pwm.normalize()

        #import ipdb
        #ipdb.set_trace()

        return new_pwm

    
    def to_motif_file(
            self,
            motif_file,
            motif_format="homer",
            pseudo_counts=500):
        """Write PWM out to file
        """
        # TODO allow various formats of output
        # such as transfac, homer, etc
        with open(motif_file, 'a') as fp:
            if motif_format == "homer":
                fp.write('>{}\n'.format(self.name))
                for i in range(self.weights.shape[1]):
                    vals = self.weights[:,i].tolist()
                    val_strings = [str(val) for val in vals]
                    fp.write('{}\n'.format('\t'.join(val_strings)))
            elif motif_format == "transfac":
                # TODO does not include consensus letter at the moment
                fp.write('ID {}\n'.format(self.name))
                fp.write('BF Homo_sapiens\n')
                fp.write('P0\tA\tC\tG\tT\n')
                for i in range(self.weights.shape[1]):
                    exp_vals = np.exp(self.weights[:,i])
                    vals = pseudo_counts * (exp_vals / np.sum(exp_vals))
                    val_strings = [str(val) for val in vals.tolist()]
                    fp.write("{num:02d}\t{}\n".format("\t".join(val_strings), num=i+1))
                fp.write("XX\n")
                fp.write("//\n")
        
        return None

    
    def plot(self, out_file, tmp_dir="."):
        """Plot out PWM to visualize
        """
        # save out in transfac format
        tmp_out_file = "{}/motif.{}.vals.transfac.tmp".format(
            tmp_dir, self.name.strip().split("_")[0])
        self.to_motif_file(tmp_out_file, motif_format="transfac")

        # and call weblogo
        weblogo_cmd = (
            "weblogo "
            "-X NO --errorbars NO --fineprint \"\" "
            "-C \"#CB2026\" A A "
            "-C \"#34459C\" C C "
            "-C \"#FBB116\" G G "
            "-C \"#0C8040\" T T "
            "-f {0} "
            "-D transfac "
            "-F pdf "
            "-o {1}").format(
                tmp_out_file, out_file)
        print weblogo_cmd
        os.system(weblogo_cmd)

        # and remove tmp file
        os.system("rm {}".format(tmp_out_file))
        
        return None



# set up for filtering pwm list (to use a subset of pwms)
def setup_pwms(master_pwm_file, pwm_subset_list_file):
    """setup which pwms are being used
    """
    # open the pwm subset file to get names of the pwms to use
    pwms_to_use = []
    with open(pwm_subset_list_file, "r") as fp:
        for line in fp:
            pwms_to_use.append(line.strip().split('\t')[0])        
            
    # then open the master file and filter out unused ones
    pwm_list = read_pwm_file(master_pwm_file)
    pwm_list_filt = []
    pwm_list_filt_indices = []
    for i in xrange(len(pwm_list)):
        pwm = pwm_list[i]
        for pwm_name in pwms_to_use:
            if pwm_name in pwm.name:
                pwm_list_filt.append(pwm)
                pwm_list_filt_indices.append(i)
    #print "Using PWMS:", [pwm.name for pwm in pwm_list_filt]
    print len(pwm_list_filt)
    pwm_names_filt = [pwm.name for pwm in pwm_list_filt]

    return pwm_list, pwm_list_filt, pwm_list_filt_indices, pwm_names_filt


# set up pwm metadata
def setup_pwm_metadata(metadata_file):
    """read in metadata to dicts for easy use
    """
    pwm_name_to_hgnc = {}
    hgnc_to_pwm_name = {}
    with open(metadata_file, "r") as fp:
        for line in fp:
            fields = line.strip().split("\t")
            try:
                pwm_name_to_hgnc[fields[0]] = fields[4]
                hgnc_to_pwm_name[fields[4]] = fields[0]
            except:
                pwm_name_to_hgnc[fields[0]] = fields[0].split(".")[0].split("_")[2]
                pwm_name_to_hgnc[fields[0]] = "UNK"

    return pwm_name_to_hgnc, hgnc_to_pwm_name



    



    
# TODO - delete this?
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
