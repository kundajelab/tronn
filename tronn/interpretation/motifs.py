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



    
def get_minimal_motifset(
        h5_file,
        dataset_keys,
        cluster_key,
        out_key,
        pwm_list,
        grammar_files=True,
        visualize=False):
    """Given clusters, get the subset and determine minimal motifset
    that explains the max of the cluster. Greedy algorithm.
    """
    with h5py.File(h5_file, "a"):
        num_examples = hf["example_metadata"].shape[0]
        num_motifs = hf[dataset_keys[0]].shape[1]

        # get metaclusters
        metacluster_by_region = hf[cluster_key][:,0]
        metaclusters = list(set(hf[cluster_key][:,0].tolist()))
        
        # set up out matrix {metacluster, task, M} <- save as motif vector
        metacluster_motifs_hf = hf.create_dataset(
            out_key, (len(metaclusters), len(dataset_keys), num_motifs))
        metacluster_motifs_hf.attrs["pwm_names"] = hf[dataset_keys[0]].attrs["pwm_names"]

        # for each metacluster, for each dataset, get matrix
        for i in xrange(len(metaclusters)):
            for j in xrange(len(dataset_keys)):

                metacluster_id = metaclusters[i]
                dataset_key = dataset_keys[j]
                
                sub_dataset = hf[dataset_key][:][
                    np.where(metacluster_by_region == metacluster_id),:]
        
                # try hagglom first, and then plot out - might see bimodal situation
                
                # after hagglom, determine a threshold and reduce
                motif_vector = None

            # then save out
            metacluster_motifs_hf[i, j, :] = motif_vector

            if grammar_files:
                # save out grammar files
                pass

            if visualize:
                # network plots?
                pass

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


