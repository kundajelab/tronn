# description: utils for managing position weight matrices (PWMs)


import math

import numpy as np

from multiprocessing import Pool
from numpy.random import RandomState

from scipy.stats import pearsonr
from scipy.signal import correlate2d
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


class PWM(object):
    """PWM class for PWM operations
    """
    
    def __init__(self, weights, name=None, threshold=None):
        self.weights = weights
        self.name = name
        self.threshold = threshold
        self.idx_to_string = {
            0: "A", 1: "C", 2: "G", 3: "T"}

        
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

    
    def get_consensus_string(self):
        """get argmax consensus string
        """
        sequence = []
        for idx in range(self.weights.shape[1]):
            sampled_idx = np.argmax(self.weights[:,idx])
            sequence.append(self.idx_to_string[sampled_idx])
            
        return "".join(sequence)

    
    def get_sampled_string(self, rand_seed=1):
        """get a sampled string from weights
        """
        sequence = []
        probs = self.get_probs()
        rand_state = RandomState(rand_seed)
        for idx in range(self.weights.shape[1]):
            sampled_idx = rand_state.choice([0,1,2,3], p=probs[:,idx])
            sequence.append(self.idx_to_string[sampled_idx])
            
        return "".join(sequence)
    
    
    def reverse_complement(self, new_name=None):
        """Produce a new PWM that is the reverse complement
        Assumes the bp order is ACGT - this means that you
        can just flip in both directions to get reverse
        complement
        """
        new_weights = np.flip(np.flip(self.weights, 0), 1)
        if new_name is None:
            new_name = "{}.RC".format(self.name)

        new_pwm = PWM(new_weights, name=new_name, threshold=self.threshold)

        return new_pwm
    
    
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


def _pool_correlate_pwm_pair(input_list):
    """get cor and ncor for pwm1 and pwm2
    Set up this way because multiprocessing pool only takes 1
    input
    """
    i = input_list[0]
    j = input_list[1]
    pwm1 = input_list[2]
    pwm2 = input_list[3]
    
    motif_cor = pwm1.rsat_cor(pwm2)
    motif_ncor = pwm1.rsat_cor(pwm2, ncor=True)

    return i, j, motif_cor, motif_ncor



class MotifSetManager(object):
    """manager for a pwm set"""

    def __init__(self):
        self.pwms = []
        self.pwm_dict= {}

        
    @staticmethod
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

    
    @staticmethod
    def correlate_pwms_by_seq_similarity(
            pwm_list,
            cor_thresh=0.6,
            ncor_thresh=0.4,
            num_threads=24):
        """cross correlate the pwms to get 
        distances from pwm to pwm
        """
        # set up correlation arrays
        num_pwms = len(pwm_list)
        cor_mat = np.zeros((num_pwms, num_pwms))
        ncor_mat = np.zeros((num_pwms, num_pwms))

        # set up multiprocessing
        pool = Pool(processes=num_threads)
        pool_inputs = []

        # for each pair of motifs, get correlation information
        for i in xrange(num_pwms):
            for j in xrange(num_pwms):
            
                # only calculate upper triangle
                if i > j:
                    continue

                pwm_i = pwm_list[i]
                pwm_j = pwm_list[j]
            
                pool_inputs.append((i, j, pwm_i, pwm_j))

        # run multiprocessing
        pool_outputs = pool.map(
            _pool_correlate_pwm_pair, pool_inputs)
        
        for i, j, motif_cor, motif_ncor in pool_outputs:
            # if passes cutoffs, save out to matrix
            if (motif_cor >= cor_thresh) and (motif_ncor >= ncor_thresh):
                cor_mat[i,j] = motif_cor
                ncor_mat[i,j] = motif_ncor        

        # and reflect over the triangle
        lower_triangle_indices = np.tril_indices(cor_mat.shape[0], -1)
        cor_mat[lower_triangle_indices] = cor_mat.T[lower_triangle_indices]
        ncor_mat[lower_triangle_indices] = ncor_mat.T[lower_triangle_indices]
        
        # multiply each by the other to double threshold
        cor_present = (cor_mat > 0).astype(float)
        ncor_present = (ncor_mat > 0).astype(float)

        # and mask
        cor_filt_mat = cor_mat * ncor_present
        ncor_filt_mat = ncor_mat * cor_present

        # fill diagonal
        np.fill_diagonal(cor_filt_mat, 1)
        np.fill_diagonal(ncor_filt_mat, 1)
    
        return cor_filt_mat, ncor_filt_mat

    
    @staticmethod
    def hclust(distances):
        """cluster based on distance array 
        (get back an hclust object)
        """
        return linkage(squareform(1 - distances), method="ward")
    
    
    @staticmethod
    def hagglom_pwms_by_signal(
            pwm_list,
            hclust,
            signal_vector,
            cor_thresh=0.6,
            ncor_thresh=0.4):
        """agglomerate the pwms
        hclust is a linkage object from scipy on the pwms
        signal vector is a 1d vector of scores for each pwm
        when merging, keep the pwm with the higher SCORE
        """
        # set up lists and vectors
        hclust_pwms = [(pwm, 1.0) for pwm in pwm_list]
        pwm_vector = np.zeros((len(hclust_pwms)))
        pwm_position = {}
        for i in xrange(len(pwm_list)):
            pwm_position[pwm_list[i].name] = i

        # go through pwms. merge if cor and ncor are above thresh,
        # and keep the pwm with better signal.
        for i in xrange(hclust.shape[0]):
            idx1, idx2, dist, cluster_size = hclust[i,:]
            pwm1, pwm1_weight = hclust_pwms[int(idx1)]
            pwm2, pwm2_weight = hclust_pwms[int(idx2)]

            # check if pwms are None
            if (pwm1 is None) and (pwm2 is None):
                hclust_pwms.append((None, None))
                continue
            elif (pwm1 is None):
                # mark PWM 2
                pwm_vector[pwm_position[pwm2.name]] = 1
                hclust_pwms.append((None, None))
                continue
            elif (pwm2 is None):
                # mark PWM 1
                pwm_vector[pwm_position[pwm1.name]] = 1
                hclust_pwms.append((None, None))
                continue

            # try check
            try:
                cor_val, offset = pwm1.pearson_xcor(pwm2, ncor=False)
                ncor_val, offset = pwm1.pearson_xcor(pwm2, ncor=True)
            except:
                print "something unexpected happened"
                import ipdb
                ipdb.set_trace()

            # if good match, now check the signal vector for which is stronger and keep that one
            if (cor_val > cor_thresh) and (ncor_val >= ncor_thresh):
                if signal_vector[pwm_position[pwm1.name]] >= signal_vector[pwm_position[pwm2.name]]:
                    hclust_pwms.append((pwm1, 1.0))
                else:
                    hclust_pwms.append((pwm2, 1.0))
            else:
                # mark out both
                pwm_vector[pwm_position[pwm1.name]] = 1
                pwm_vector[pwm_position[pwm2.name]] = 1
                hclust_pwms.append((None, None))

        return pwm_vector


    @staticmethod
    def reduce_pwms(
            pwm_list,
            hclust,
            signal_vector,
            remove_long=True):
        """reduce redundancy in the pwms
        current preferred method is hagglom with signal vector
        """
        assert len(pwm_list) == signal_vector.shape[0]

        # hagglom
        keep_pwms = MotifSetManager.hagglom_pwms_by_signal(
            pwm_list,
            hclust,
            signal_vector,
            cor_thresh=0.3,
            ncor_thresh=0.2)

        # ignore long pwms
        if remove_long:
            current_indices = np.where(keep_pwms > 0)[0].tolist()
            for idx in current_indices:
                if pwm_list[idx].weights.shape[1] > 15:
                    keep_pwms[idx] = 0
                    
        return keep_pwms

    
    def add_pwms(self, pwm_file):
        """add pwms to the set
        """
        self.pwms.append(self.read_pwm_file(pwm_file))
        self.pwms.update(self.read_pwm_file(pwm_file, as_dict=True))

