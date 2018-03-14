"""Description: code to analyze motif matrices
"""

import h5py
import math

import numpy as np
import pandas as pd

import tensorflow as tf

from multiprocessing import Pool

from scipy.stats import pearsonr
from scipy.signal import correlate2d
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from sklearn.metrics import precision_recall_curve

from tronn.visualization import plot_weights

from tronn.interpretation.clustering import get_distance_matrix
from tronn.interpretation.clustering import sd_cutoff




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




def make_threshold_at_fdr(fdr):
    """Construct function to get recall at FDR
    
    Args:
      fdr: FDR value for precision cutoff

    Returns:
      recall_at_fdr: Function to generate the recall at
        fdr fraction (number of positives
        correctly called as positives out of total 
        positives, at a certain FDR value)
    """
    def threshold_at_fdr(labels, probs):
        pr_curve = precision_recall_curve(labels, probs)
        precision, recall, thresholds = pr_curve

        threshold_index = np.searchsorted(precision - fdr, 0)
        #print "precision at thresh", precision[index]
        print "recall at thresh", recall[threshold_index]
        try:
            print "threshold val", thresholds[threshold_index]
            return thresholds[threshold_index]
        except:
            return 0 # TODO figure out what to do here...
        
    return threshold_at_fdr

    
def get_minimal_motifsets( # minimal feature sets?
        h5_file,
        dataset_keys,
        cluster_key,
        refined_cluster_key,
        out_key,
        pwm_list,
        pwm_dict,
        pwm_file=None,
        grammar_files=True,
        visualize=False):
    """Given clusters, get the subset and determine minimal motifset
    by reducing motif redundancy and motifs with low signal
    """
    from tronn.interpretation.grammars import Grammar
    
    with h5py.File(h5_file, "a") as hf:
        num_examples, num_motifs = hf[dataset_keys[0]].shape

        # set up a new cluster dataset, refined by the minimal motif set
        del hf[refined_cluster_key]
        refined_clusters_hf = hf.create_dataset(
            refined_cluster_key, hf[cluster_key].shape, dtype=int)

        # get metaclusters and ignore the non-clustered set (highest ID value)
        metacluster_by_region = hf[cluster_key][:,0]
        metaclusters = list(set(hf[cluster_key][:,0].tolist()))
        max_id = max(metaclusters)
        metaclusters.remove(max_id)

        # set up out matrix {metacluster, task, M} <- save as motif vector
        del hf[out_key]
        metacluster_motifs_hf = hf.create_dataset(
            out_key, (len(metaclusters), len(dataset_keys), num_motifs))
        metacluster_motifs_hf.attrs["pwm_names"] = hf[dataset_keys[0]].attrs["pwm_names"]
        # finish filling out attributes first, then append to dataset
        thresholds = np.zeros((
            len(metaclusters), len(dataset_keys), 1))
        
        # for each metacluster, for each dataset, get matrix
        for i in xrange(len(metaclusters)):
            metacluster_id = metaclusters[i]
            print"metacluster:", metacluster_id
            metacluster_thresholds = np.zeros((len(dataset_keys), 1))
            
            for j in xrange(len(dataset_keys)):
                print "task:", j
                dataset_key = dataset_keys[j]
                
                # get subset
                sub_dataset = hf[dataset_key][:][
                    np.where(metacluster_by_region == metacluster_id)[0],:]
                print "total examples used:", sub_dataset.shape

                # reduce pwms by signal similarity - a hierarchical clustering
                pwm_vector = reduce_pwms_by_signal_similarity(
                    sub_dataset, pwm_list, pwm_dict)

                # then set a cutoff - assume Gaussian noise, this controls
                # false positive rate
                pwm_vector = sd_cutoff(sub_dataset, pwm_vector)

                # ignore long pwms...
                current_indices = np.where(pwm_vector > 0)[0].tolist()
                for idx in current_indices:
                    if pwm_list[idx].weights.shape[1] > 15:
                        pwm_vector[idx] = 0
                
                print "final pwm count:", np.sum(pwm_vector)
                indices = np.where(pwm_vector > 0)[0].tolist()
                print [pwm_list[k].name for k in indices]

                # reduce the cluster size to those that have
                # all of the motifs in the pwm vector
                refined_clusters_hf[metacluster_by_region==metacluster_id, 0] = max_id
                masked_data = np.multiply(
                    hf[dataset_key][:],
                    np.expand_dims(pwm_vector, axis=0))
                minimal_motif_mask = np.sum(masked_data > 0, axis=1) >= np.sum(pwm_vector)
                metacluster_mask = metacluster_by_region == metacluster_id
                final_mask = np.multiply(minimal_motif_mask, metacluster_mask)
                refined_clusters_hf[final_mask > 0, 0] = metacluster_id

                # save out the pwm vector
                metacluster_motifs_hf[i, j, :] = pwm_vector

                # determine best threshold and save out to attribute, set at 5% FDR
                scores = np.sum(
                    np.multiply(
                        masked_data,
                        np.expand_dims(minimal_motif_mask, axis=1)),
                    axis=1)
                labels = metacluster_mask
                get_threshold = make_threshold_at_fdr(0.05)
                threshold = get_threshold(labels, scores)
                metacluster_thresholds[j, 0] = threshold
                
            # after finishing all tasks, save out to grammar file
            if grammar_files:
                grammar_file = "{0}.metacluster-{1}.motifset.grammar".format(
                    h5_file.split(".h5")[0], metacluster_id)
                for j in xrange(len(dataset_keys)):
                    pwm_vector = metacluster_motifs_hf[i,j,:]
                    pwm_names = np.array(metacluster_motifs_hf.attrs["pwm_names"])
                    pwm_names = pwm_names[np.where(pwm_vector)]
                    threshold = metacluster_thresholds[j,0]
                    print threshold
                    print pwm_names
                    node_dict = {}
                    for pwm in pwm_names:
                        node_dict[pwm] = 1.0
                    task_grammar = Grammar(
                        pwm_file,
                        node_dict,
                        {},
                        ("taskidx={0};"
                         "type=metacluster;"
                         "directed=no;"
                         "threshold={1}").format(j, threshold),
                        "metacluster-{0}.taskidx-{1}".format(
                            metacluster_id, j)) # TODO consider adjusting the taskidx here
                    task_grammar.to_file(grammar_file)
                    
            if visualize:
                # network plots?
                pass
                    
            # add thresholds    
            thresholds[i,:,:] = metacluster_thresholds

        # append
        metacluster_motifs_hf.attrs["thresholds"] = thresholds                    

    return None


def threshold_motifs(array, std_thresh=3):
    """Given a matrix, threshold out motifs (columns) that are low in signal
    """
    # opt 1 - just get top k
    # opt 2 - fit a normal distr and use standard dev cutoff
    # opt 3 - shuffled vals?

    # row normalize
    array_norm = np.divide(array, np.max(array, axis=1, keepdims=True))
    array_means = np.mean(array_norm, axis=0)
    
    # for now - across all vals, get a mean and standard dev
    mean_val = np.mean(array_means)
    std_val = np.std(array_means)

    # and threshold
    keep_indices = np.where(array_means > (mean_val + (std_val * std_thresh)))
    
    return keep_indices


def correlate_pwm_pair(input_list):
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



def correlate_pwms(
        pwms,
        cor_thresh=0.6,
        ncor_thresh=0.4,
        num_threads=24):
    """Correlate PWMS
    """
    # set up
    num_pwms = len(pwms)
    cor_mat = np.zeros((num_pwms, num_pwms))
    ncor_mat = np.zeros((num_pwms, num_pwms))

    pool = Pool(processes=num_threads)
    pool_inputs = []
    # for each pair of motifs, get correlation information
    for i in xrange(num_pwms):
        for j in xrange(num_pwms):

            # only calculate upper triangle
            if i > j:
                continue

            pwm_i = pwms[i]
            pwm_j = pwms[j]
            
            pool_inputs.append((i, j, pwm_i, pwm_j))

    # run multiprocessing
    pool_outputs = pool.map(correlate_pwm_pair, pool_inputs)

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



def hagglom_pwms(
        array,
        cor_mat_file,
        pwm_dict,
        ic_thresh=0.4,
        cor_thresh=0.8,
        ncor_thresh=0.65):
    """hAgglom on the PWMs to reduce redundancy
    """
    # read in table
    cor_df = pd.read_table(cor_mat_file, index_col=0)

    # set up pwm lists
    # set up (PWM, weight)
    hclust_pwms = [(pwm_dict[key], 1.0) for key in cor_df.columns.tolist()]
    non_redundant_pwms = []
    pwm_position = {}
    for i in xrange(len(hclust_pwms)):
        pwm, _ = hclust_pwms[i]
        pwm_position[pwm.name] = i

    # hierarchically cluster
    hclust = linkage(squareform(1 - cor_df.as_matrix()), method="ward")

    # keep a list of pwms in hclust, when things get merged add to end
    # (to match the scipy hclust structure)
    # put a none if not merging
    # if the motif did not successfully merge with its partner, pull out
    # it and its partner. if there was a successful merge, keep in there
    for i in xrange(hclust.shape[0]):
        idx1, idx2, dist, cluster_size = hclust[i,:]

        # check if indices are None
        pwm1, pwm1_weight = hclust_pwms[int(idx1)]
        pwm2, pwm2_weight = hclust_pwms[int(idx2)]

        if (pwm1 is None) and (pwm2 is None):
            hclust_pwms.append((None, None))
            continue
        elif (pwm1 is None):
            # save out PWM 2
            #print "saving out {}".format(pwm2.name)
            non_redundant_pwms.append(pwm2)
            hclust_pwms.append((None, None))
            continue
        elif (pwm2 is None):
            # save out PWM1
            #print "saving out {}".format(pwm1.name)
            non_redundant_pwms.append(pwm1)
            hclust_pwms.append((None, None))
            continue

        # try check
        try:
            cor_val, offset = pwm1.pearson_xcor(pwm2, ncor=False)
            ncor_val, offset = pwm1.pearson_xcor(pwm2, ncor=True)
        except:
            import ipdb
            ipdb.set_trace()

        if (cor_val > cor_thresh) and (ncor_val >= ncor_thresh):
            # if good match, now check the mat_df for which one
            # is most represented across sequences, and keep that one
            pwm1_presence = np.where(array[:,pwm_position[pwm1.name]] > 0)
            pwm2_presence = np.where(array[:,pwm_position[pwm2.name]] > 0)

            if pwm1_presence[0].shape[0] >= pwm2_presence[0].shape[0]:
                # keep pwm1
                #print "keep {} over {}".format(pwm1.name, pwm2.name)
                hclust_pwms.append((pwm1, 1.0))
            else:
                # keep pwm2
                #print "keep {} over {}".format(pwm2.name, pwm1.name)
                hclust_pwms.append((pwm2, 1.0))
        else:
            #print "saving out {}".format(pwm1.name)
            #print "saving out {}".format(pwm2.name)
            non_redundant_pwms.append(pwm1)
            non_redundant_pwms.append(pwm2)
            hclust_pwms.append((None, None))

    return non_redundant_pwms



def reduce_pwm_redundancy(
        pwms,
        pwm_dict,
        array,
        tmp_prefix="motifs",
        ic_thresh=0.4,
        cor_thresh=0.6,
        ncor_thresh=0.4,
        num_threads=28):
    """

    Note that RSAT stringent thresholds were ncor 0.65, cor 0.8
    Nonstringent is ncor 0.4 and cor 0.6
    """
    # trim pwms
    pwms = [pwm.chomp(ic_thresh=ic_thresh) for pwm in pwms]
    for key in pwm_dict.keys():
        pwm_dict[key] = pwm_dict[key].chomp(ic_thresh=ic_thresh)
    pwms_ids = [pwm.name for pwm in pwms]
    
    # correlate pwms - uses multiprocessing
    cor_mat_file = "{}.cor.motifs.mat.txt".format(tmp_prefix)
    ncor_mat_file = "{}.ncor.motifs.mat.txt".format(tmp_prefix)

    cor_filt_mat, ncor_filt_mat = correlate_pwms(
        pwms,
        cor_thresh=cor_thresh,
        ncor_thresh=ncor_thresh,
        num_threads=num_threads)
        
    # pandas and save out
    cor_df = pd.DataFrame(cor_filt_mat, index=pwms_ids, columns=pwms_ids)
    cor_df.to_csv(cor_mat_file, sep="\t")
    ncor_df = pd.DataFrame(ncor_filt_mat, index=pwms_ids, columns=pwms_ids)
    cor_df.to_csv(ncor_mat_file, sep="\t")

    # read in matrix to save time
    pwm_subset = hagglom_pwms(
        ncor_mat_file,
        pwm_dict,
        array,
        ic_thresh=ic_thresh,
        cor_thresh=cor_thresh,
        ncor_thresh=ncor_thresh)

    # once done, clean up
    os.system("rm {} {}".format(cor_mat_file, ncor_mat_file))

    return pwm_subset



def hagglom_pwms_by_signal(
        example_x_pwm_array,
        pwm_list,
        pwm_dict,
        cor_thresh=0.6,
        ncor_thresh=0.4):
    """hAgglom on the PWMs to reduce redundancy
    """
    # doing it by pwms
    if True:
        cor_filt_mat, distances = correlate_pwms(
            pwm_list,
            cor_thresh=cor_thresh,
            ncor_thresh=ncor_thresh,
            num_threads=24)
    
    # get the distance matrix on the example_x_pwm array
    # use continuous jaccard
    if False:
        distances, distance_pvals = get_distance_matrix(
            example_x_pwm_array, corr_method="continuous_jaccard")

    # set up pwm stuff
    # set up (PWM, weight)
    #hclust_pwms = [(pwm_dict[key], 1.0) for key in cor_df.columns.tolist()]
    hclust_pwms = [(pwm, 1.0) for pwm in pwm_list]
    pwm_mask = np.zeros((len(hclust_pwms)))
    pwm_position = {}
    for i in xrange(len(hclust_pwms)):
        pwm, _ = hclust_pwms[i]
        pwm_position[pwm.name] = i

    # hierarchically cluster
    hclust = linkage(squareform(1 - distances), method="ward")

    # keep a list of pwms in hclust, when things get merged add to end
    # (to match the scipy hclust structure)
    # put a none if not merging
    # if the motif did not successfully merge with its partner, pull out
    # it and its partner. if there was a successful merge, keep in there
    for i in xrange(hclust.shape[0]):
        idx1, idx2, dist, cluster_size = hclust[i,:]

        # check if indices are None
        pwm1, pwm1_weight = hclust_pwms[int(idx1)]
        pwm2, pwm2_weight = hclust_pwms[int(idx2)]

        if (pwm1 is None) and (pwm2 is None):
            hclust_pwms.append((None, None))
            continue
        elif (pwm1 is None):
            # mark PWM 2
            pwm_mask[pwm_position[pwm2.name]] = 1
            hclust_pwms.append((None, None))
            continue
        elif (pwm2 is None):
            # mark PWM 1
            pwm_mask[pwm_position[pwm1.name]] = 1
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

        if (cor_val > cor_thresh) and (ncor_val >= ncor_thresh):
            # if good match, now check the mat_df for which one
            # is most represented across sequences, and keep that one
            pwm1_signal = np.sum(example_x_pwm_array[:,pwm_position[pwm1.name]])
            pwm2_signal = np.sum(example_x_pwm_array[:,pwm_position[pwm2.name]])
            
            #pwm1_presence = np.where(example_x_pwm_array[:,pwm_position[pwm1.name]] > 0)[0].shape[0]
            #pwm2_presence = np.where(example_x_pwm_array[:,pwm_position[pwm2.name]] > 0)[0].shape[0]

            if pwm1_signal >= pwm2_signal:
                # keep pwm1 in the running
                hclust_pwms.append((pwm1, 1.0))
            else:
                # keep pwm2 in the running
                hclust_pwms.append((pwm2, 1.0))
        else:
            # mark out both
            pwm_mask[pwm_position[pwm1.name]] = 1
            pwm_mask[pwm_position[pwm2.name]] = 1
            hclust_pwms.append((None, None))

    return pwm_mask



def reduce_pwms_by_signal_similarity(
        example_x_pwm_array,
        pwms,
        pwm_dict,
        tmp_prefix="motifs",
        ic_thresh=0.4,
        cor_thresh=0.6,
        ncor_thresh=0.4,
        num_threads=24):
    """
    Takes in the example x pwm signal matrix, does an hclust on it.
    Then goes up the hclust, "merging" if pwms pass motif similarity
    The "merge" is actually just choosing the pwm with the higher
    signal.

    Note that RSAT stringent thresholds were ncor 0.65, cor 0.8
    Nonstringent is ncor 0.4 and cor 0.6
    """
    # trim pwms
    pwms = [pwm.chomp(ic_thresh=ic_thresh) for pwm in pwms]
    for key in pwm_dict.keys():
        pwm_dict[key] = pwm_dict[key].chomp(ic_thresh=ic_thresh)
    pwms_ids = [pwm.name for pwm in pwms]

    # choose motifs by hierarchical choice (hclust on signal)
    # and only merging if the motifs are similar. Only keep one
    # with strongest signal, doesn't merge the pwms.
    pwm_mask = hagglom_pwms_by_signal(
        example_x_pwm_array,
        pwms,
        pwm_dict,
        cor_thresh=cor_thresh,
        ncor_thresh=ncor_thresh)

    return pwm_mask








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


