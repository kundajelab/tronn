"""Description: code to analyze motif matrices
"""

import h5py
import math

import numpy as np
import pandas as pd

from multiprocessing import Pool

from scipy.stats import pearsonr
from scipy.signal import correlate2d
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

# TODO check if this needed?
from tronn.interpretation.clustering import get_distance_matrix

from tronn.stats.nonparametric import select_features_by_permutation_test

from tronn.util.utils import DataKeys
from tronn.util.h5_utils import AttrKeys

# TODO move to a pwm module
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


def select_pwms_by_permutation_test_and_reduce(
        array,
        pwm_list,
        hclust,
        num_shuffles=1000,
        pval_thresh=0.01):
    """use permutation test to get sig pwms
    and then reduce by similarity
    """
    # select features
    sig_pwms = select_features_by_permutation_test(
        array,
        num_shuffles=num_shuffles,
        pval_thresh=pval_thresh)

    # and then reduce
    sig_pwms = np.multiply(sig_pwms, reduce_pwms(array, hclust, pwm_list))

    # debug
    print "final pwm count:", np.sum(sig_pwms)
    indices = np.where(sig_pwms > 0)[0].tolist()
    print [pwm_list[k].name for k in indices]
    
    return sig_pwms


# TODO move to... stats?
def aggregate_array(
        array,
        agg_fn=np.median,
        agg_axis=0,
        mask=None):
    """aggregate dataset
    """
    agg_array = agg_fn(array, axis=agg_axis) # {task, M}
    agg_array = agg_array[:,mask>0]
    
    return agg_array


def select_task_pwms(array, pwm_list, hclust):
    """wrapper to make it easy to work with task dimension
    """
    assert len(array.shape) == 3

    sig_pwms = np.zeros((array.shape[2]))
    for task_idx in xrange(array.shape[1]):
        task_data = array[:,task_idx,:]
        sig_pwms += select_pwms_by_permutation_test_and_reduce(
            task_data, pwm_list, hclust)
        
    return sig_pwms


def refine_sig_pwm_clusters(clusters):
    """adjust clusters
    """
    # TODO if needed

    

    return


def extract_significant_pwms(
        h5_file,
        pwm_list,
        data_key=DataKeys.FEATURES,
        cluster_key=DataKeys.CLUST_FILT,
        pwm_names_attr_key=AttrKeys.PWM_NAMES,
        pwm_sig_global_key=DataKeys.PWM_SIG_GLOBAL,
        pwm_scores_agg_global_key=DataKeys.PWM_SCORES_AGG_GLOBAL,
        pwm_sig_clusters_key=DataKeys.PWM_SIG_CLUST,
        pwm_sig_clusters_all_key=DataKeys.PWM_SIG_CLUST_ALL,
        pwm_scores_agg_clusters_key=DataKeys.PWM_SCORES_AGG_CLUST,
        refine_clusters=False):
    """
    """
    # get a hierarchical clustering on pwms by sequence distance
    cor_filt_mat, distances = correlate_pwms(
        pwm_list, cor_thresh=0.3, ncor_thresh=0.2, num_threads=24)
    hclust = linkage(squareform(1 - distances), method="ward")

    # outputs
    outputs = {}
    
    # pull out data
    with h5py.File(h5_file, "r") as hf:
        data = hf[data_key][:] # {N, task, M}
        pwm_names = hf[data_key].attrs[pwm_names_attr_key]
        
    # (1) get global sig (and aggregate results)
    # TODO when stable use multiprocessing Pool
    sig_pwms = select_task_pwms(data, pwm_list, hclust)
    agg_data = aggregate_array(data, mask=sig_pwms)

    # save out
    pwm_sig_global_names = pwm_names[sig_pwms > 0]
    outputs[pwm_sig_global_key] = (sig_pwms, pwm_sig_global_names)
    outputs[pwm_scores_agg_global_key] = (agg_data, pwm_sig_global_names)
    print "globally significant:", pwm_sig_global_names
    
    # (2) get cluster sig (and aggregate results)
    with h5py.File(h5_file, "r") as hf:
        clusters = hf[cluster_key][:]

    if len(clusters.shape) == 1:
        # hard clusters
        cluster_ids = sorted(list(set(clusters.tolist())))
        if -1 in cluster_ids: cluster_ids.remove(-1)
        hard_clusters = True
    else:
        # soft clusters
        cluster_ids = range(clusters.shape[1])
        hard_clusters = False

    # go through clusters to get sig pwms
    sig_pwms = np.zeros((len(cluster_ids), data.shape[2]))
    for cluster_idx in xrange(len(cluster_ids)):
        print cluster_idx,
        cluster_id = cluster_ids[cluster_idx]

        # get which examples are in cluster
        if hard_clusters:
            in_cluster = clusters == cluster_id
        else:
            in_cluster = clusters[:,cluster_id] == 1
        print "num examples: {}".format(np.sum(in_cluster))
            
        # select
        cluster_data = data[np.where(in_cluster)[0],:,:] # {N, task, M}
        cluster_pwms = np.zeros((data.shape[2]))

        # get sig and aggregate results
        sig_pwms[cluster_idx,:] = select_task_pwms(cluster_data, pwm_list, hclust)
        pwm_sig_cluster_names = pwm_names[sig_pwms[cluster_idx,:] > 0]
        print pwm_sig_cluster_names
    
    # adjust clustering as needed
    if refine_clusters:
        pass

    # save out
    outputs[pwm_sig_clusters_all_key] = np.any(sig_pwms > 0, axis=0) # {M}
    pwm_sig_cluster_global_names = pwm_names[outputs[pwm_sig_clusters_all_key] > 0]
    print pwm_sig_cluster_global_names
    
    # and aggregate {cluster, task, M}
    agg_data = np.zeros((
        len(cluster_ids),
        data.shape[1],
        np.sum(outputs[pwm_sig_clusters_all_key])))
    for cluster_idx in xrange(len(cluster_ids)):
        print cluster_idx
        cluster_id = cluster_ids[cluster_idx]

        # get which examples are in cluster
        if hard_clusters:
            in_cluster = clusters == cluster_id
        else:
            in_cluster = clusters[:,cluster_id] == 1

        cluster_data = data[np.where(in_cluster)[0],:,:]
        agg_data[cluster_idx,:,:] = aggregate_array(
            cluster_data, mask=outputs[pwm_sig_clusters_all_key])
    
    # save out
    outputs[pwm_sig_clusters_key] = (sig_pwms, pwm_sig_cluster_global_names)
    outputs[pwm_scores_agg_clusters_key] = (agg_data, pwm_sig_cluster_global_names)

    import ipdb
    ipdb.set_trace()
    
    # and then save all of this out
    with h5py.File(h5_file, "a") as out:
        for key in outputs.keys():
            out.create_dataset(key, data=outputs[key][0])
            out[key].attrs[AttrKeys.PWM_NAMES] = outputs[key][1]
    
    return None


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



def hagglom_pwms_by_signal_old(
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
            # TODO - this bit should only be summed once
            # ideally push in a 1D "signal strength" vector which is the choice between the two.
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


def hagglom_pwms_by_signal(
        hclust,
        signal_vector,
        pwm_list,
        cor_thresh=0.6,
        ncor_thresh=0.4):
    """agglomerate the pwms
    hclust is a linkage object from scipy on the pwms
    signal vector is a 1d vector of scores for each pwm
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

        # if there's no signal, ignore and save time
        # CHECK - not sure if this works
        #if signal_vector[pwm_position[pwm1.name]] == 0:
        #    pwm1 = None
        #if signal_vector[pwm_position[pwm2.name]] == 0:
        #    pwm2 = None
        
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



def reduce_pwms(data, hclust, pwm_list, std_thresh=3):
    """Wrapper for all pwm reduction functions
    """
    assert len(pwm_list) == data.shape[1]
    
    # hagglom
    signal_vector = np.median(data, axis=0)
    #pwm_vector = pwm_vector * hagglom_pwms_by_signal(
    pwm_vector = hagglom_pwms_by_signal(
        hclust,
        signal_vector,
        pwm_list,
        cor_thresh=0.3,
        ncor_thresh=0.2)
    
    # ignore long pwms
    if True:
        current_indices = np.where(pwm_vector > 0)[0].tolist()
        for idx in current_indices:
            if pwm_list[idx].weights.shape[1] > 15:
                pwm_vector[idx] = 0

    # debug
    if False:
        print "final pwm count:", np.sum(pwm_vector)
        indices = np.where(pwm_vector > 0)[0].tolist()
        print [pwm_list[k].name for k in indices]
        pwm_to_index = {}
        for k in indices:
            pwm_to_index[pwm_list[k].name] = k
    
    return pwm_vector


def reduce_pwms_by_signal_similarity(
        example_x_pwm_array,
        pwms,
        pwm_dict,
        tmp_prefix="motifs",
        ic_thresh=0.4,
        cor_thresh=0.3, # low thresh
        ncor_thresh=0.2, # low thresh
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

    # TODO get the pwm x pwm distance matrix here (just run once)

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
