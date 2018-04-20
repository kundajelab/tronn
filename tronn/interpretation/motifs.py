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
from sklearn.metrics import roc_curve

from tronn.visualization import plot_weights

from tronn.interpretation.clustering import get_distance_matrix
from tronn.interpretation.clustering import sd_cutoff
from tronn.interpretation.clustering import get_threshold_on_jaccard_similarity
from tronn.interpretation.clustering import get_threshold_on_euclidean_distance
from tronn.interpretation.clustering import get_threshold_on_dot_product
from tronn.interpretation.learning import build_polynomial_model

from tronn.interpretation.learning import build_regression_model
from tronn.interpretation.learning import threshold_at_recall

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures



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
        #print precision
        threshold_index = np.searchsorted(precision, fdr)
        #threshold_index = np.searchsorted(1 - recall, 1 - fdr)
        #print "precision at thresh", precision[index]
        print "recall at thresh", recall[threshold_index]
        try:
            #print thresholds
            #print threshold_index
            print "threshold val", thresholds[threshold_index]
            return thresholds[threshold_index]
        except:
            return 0 # TODO figure out what to do here...
        
    return threshold_at_fdr

 
def make_threshold_at_tpr(desired_tpr):
    """Construct function to get recall at FDR
    
    Args:
      fdr: FDR value for precision cutoff

    Returns:
      recall_at_fdr: Function to generate the recall at
        fdr fraction (number of positives
        correctly called as positives out of total 
        positives, at a certain FDR value)
    """
    def threshold_at_tpr(labels, probs):
        fpr, tpr, thresholds = roc_curve(labels, probs)
        print tpr
        threshold_index = np.searchsorted(tpr, desired_tpr)
        #threshold_index = np.searchsorted(1 - recall, 1 - fdr)
        #print "precision at thresh", precision[index]
        print "fpr at thresh", fpr[threshold_index]
        try:
            print thresholds
            print threshold_index
            print "threshold val", thresholds[threshold_index]
            return thresholds[threshold_index]
        except:
            return 0 # TODO figure out what to do here...
        
    return threshold_at_tpr


def reduce_pwms(data, pwm_list, pwm_dict):
    """Wrapper for all pwm reduction functions
    """
    assert len(pwm_list) == data.shape[1]
    
    # set a cutoff - assume Gaussian noise, this controls
    # false positive rate
    #pwm_vector = sd_cutoff(data, col_mask=pwm_vector)
    pwm_vector = sd_cutoff(data, std_thresh=3) # was 2
    
    # reduce by motif similarity, keeping the motif
    # with the strongest signal
    pwm_vector = pwm_vector * reduce_pwms_by_signal_similarity(
        data, pwm_list, pwm_dict)

    
    # ignore long pwms
    if True:
        current_indices = np.where(pwm_vector > 0)[0].tolist()
        for idx in current_indices:
            if pwm_list[idx].weights.shape[1] > 15:
                pwm_vector[idx] = 0

    # debug
    print "final pwm count:", np.sum(pwm_vector)
    indices = np.where(pwm_vector > 0)[0].tolist()
    print [pwm_list[k].name for k in indices]
    pwm_to_index = {}
    for k in indices:
        pwm_to_index[pwm_list[k].name] = k
    
    return pwm_vector


def distill_to_linear_models(
        h5_file,
        dataset_keys,
        cluster_key,
        refined_cluster_key,
        out_key,
        pwm_list,
        pwm_dict,
        pwm_file=None,
        grammar_files=True,
        visualize=False,
        label_indices=[]):
    """Given clusters, get the subset and determine minimal motifset
    by reducing motif redundancy and motifs with low signal
    """
    assert len(label_indices) > 0
    from tronn.interpretation.grammars import Grammar
    
    with h5py.File(h5_file, "a") as hf:
        num_examples, num_motifs = hf[dataset_keys[0]].shape
        raw_scores = hf["raw-pwm-scores"][:] # pull raw sequence

        # set up a new cluster dataset, refined by the thresholded scores
        del hf[refined_cluster_key]
        refined_clusters_hf = hf.create_dataset(
            refined_cluster_key, hf[cluster_key].shape, dtype=int)

        # get metaclusters and ignore the non-clustered set (highest ID value)
        metacluster_by_region = hf[cluster_key][:,0]
        metaclusters = list(set(hf[cluster_key][:,0].tolist()))
        max_id = max(metaclusters)
        metaclusters.remove(max_id)
        print metaclusters
        
        # for each metacluster, for each dataset, get matrix
        for i in xrange(len(metaclusters)):
        #for i in xrange(1, len(metaclusters)):
            metacluster_id = metaclusters[i]
            print"metacluster:", metacluster_id

            if grammar_files:
                # set up grammar file name
                grammar_file = "{0}.metacluster-{1}.motifset.grammar".format(
                    h5_file.split(".h5")[0], metacluster_id)

            for j in xrange(len(dataset_keys)):
                print "task:", j
                dataset_key = dataset_keys[j]
                label_index = label_indices[j]
                print "label index:", label_index # eventually change this
                
                # get subsets
                cluster_labels = metacluster_by_region == metacluster_id
                cluster_weighted_scores = hf[dataset_key][:][
                    np.where(cluster_labels)[0],:]
                cluster_raw_scores = raw_scores[
                    np.where(cluster_labels)[0],:]
                print "total examples used:", cluster_weighted_scores.shape
                
                # row normalize
                max_vals = np.max(cluster_weighted_scores, axis=1, keepdims=True)
                cluster_weighted_scores = np.divide(
                    cluster_weighted_scores,
                    max_vals,
                    out=np.zeros_like(cluster_weighted_scores),
                    where=max_vals!=0)
                cluster_weighted_scores = cluster_weighted_scores[np.max(cluster_weighted_scores, axis=1) >0]
                print "after remove zeroes:", cluster_weighted_scores.shape
                
                # keep mean vectors and weights
                mean_score_vector = np.mean(cluster_raw_scores, axis=0)
                mean_weighted_score_vector = np.mean(cluster_weighted_scores, axis=0)
                mean_weights = np.divide(mean_weighted_score_vector, mean_score_vector)
                
                # weighted according to the weights from the cluster
                weighted_raw_scores = np.multiply(
                    raw_scores,
                    np.expand_dims(mean_weights, axis=0))

                # get threshold on distance metric
                similarity_threshold, threshold_filter = get_threshold_on_dot_product(
                    mean_weighted_score_vector,
                    weighted_raw_scores,
                    cluster_labels,
                    recall_thresh=0.68) # 0.4?
                print np.sum(threshold_filter)
                passing_filter = threshold_filter # passing filter is only on jaccard and pwm thresholds
                
                # filter useful pwms
                pwm_vector = reduce_pwms(cluster_weighted_scores, pwm_list, pwm_dict)
                if np.sum(pwm_vector) == 0:
                    continue

                # adjust pwm thresholds
                recall_thresh = 0.95
                indices = np.where(pwm_vector > 0)[0].tolist()
                pwm_thresholds = {}
                for pwm_idx in indices:
                    threshold = threshold_at_recall(
                        cluster_labels, raw_scores[:,pwm_idx], recall_thresh=recall_thresh)
                    pwm_thresholds[pwm_list[pwm_idx].name] = threshold
                    passing_filter = np.multiply(
                        passing_filter, raw_scores[:,pwm_idx] > threshold)
                    
                print np.sum(passing_filter)

                print "final pwm count:", np.sum(pwm_vector)
                print [pwm_list[k].name for k in indices]
                #import ipdb
                #ipdb.set_trace()

                # build polynomial (deg 2) model on the subset
                X = raw_scores[:, pwm_vector > 0]
                X = np.multiply(X, np.expand_dims(passing_filter, axis=1)) # pass features through filters
                y = np.multiply(passing_filter, hf["logits"][:,label_index])
                #y = np.multiply(passing_filter, hf["probs"][:,label_index])
                #y = np.multiply(passing_filter, hf["labels"][:,label_index])
                
                feature_names = [pwm_list[pwm_idx].name for pwm_idx in indices]
                poly = PolynomialFeatures(
                    2, interaction_only=False, include_bias=False)
                X = poly.fit_transform(X)
                poly_names = poly.get_feature_names(feature_names)
                
                #X_learn = X[passing_filter > 0]
                #y_learn = y[passing_filter > 0]
                final_set = np.multiply(cluster_labels, passing_filter)
                X_learn = X[final_set > 0] # only learn on the things in the cluster that pass filter
                y_learn = y[final_set > 0]
                print X_learn.shape
                clf = build_regression_model(X_learn, y_learn)

                # back check and get threshold
                fdr_thresh = 0.75 # opposite: so if want FDR 0.05, put in 0.95
                scores = clf.predict(X)
                from tronn.run_evaluate import auprc
                # how well do we do at getting back the cluster
                print "AUPRC for cluster:", auprc(cluster_labels, scores)

                #from scipy.stats import describe
                #print describe(scores)
                
                # threshold and get threshold (remove intercept so don't need to track it)
                get_threshold = make_threshold_at_fdr(fdr_thresh)
                model_threshold = get_threshold(cluster_labels, scores)

                # allow the null labels to be included
                allowed_labels = np.logical_or(cluster_labels, metacluster_by_region == max_id)
                #model_threshold = get_threshold(cluster_labels, scores)
                model_threshold = get_threshold(allowed_labels, scores)
                model_threshold_adj = model_threshold - clf.intercept_
                print "model threshold", model_threshold
                print "fraction passing threshold", np.sum(scores > model_threshold)

                # save out things
                # save out weights into grammar file
                motifspace_dict = {}
                motifspace_param_string = "measure=dot_product;threshold={}".format(
                    similarity_threshold)
                # keep the mean vector and weighting info
                for pwm_idx in xrange(mean_score_vector.shape[0]):
                    motifspace_dict[pwm_list[pwm_idx].name] = (
                        mean_weighted_score_vector[pwm_idx], mean_weights[pwm_idx])

                    
                node_dict = {}
                edge_dict = {}
                for coef_idx in xrange(len(poly_names)):

                    # TODO - still need to save out!
                    if clf.coef_[coef_idx] == 0:
                        nodes = poly_names[coef_idx].split()
                        if len(nodes) == 1:
                            if "^2" not in nodes[0]:
                                # single - save to node dict
                                node_dict[nodes[0]] = (pwm_thresholds[nodes[0]], clf.coef_[coef_idx])
                        continue

                    # get name and split
                    nodes = poly_names[coef_idx].split()

                    # check for squared nodes
                    if len(nodes) == 1:
                        if "^2" not in nodes[0]:
                            # single - save to node dict
                            node_dict[nodes[0]] = (pwm_thresholds[nodes[0]], clf.coef_[coef_idx])
                        else:
                            # squared term - save to edge_dict
                            node_name = nodes[0].split("^")[0]
                            edge_dict[(node_name, node_name)] =  clf.coef_[coef_idx]
                    elif len(nodes) == 2:
                        # interaction term - save to edge_dict
                        edge_dict[(nodes[0], nodes[1])] =  clf.coef_[coef_idx]
                    else:
                        raise Exception("Higher term polynomial not implemented yet")

                # TODO given the model, get the AUPRC, AUROC, and recall at FDR
                # for each label set
                # TODO separate this out into a function in a metrics package
                num_labels = hf["labels"].shape[1]
                auroc_by_label = np.zeros((num_labels))
                auprc_by_label = np.zeros((num_labels))
                recall_by_label = np.zeros((num_labels))
                accuracy_by_label = np.zeros((num_labels))
                for label_idx in xrange(num_labels):
                    label_set = hf["labels"][:, label_idx]

                    from tronn.run_evaluate import auprc
                    from tronn.run_evaluate import make_recall_at_fdr
                    from sklearn.metrics import roc_auc_score
                    # calculate metrics
                    try:
                        auprc_by_label[label_idx] = roc_auc_score(label_set, scores)
                    except:
                        auprc_by_label[label_idx] = 0
                    try:
                        auprc_by_label[label_idx] = auprc(label_set, scores)
                    except:
                        auprc_by_label[label_idx] = 0
                    try:
                        recall_at_fdr = make_recall_at_fdr(0.05)
                        recall_by_label[label_idx] = recall_at_fdr(label_set, scores)
                    except:
                        recall_by_label[label_idx] = 0

                    accuracy_by_label[label_idx] = np.sum(np.multiply(
                        label_set, scores > model_threshold)) / float(np.sum(label_set))
                    #accuracy_by_label[label_idx] = np.sum(np.multiply(
                    #    label_set, final_mask)) / np.sum(final_mask)
                    #accuracy_by_label[label_idx] = np.sum(np.multiply(
                    #    label_set, scores)) / np.sum(label_set)
                print accuracy_by_label[13:24]
                print auprc_by_label[13:24]

                task_grammar = Grammar(
                    pwm_list,
                    node_dict,
                    edge_dict,
                    ("taskidx={0};"
                     "type=elastic_net;"
                     "directed=no;"
                     "threshold={1}").format(j, model_threshold_adj),
                    motifspace_dict=motifspace_dict,
                    motifspace_param_string=motifspace_param_string,
                    name="metacluster-{0}.taskidx-{1}".format(
                        metacluster_id, j)) # TODO consider adjusting the taskidx here
                task_grammar.to_file(grammar_file)

                # save out as a table
                test = pd.DataFrame()
                test["nn_logits"] = y_learn
                test["linear_predictions"] = clf.predict(X_learn)

                test.to_csv("testing.txt", sep="\t")

            if visualize:
                # network plots?

                # heatmap of the weights on the model?
                pass

    return None

    
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
        visualize=False,
        label_indices=[]):
    """Given clusters, get the subset and determine minimal motifset
    by reducing motif redundancy and motifs with low signal
    """
    assert len(label_indices) > 0
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

            if grammar_files:
                grammar_file = "{0}.metacluster-{1}.motifset.grammar".format(
                    h5_file.split(".h5")[0], metacluster_id)

            for j in xrange(len(dataset_keys)):
                print "task:", j
                dataset_key = dataset_keys[j]
                label_index = label_indices[j]
                print "label index:", label_index
                
                # get subset
                sub_dataset = hf[dataset_key][:][
                    np.where(metacluster_by_region == metacluster_id)[0],:]
                print "total examples used:", sub_dataset.shape

                # row normalize
                sub_dataset = np.divide(
                    sub_dataset, np.max(sub_dataset, axis=1, keepdims=True))
                
                # reduce pwms by signal similarity - a hierarchical clustering
                pwm_vector = np.ones((len(pwm_list)))
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
                pwm_to_index = {}
                for k in indices:
                    pwm_to_index[pwm_list[k].name] = k

                
                # then on that set, run modeling

                    
                # reduce the cluster size to those that have
                # all of the motifs in the pwm vector
                refined_clusters_hf[metacluster_by_region==metacluster_id, 0] = max_id
                data_norm = hf["pwm-scores-raw"][:] # do not adjust, these are as if raw sequence
                masked_data = np.multiply(
                    data_norm,
                    np.expand_dims(pwm_vector, axis=0)) # the mask
                # TODO - problem here, the minimal motif mask doesn't do anything
                # or very little
                minimal_motif_mask = np.sum(masked_data > 0, axis=1) >= np.sum(pwm_vector)
                metacluster_mask = metacluster_by_region == metacluster_id
                final_mask = np.multiply(minimal_motif_mask, metacluster_mask) # the label set
                refined_clusters_hf[final_mask > 0, 0] = metacluster_id

                # TODO first thing may be to adjust thresholds on PWMs
                # want a pwm score that has perfect recall (gets all instances back)
                # but no higher - so highest threshold with perfect recall
                # refine the minimal motif mask
                for k in indices:
                    threshold = threshold_at_perfect_recall(
                        final_mask, data_norm[:,k], recall_thresh=0.95)
                    print np.min(data_norm[:,k])
                    print threshold
                    print np.sum(data_norm[:,k] > threshold)
                    minimal_motif_mask = np.multiply(minimal_motif_mask, data_norm[:,k] > threshold)
                    
                final_mask = np.multiply(minimal_motif_mask, metacluster_mask) # the label set
                    
                # fit a polynomial (degree 2, just pairwise interactions) model
                sub_dataset = masked_data[:,np.sum(masked_data, axis=0) > 0]
                
                # sub_dataset also needs to be filtered by motif presence
                sub_dataset = np.multiply(
                    sub_dataset,
                    np.expand_dims(minimal_motif_mask, axis=1))
                
                print sub_dataset.shape
                
                print np.sum(minimal_motif_mask)
                print np.sum(final_mask)
                print np.sum(pwm_vector)


                
                import ipdb
                ipdb.set_trace()
                
                
                # build polynomial features
                poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
                X = poly.fit_transform(sub_dataset)
                #X = sub_dataset
                poly_names = poly.get_feature_names([pwm_list[k].name for k in indices])
                print poly_names
                y = np.multiply(final_mask, hf["logits"][:,label_index]) # match to logits

                X_pos = X[final_mask == 1,:]
                y_pos = y[final_mask == 1]

                print X_pos.shape
                print y_pos.shape

                if y_pos.shape[0] == 0:
                    continue

                # note - if can't build a model, probably means not real - so ignore?
                # LATER - fit the residuals with negative scores?
                clf = build_regression_model(X_pos, y_pos)

                # TO FIX - need to consider the indicator functions (all motifs present)
                scores = clf.predict(X)
                
                # get threshold, on full dataset
                print "total in cluster:", np.sum(final_mask)
                true_pos_rate = np.sum(final_mask) / float(final_mask.shape[0])
                print true_pos_rate
                get_threshold = make_threshold_at_tpr(true_pos_rate)
                #get_threshold = make_threshold_at_fdr(true_pos_rate)
                threshold = get_threshold(final_mask, scores)

                # save out weights into grammar file
                node_dict = {}
                edge_dict = {}
                for coef_idx in xrange(len(poly_names)):

                    if clf.coef_[coef_idx] == 0:
                        continue
                    
                    # get name and split
                    nodes = poly_names[coef_idx].split()

                    # check for squared nodes
                    if len(nodes) == 1:
                        if "^2" not in nodes[0]:
                            # single - save to node dict
                            node_dict[nodes[0]] = clf.coef_[coef_idx]
                        else:
                            # squared term - save to edge_dict
                            edge_dict[(nodes[0], nodes[0])] =  clf.coef_[coef_idx]
                    elif len(nodes) == 2:
                        # interaction term - save to edge_dict
                        edge_dict[(nodes[0], nodes[1])] =  clf.coef_[coef_idx]
                    else:
                        raise Exception("Higher term polynomial not implemented yet")

                # TODO given the model, get the AUPRC, AUROC, and recall at FDR
                # for each label set
                num_labels = hf["labels"].shape[1]
                auroc_by_label = np.zeros((num_labels))
                auprc_by_label = np.zeros((num_labels))
                recall_by_label = np.zeros((num_labels))
                accuracy_by_label = np.zeros((num_labels))
                for label_idx in xrange(num_labels):
                    label_set = hf["labels"][:, label_idx]

                    from tronn.run_evaluate import auprc
                    from tronn.run_evaluate import make_recall_at_fdr
                    from sklearn.metrics import roc_auc_score
                    # calculate metrics
                    try:
                        auprc_by_label[label_idx] = roc_auc_score(label_set, scores)
                    except:
                        auprc_by_label[label_idx] = 0
                    try:
                        auprc_by_label[label_idx] = auprc(label_set, scores)
                    except:
                        auprc_by_label[label_idx] = 0
                    try:
                        recall_at_fdr = make_recall_at_fdr(0.05)
                        recall_by_label[label_idx] = recall_at_fdr(label_set, scores)
                    except:
                        recall_by_label[label_idx] = 0

                    #accuracy_by_label[label_idx] = np.sum(np.multiply(
                    #    label_set, scores > threshold)) / np.sum(label_set)
                    accuracy_by_label[label_idx] = np.sum(np.multiply(
                        label_set, final_mask)) / np.sum(label_set)
                print accuracy_by_label[14:23]

                task_grammar = Grammar(
                    pwm_file,
                    node_dict,
                    edge_dict,
                    ("taskidx={0};"
                     "type=elastic_net;"
                     "directed=no;"
                     "threshold={1}").format(j, threshold),
                    "metacluster-{0}.taskidx-{1}".format(
                        metacluster_id, j)) # TODO consider adjusting the taskidx here
                task_grammar.to_file(grammar_file)
                    
                # save out as a table
                test = pd.DataFrame()
                test["nn_logits"] = y_pos
                test["linear_predictions"] = clf.predict(X_pos)

                test.to_csv("testing.txt", sep="\t")

                import ipdb
                ipdb.set_trace()
                
                # TODO - this is where to save out coefficients in proper places
                # have both pwm vector AND adjacency matrix (pairwise)
                # use the names as indices
                coefficients = clf.coef_.tolist()
                
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
            if False:
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
                         "threshold={1}").format(label_index, threshold),
                        "metacluster-{0}.taskidx-{1}".format(
                            metacluster_id, label_index))
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


