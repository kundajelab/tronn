# description: code for dealing with variants/mutants

import h5py
import logging

import numpy as np

from tronn.interpretation.clustering import get_clusters_from_h5

from tronn.stats.nonparametric import run_delta_permutation_test
from tronn.stats.nonparametric import threshold_by_qvalues

from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys


def get_significant_delta_logit_responses(
        h5_file,
        clusters_key,
        true_logits_key=DataKeys.LOGITS,
        mut_logits_key=DataKeys.MUT_MOTIF_LOGITS):
    """get delta responses
    """
    with h5py.File(h5_file, "r") as hf:
        true_logits = np.expand_dims(hf[DataKeys.LOGITS][:], axis=1)
        mut_logits = hf[DataKeys.MUT_MOTIF_LOGITS][:]
        delta_logits = np.subtract(mut_logits, true_logits)
        
    clusters = get_clusters_from_h5(h5_file, clusters_key)
    generator = clusters.cluster_mask_generator()
    cluster_idx = 0
    for cluster_id, cluster_mask in generator:

        cluster_delta_logits = delta_logits[np.where(cluster_mask)[0],:,:]
        
        cluster_sig_motifs = run_delta_permutation_test(cluster_delta_logits)

        import ipdb
        ipdb.set_trace()

    
    return


# TODO rename this and move somewhere else! interpretation.dmim?
def run_permutation_dmim_score_test(
        h5_file,
        target_key,
        target_indices,
        overall_sig_pwms,
        sig_pwms_key,
        dmim_score_key=DataKeys.FEATURES,
        qval_thresh=0.05,
        reduce_sig_type="any",
        num_threads=24):
    """given a differential score, test to see if this score could have 
    arisen by chance (paired permutation test)
    """
    # get foregound info
    with h5py.File(h5_file, "r") as hf:
        targets = hf[target_key][:] # {N, target}
        dmim_scores = hf[dmim_score_key][:] # {N, mutM, task, M}
        dx = hf["dx"][:] # {N, mutM}
        #pwm_names = hf[dmim_score_key].attrs[AttrKeys.PWM_NAMES]

    # adjust indices and filter
    if len(target_indices) == 0:
        target_indices = range(targets.shape[1])
    targets = targets[:,target_indices]
    
    # and iterate through for each set
    for target_idx in xrange(targets.shape[1]):
        logging.info("Running target idx {}".format(target_idx))

        subset_key = "{}/{}/{}".format(
            DataKeys.DMIM_DIFF_GROUP,
            target_key,
            target_indices[target_idx])
        
        # set up sig pwms
        subset_sig_pwms_key = "{}/{}/{}/{}".format(
            sig_pwms_key, target_key, target_indices[target_idx], DataKeys.PWM_SIG_ROOT)
        with h5py.File(h5_file, "r") as hf:
            subset_sig_pwms = hf[subset_sig_pwms_key][:] # {M}
        subset_sig_pwms = subset_sig_pwms[np.where(overall_sig_pwms)[0]]
        keep_pwms = np.where(subset_sig_pwms != 0)[0]
        
        # get region set
        subset_indices = np.where(targets[:,target_idx] > 0)[0]
        subset_scores = dmim_scores[subset_indices][:,keep_pwms]
        subset_dx = dx[subset_indices][:,keep_pwms]

        # only look at the mut motifs
        if True:
            subset_scores = subset_scores[:,:,:,np.where(subset_sig_pwms != 0)[0]]
        
        # set up dy/dx
        # remember that dy/dx is POSITIVE if the feature drops in response to mutation
        # the dy/dx is NEGATIVE if the feature increases in response to the mutation
        dx = np.reshape(
            dx,
            list(dx.shape) + [1 for i in xrange(len(subset_scores.shape)-len(dx.shape))])
        dydx = np.divide(subset_scores, dx, out=np.zeros_like(subset_scores), where=dx!=0)

        # run permutation test
        pvals = run_delta_permutation_test(
            dydx, qval_thresh=0.10, num_threads=num_threads) # {mutM, task, M}

        # use qvals to threshold
        pass_qval_thresh = threshold_by_qvalues(
            pvals, qval_thresh=qval_thresh, num_bins=50)

        # also get a condensed view across tasks
        if reduce_sig_type == "any":
            reduce_fn = np.any
        else:
            reduce_fn = np.all
        sig_dmim = reduce_fn(pass_qval_thresh, axis=1) # {mutM, M}

        # and save all of this out
        with h5py.File(h5_file, "a") as hf:
            pvals_key = "{}/{}".format(subset_key, DataKeys.DMIM_PVALS)
            if hf.get(pvals_key) is not None:
                del hf[pvals_key]
            hf.create_dataset(pvals_key, data=pvals)

            sig_full_key = "{}/{}".format(subset_key, DataKeys.DMIM_SIG_ALL)
            if hf.get(sig_full_key) is not None:
                del hf[sig_full_key]
            hf.create_dataset(sig_full_key, data=pass_qval_thresh)
            
            sig_dmim_key = "{}/{}".format(subset_key, DataKeys.DMIM_SIG_ROOT)
            if hf.get(sig_dmim_key) is not None:
                del hf[sig_dmim_key]
            hf.create_dataset(sig_dmim_key, data=sig_dmim)
        
    return


    


def get_interacting_motifs(
        h5_file,
        clusters_key, # need this to get sig pwms and then sig names and that's it
        out_key,
        mut_effects_key=DataKeys.FEATURES):
    """determine which motifs responded to muts
    """
    with h5py.File(h5_file, "r") as hf:
        mut_data = hf[mut_effects_key][:] # {N, mutM, task, M}
        sig_pwms = hf[DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL][:]
        pwm_names = hf[DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL].attrs[AttrKeys.PWM_NAMES]
        dx = hf["dx"][:] # {N, mutM}
        # TODO - also logits? {N, mutM, task, 1}
        # if have this can also plot out logits according to muts
        
    # subset down for time sake
    mut_data = mut_data[:,:,:,np.where(sig_pwms > 0)[0]] #in practice DONT need to do this!
    
    # and get dy/dx
    dx = np.reshape(dx, list(dx.shape) + [1 for i in xrange(len(mut_data.shape)-len(dx.shape))])
    dydx = np.divide(mut_data, dx)
    dydx = np.where(
        np.isfinite(dydx),
        dydx,
        np.zeros_like(dydx))
                    
    # try global first
    # remember that dy/dx is POSITIVE if the feature drops in response to mutation
    # the dy/dx is NEGATIVE if the feature increases in response to the mutation
    pvals, sig_responses = run_delta_permutation_test(dydx) # {mutM, task, M}
    #sig_responses = np.ones(dydx.shape[1:])

    # get mean for places where this score exists
    if True:
        #agg_mut_data = np.divide(
        #    np.sum(mut_data, axis=0),
        #    np.sum(mut_data != 0, axis=0))
        #agg_mut_data = np.sum(mut_data, axis=0)
        agg_mut_data = np.mean(mut_data, axis=0)
    else:
        agg_mut_data = np.sum(mut_data != 0, axis=0)
        
    # multiply by the scores
    agg_mut_data_sig = np.multiply(sig_responses, agg_mut_data)
    agg_mut_data_sig = np.where(
        np.isfinite(agg_mut_data_sig),
        agg_mut_data_sig,
        np.zeros_like(agg_mut_data_sig))

    # save this to the h5 file
    # TODO save out the sig mask also
    with h5py.File(h5_file, "a") as out:
        if out.get(out_key) is not None:
            del out[out_key]
        out.create_dataset(out_key, data=agg_mut_data_sig)
        out[out_key].attrs[AttrKeys.PWM_NAMES] = pwm_names

    return None



def visualize_interacting_motifs_R(
        h5_file,
        visualize_key,
        pwm_names_attr_key=AttrKeys.PWM_NAMES):
    """visualize results
    """
    
    r_cmd = "plot-h5.dmim_sig_agg.R {} {} {}".format(
        h5_file, visualize_key, pwm_names_attr_key)
    print r_cmd
    
    return None
