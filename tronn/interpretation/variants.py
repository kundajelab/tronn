# description: code for dealing with variants/mutants

import h5py

import numpy as np

from tronn.interpretation.clustering import get_clusters_from_h5

from tronn.stats.nonparametric import run_delta_permutation_test

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


def get_interacting_motifs(
        h5_file,
        clusters_key,
        out_key,
        mut_effects_key=DataKeys.FEATURES):
    """determine which motifs responded to muts
    """
    with h5py.File(h5_file, "r") as hf:
        mut_data = hf[mut_effects_key][:]
        sig_pwms = hf[DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL][:]
        pwm_names = hf[DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL].attrs[AttrKeys.PWM_NAMES]
        dx = hf["dx"][:] # {N, mutM}
        # TODO - also logits? {N, mutM, task, 1}
        # if have this can also plot out logits according to muts
        
    # subset down for time sake
    mut_data = mut_data[:,:,:,np.where(sig_pwms > 0)[0]]
    
    # and get dy/dx
    dx = np.reshape(dx, list(dx.shape) + [1 for i in xrange(len(mut_data.shape)-len(dx.shape))])
    dydx = np.divide(mut_data, dx)
    dydx = np.where(
        np.isfinite(dydx),
        dydx,
        np.zeros_like(dydx))
                    
    # try global first
    sig_responses = run_delta_permutation_test(dydx) # {mutM, task, M}
    #sig_responses = np.ones(dydx.shape[1:])

    
    # get mean for places where this score exists
    if True:
        #agg_mut_data = np.divide(
        #    np.sum(mut_data, axis=0),
        #    np.sum(mut_data != 0, axis=0))
        agg_mut_data = np.sum(mut_data, axis=0)
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
