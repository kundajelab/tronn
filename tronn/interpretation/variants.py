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
        
    # subset down for time sake
    mut_data = mut_data[:,:,:,np.where(sig_pwms > 0)[0]]
    
    # try global first
    sig_responses = run_delta_permutation_test(mut_data) # {mutM, task, M}

    # multiply by the scores
    agg_mut_data = np.divide(
        np.sum(mut_data, axis=0),
        np.sum(mut_data != 0, axis=0))
    agg_mut_data_sig = np.multiply(sig_responses, agg_mut_data)
    agg_mut_data_sig = np.where(
        np.isfinite(agg_mut_data_sig),
        agg_mut_data_sig,
        np.zeros_like(agg_mut_data_sig))
    
    import ipdb
    ipdb.set_trace()

    # TODO save this to the h5 file
    with h5py.File(h5_file, "a") as out:
        out.create_dataset(out_key, data=sig_responses)
        out[out_key].attrs[AttrKeys.PWM_NAMES] = pwm_names

    return None
