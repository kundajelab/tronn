# description: code for dealing with variants/mutants

import h5py

import numpy as np

from tronn.interpretation.clustering import get_clusters_from_h5

from tronn.stats.nonparametric import run_delta_permutation_test

from tronn.util.utils import DataKeys


# NOTE: for these, it may be best to just aggregate results
# per motif (DONT look at clusters here)
# {N, mutM, logit}
# expectation: every motif is significant (otherwise we
# have an issue with extracting significant motifs)
# the distinction might be which drive the biggest changes individually
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
        mut_effects_key=DataKeys.FEATURES):
    """determine which motifs responded to muts
    """
    with h5py.File(h5_file, "r") as hf:
        mut_data = hf[mut_effects_key][:]
        sig_pwms = hf[DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL][:]
        
    # subset down for time sake
    mut_data_tmp = mut_data[:,:,:,np.where(sig_pwms > 0)[0]]
    
    import ipdb
    ipdb.set_trace()
    
    # try global first
    sig_responses = run_delta_permutation_test(mut_data)

    import ipdb
    ipdb.set_trace()

    return
