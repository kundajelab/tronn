# description: additional preprocessing to be done before sending inputs to the model

import tensorflow as tf

from tronn.nets.importance_nets import DeltaFeatureImportanceMapper
from tronn.nets.motif_nets import filter_for_any_sig_pwms
from tronn.nets.mutate_nets import mutate_weighted_motif_sites


def mutate_sequences_single_motif(inputs, params):
    """filter for sequences that have motif desired, and mutate
    """
    # filter and mutate
    outputs, params = filter_for_any_sig_pwms(inputs, params)
    outputs, params = mutate_weighted_motif_sites(outputs, params)

    # attach
    dfim = DeltaFeatureImportanceMapper(None)
    outputs, params = dfim.preprocess(outputs, params)
    
    return outputs, params


def postprocess_mutate(inputs, params):
    """after model, do this
    """
    # remove the shuffles - weighted seq (orig seq shuffles are already saved)
    params.update({"rebatch_name": "detach_mut_motif_seq"})

    # 
    dfim = DeltaFeatureImportanceMapper(None)    
    outputs, params = detach_auxiliary_tensors(inputs, params)
    outputs, params = dfim.adjust_aux_axes(outputs, params)

    # adjust logits etc 
    

    return
