# description: additional preprocessing to be done before sending inputs to the model

import numpy as np
import tensorflow as tf

from tronn.nets.importance_nets import DeltaFeatureImportanceMapper
from tronn.nets.motif_nets import filter_for_any_sig_pwms
from tronn.nets.motif_nets import extract_null_results
from tronn.nets.motif_nets import get_sig_mut_logits
from tronn.nets.mutate_nets import mutate_weighted_motif_sites
from tronn.nets.util_nets import detach_auxiliary_tensors

from tronn.util.utils import DataKeys

def mutate_sequences_single_motif(inputs, params):
    """filter for sequences that have motif desired, and mutate
    """
    # check if rc PWMs used - can note it by looking at sig pwm vector
    # compared to score tensor, and then adjust sig pwm vector as needed.
    len_sig_pwms = params["sig_pwms"].shape[0]
    len_score_pwms = inputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM].shape[2]
    if 2*len_sig_pwms == len_score_pwms:
        params["sig_pwms"] = np.concatenate(
            [params["sig_pwms"], params["sig_pwms"]], axis=0)
    
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

    #  use dfim to detach
    dfim = DeltaFeatureImportanceMapper(None)
    params.update({"save_aux": {DataKeys.LOGITS: DataKeys.MUT_MOTIF_LOGITS}})
    outputs, params = detach_auxiliary_tensors(inputs, params)
    outputs, _ = extract_null_results(outputs, params)
    outputs, _ = get_sig_mut_logits(outputs, params)
    del outputs[DataKeys.FEATURES]

    return outputs, params
