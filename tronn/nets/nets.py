"""Access point for all nets
"""

from tronn.nets.deep_nets import basset
from tronn.nets.deep_nets import fbasset
from tronn.nets.deep_nets import basset_plus
from tronn.nets.deep_nets import danq
from tronn.nets.deep_nets import deepsea
from tronn.nets.deep_nets import resnet
from tronn.nets.deep_nets import tfslim_inception
from tronn.nets.deep_nets import tfslim_resnet
from tronn.nets.deep_nets import empty_net

from tronn.nets.preprocess_nets import postprocess_mutate

from tronn.nets.inference_nets import sequence_to_importances
from tronn.nets.inference_nets import sequence_to_pwm_scores
from tronn.nets.inference_nets import importance_scores_to_pwm_scores
from tronn.nets.inference_nets import pwm_scores_to_dmim
from tronn.nets.inference_nets import sequence_to_synergy
from tronn.nets.inference_nets import sequence_to_synergy_sims
from tronn.nets.inference_nets import variants_to_scores

net_fns = {
    "basset": basset,
    "fbasset": fbasset,
    "resbasset": basset_plus,
    "danq": danq,
    "deepsea": deepsea,
    "inception": tfslim_inception,
    "resnet": tfslim_resnet,
    "empty_net": empty_net,

    "postprocess_mutate": postprocess_mutate,
    
    "sequence_to_importances": sequence_to_importances,
    "sequence_to_motif_scores": sequence_to_pwm_scores,
    "importance_scores_to_motif_scores": importance_scores_to_pwm_scores,
    "sequence_to_dmim": pwm_scores_to_dmim,
    "sequence_to_synergy": sequence_to_synergy,
    "sequence_to_synergy_sims": sequence_to_synergy_sims,
    "variants_to_scores": variants_to_scores
}






