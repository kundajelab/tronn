"""Access point for all nets
"""

from tronn.nets.deep_nets import fbasset
from tronn.nets.deep_nets import basset
from tronn.nets.deep_nets import danq
from tronn.nets.deep_nets import deepsea
from tronn.nets.deep_nets import resnet
from tronn.nets.deep_nets import tfslim_inception
from tronn.nets.deep_nets import tfslim_resnet
from tronn.nets.deep_nets import empty_net
from tronn.nets.deep_nets import ensemble

#from tronn.nets.inference_nets import sequence_to_importance_scores
from tronn.nets.inference_nets import sequence_to_motif_scores
from tronn.nets.inference_nets import sequence_to_motif_scores_from_regression
from tronn.nets.inference_nets import sequence_to_dmim
from tronn.nets.inference_nets import variants_to_predictions

net_fns = {
    "basset": basset,
    "fbasset": fbasset, #factorized basset
    "danq": danq,
    "deepsea": deepsea,
    "inception": tfslim_inception,
    "resnet": tfslim_resnet,
    "empty_net": empty_net,
    "ensemble": ensemble,
    #"sequence_to_importance_scores": sequence_to_importance_scores,
    "sequence_to_motif_scores": sequence_to_motif_scores,
    "sequence_to_motif_scores_from_regression": sequence_to_motif_scores_from_regression,
    "sequence_to_dmim": sequence_to_dmim,
    "variants_to_predictions": variants_to_predictions
}






