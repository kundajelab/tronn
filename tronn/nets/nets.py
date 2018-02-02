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

from tronn.nets.inference_nets import sequence_to_importance_scores
from tronn.nets.inference_nets import sequence_to_motif_scores
from tronn.nets.inference_nets import sequence_to_grammar_scores

from tronn.nets.mutate_nets import ism_for_grammar_dependencies


net_fns = {
    "basset": basset,
    "fbasset": fbasset, #factorized basset
    "danq": danq,
    "deepsea": deepsea,
    "inception": tfslim_inception,
    "resnet": tfslim_resnet,
    "empty_net": empty_net,
    "sequence_to_importance_scores": sequence_to_importance_scores,
    "sequence_to_motif_scores": sequence_to_motif_scores,
    "sequence_to_grammar_scores": sequence_to_grammar_scores,
    "ism": ism_for_grammar_dependencies
}






