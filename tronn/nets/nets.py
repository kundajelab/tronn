"""Access point for all nets
"""

from tronn.nets.deep_nets import basset
from tronn.nets.deep_nets import danq
from tronn.nets.deep_nets import deepsea
from tronn.nets.deep_nets import resnet
from tronn.nets.deep_nets import tfslim_inception
from tronn.nets.deep_nets import tfslim_resnet

from tronn.nets.inference_nets import get_importances
from tronn.nets.inference_nets import get_top_k_motif_hits
from tronn.nets.inference_nets import sequence_to_motif_assignments
from tronn.nets.inference_nets import importances_to_motif_assignments

from tronn.nets.mutate_nets import ism_for_grammar_dependencies
from tronn.nets.grammar_nets import single_grammar
from tronn.nets.grammar_nets import multiple_grammars


net_fns = {
    "basset": basset,
    "danq": danq,
    "deepsea": deepsea,
    "inception": tfslim_inception,
    "resnet": tfslim_resnet,
    "get_importances": get_importances,
    "sequence_to_motif_assignment": sequence_to_motif_assignments,
    "importances_to_motif_assignments": importances_to_motif_assignments,
    "get_top_k_motif_hits": get_top_k_motif_hits,
    "ism": ism_for_grammar_dependencies,
    "grammar": single_grammar,
    "grammars": multiple_grammars
}






