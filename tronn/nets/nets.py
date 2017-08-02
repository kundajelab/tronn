"""Access point for all nets
"""

from tronn.nets.deep_nets import basset
from tronn.nets.deep_nets import danq
from tronn.nets.deep_nets import resnet

from tronn.nets.threshold_nets import pois_cutoff
from tronn.nets.threshold_nets import stdev_cutoff

from tronn.nets.motif_nets import pwm_convolve_v2

from tronn.nets.mutate_nets import ism_for_grammar_dependencies

from tronn.grammar_nets import single_grammar
from tronn.grammar_nets import multiple_grammars


model_fns = {
    "basset": basset,
    "danq": danq,
    "resnet": resnet,
    "pois_cutoff": pois_cutoff,
    "stdev_cutoff": stdev_cutoff,
    "pwm_convolve": pwm_convolve_v2,
    "ism": ism_for_grammar_dependencies,
    "grammar": single_grammar,
    "grammars": multiple_grammars
}






