"""Access point for all nets
"""

from tronn.nets.deep_nets import basset
from tronn.nets.deep_nets import danq
from tronn.nets.deep_nets import deepsea
from tronn.nets.deep_nets import resnet
from tronn.nets.deep_nets import rbasset
from tronn.nets.deep_nets import splitbasset
from tronn.nets.deep_nets import tfslim_inception
from tronn.nets.deep_nets import tfslim_resnet

from tronn.nets.threshold_nets import pois_cutoff
from tronn.nets.threshold_nets import stdev_cutoff

from tronn.nets.motif_nets import pwm_convolve_v2

from tronn.nets.mutate_nets import ism_for_grammar_dependencies

from tronn.nets.grammar_nets import single_grammar
from tronn.nets.grammar_nets import multiple_grammars


model_fns = {
    "basset": basset,
    "danq": danq,
    "deepsea": deepsea,
    #"resnet": resnet,
    "rbasset": rbasset,
    "splitbasset": splitbasset,
    "inception": tfslim_inception,
    "resnet": tfslim_resnet,
    "pois_cutoff": pois_cutoff,
    "stdev_cutoff": stdev_cutoff,
    "pwm_convolve": pwm_convolve_v2,
    "ism": ism_for_grammar_dependencies,
    "grammar": single_grammar,
    "grammars": multiple_grammars
}






