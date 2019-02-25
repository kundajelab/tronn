# description: build grammars from scans

import h5py
import logging

import numpy as np
import networkx as nx
import seaborn as sns

#from tronn.interpretation.networks import build_co_occurrence_graph
#from tronn.interpretation.networks import build_dmim_graph
#from tronn.interpretation.networks import get_subgraphs_and_filter
#from tronn.interpretation.networks import sort_subgraphs_by_output_strength
#from tronn.interpretation.networks import build_subgraph_per_example_array
from tronn.interpretation.networks import add_graphics_theme_to_nx_graph
#from tronn.interpretation.networks import annotate_subgraphs_with_pwm_scores
from tronn.interpretation.networks import build_full_graph
from tronn.interpretation.networks import get_maxsize_k_subgraphs
from tronn.interpretation.networks import attach_mut_logits
from tronn.interpretation.networks import attach_data_summary
from tronn.interpretation.networks import write_bed_from_graph
from tronn.interpretation.networks import stringize_nx_graph

from tronn.util.utils import DataKeys


def run(args):
    """build grammars from scan (scanmotifs or dmim)
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Building grammars")
    
    # get sig pwms vector
    with h5py.File(args.sig_pwms_file, "r") as hf:
        sig_pwms = hf[args.sig_pwms_key][args.foreground_targets]["sig"][:]
        sig_pwms_indices = np.where(sig_pwms != 0)[0]

    # adjust the min number of regions based on min fract and min support
    with h5py.File(args.scan_file, "r") as hf:
        total_examples = hf[DataKeys.FEATURES].shape[0]
    min_support = max(total_examples * args.min_support_fract, args.min_support)
    
    # make graph
    graph = build_full_graph(
        args.scan_file,
        sig_pwms_indices,
        min_positive_tasks=args.min_positive_tasks,
        min_region_num=min_support)

    # get subgraphs
    subgraphs = get_maxsize_k_subgraphs(graph, min_support)

    # attach delta logits to nodes
    subgraphs = attach_mut_logits(subgraphs, args.scan_file)

    # attach other things to nodes
    for key in args.aux_data_keys:
        subgraphs = attach_data_summary(subgraphs, args.scan_file, key)
        
    # for each subgraph, produce:
    # 1) gml file that keeps the network structure
    # 3) BED file of regions to check functional enrichment
    colors = sns.color_palette("hls", len(subgraphs)).as_hex()
    for subgraph_idx in xrange(len(subgraphs)):
        grammar_prefix = "grammar-{}".format(subgraph_idx)
        grammar_file_prefix = "{}/{}.{}.{}".format(
            args.out_dir, args.prefix, args.foreground_targets, grammar_prefix)
        grammar_gml = "{}.gml".format(grammar_file_prefix)
        grammar_bed = "{}.bed".format(grammar_file_prefix)

        # get subgraph
        subgraph = subgraphs[subgraph_idx]

        # adjust edge color
        for start_node, end_node, data in subgraph.edges(data=True):
            data["graphics"] = {"fill": colors[subgraph_idx].upper()}

        # write out
        write_bed_from_graph(subgraph, grammar_bed, merge=True)
        nx.write_gml(stringize_nx_graph(subgraph.copy()), (grammar_gml))
                
    return
