# description: build grammars from scans

import h5py
import logging

import numpy as np
import networkx as nx
import seaborn as sns

from tronn.interpretation.networks import add_graphics_theme_to_nx_graph
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

    # get pwm names, adjust as needed
    with h5py.File(args.scan_file, "r") as hf:
        pwm_names = hf[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL].attrs["pwm_names"]
        sig_pwms_names = [pwm_names[i] for i in sig_pwms_indices]
        
    # figure out min support
    with h5py.File(args.scan_file, "r") as hf:
        total_examples = hf[DataKeys.SEQ_METADATA].shape[0]
    min_support = max(total_examples * args.min_support_fract, args.min_support)
    logging.info("Using min support of {}".format(min_support))

    # adjust the pwms by presence arg
    args.keep_grammars = [pwms.split(",") for pwms in args.keep_grammars]

    # set up sig key
    if args.scan_type == "muts":
        sig_key = DataKeys.MUT_MOTIF_LOGITS_SIG
    elif args.scan_type == "impts":
        #sig_key = DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM #HITS_COUNT
        sig_key = DataKeys.WEIGHTED_SEQ_PWM_HITS_COUNT
    elif args.scan_type == "hits":
        sig_key = DataKeys.ORIG_SEQ_PWM_HITS_COUNT
    else:
        raise ValueError, "scan type must be muts, impts, or hits!"
    
    # make graph
    graph = build_full_graph(
        args.scan_file,
        sig_pwms_indices,
        sig_pwms_names,
        sig_mut_logits_key=sig_key,
        rc_pwms=args.rc_pwms_present,
        min_region_num=min_support,
        keep_grammars=args.keep_grammars,
        ignore_pwms=args.ignore_pwms)

    # get subgraphs
    subgraphs = get_maxsize_k_subgraphs(
        graph,
        min_support,
        k=args.subgraph_max_k,
        keep_grammars=args.keep_grammars)
    
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
