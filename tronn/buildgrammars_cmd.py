# description: build grammars from scans

import h5py
import logging

import numpy as np
import networkx as nx
import seaborn as sns

from tronn.interpretation.networks import build_co_occurrence_graph
from tronn.interpretation.networks import build_dmim_graph
from tronn.interpretation.networks import get_subgraphs_and_filter
from tronn.interpretation.networks import sort_subgraphs_by_output_strength
from tronn.interpretation.networks import build_subgraph_per_example_array
from tronn.interpretation.networks import add_graphics_theme_to_nx_graph
from tronn.interpretation.networks import annotate_subgraphs_with_pwm_scores
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

    quit()
        
    nodes_w_delta = []
    for i in range(len(subgraphs)):
        average_change = [data["deltalogits"][5] for node, data in subgraphs[i].nodes(data=True)]
        average_change = np.max(np.array(average_change))
        #print subgraphs[i].nodes, average_change
        nodes_w_delta.append((list(subgraphs[i].nodes), average_change))

    sorted_subgraphs = sorted(nodes_w_delta, key=lambda x:x[1])
    for subgraph in sorted_subgraphs:
        print subgraph
    
    import ipdb
    ipdb.set_trace()
    

    quit()
    
    # set up targets
    with h5py.File(args.sig_pwms_file, "r") as hf:
        target_keys = hf[args.sig_pwms_key].keys()
        
    for target_key in target_keys:
        
        # set up indices
        with h5py.File(args.sig_pwms_file, "r") as hf:
            indices = hf[args.sig_pwms_key][target_key].keys()
        indices = [i for i in indices if "pwms" not in i]
            
        for index in indices:

            # set up keys
            subgroup_key = "{}/{}/{}".format(
                args.sig_pwms_key, target_key, index)
            logging.info("Building for {}".format(subgroup_key))
            sig_pwms_key = "{}/{}".format(subgroup_key, DataKeys.PWM_SIG_ROOT)
            
            # build graph
            # here, either select co-occurrence or dmim
            if args.scan_type == "motifs":
                graph = build_co_occurrence_graph(
                    args.scan_file,
                    target_key,
                    int(index),
                    args.sig_pwms_file,
                    sig_pwms_key,
                    min_positive=args.min_positive_tasks,
                    min_co_occurring_num=args.min_support)
            elif args.scan_type == "dmim":
                sig_dmim_key = "{}/{}/{}/{}".format(
                    DataKeys.DMIM_DIFF_GROUP,
                    target_key,
                    index,
                    DataKeys.DMIM_SIG_ROOT)
                
                graph = build_dmim_graph(
                    args.scan_file,
                    target_key,
                    int(index),
                    args.sig_pwms_file,
                    sig_pwms_key,
                    sig_dmim_key,
                    min_positive=args.min_positive_tasks,
                    min_co_occurring_num=args.min_support)
            else:
                raise ValueError, "scan type not implemented"
                
            # get subgraphs
            subgraphs = get_subgraphs_and_filter(
                graph,
                args.subgraph_max_k,
                # TODO these two numbers should be one,
                # since just doing intersects here and no unions
                args.min_support, 
                args.min_support,
                args.max_overlap_fraction)

            # only keep top k if requested
            if args.return_top_k is not None:
                sorted_subgraphs = sort_subgraphs_by_output_strength(
                    subgraphs,
                    args.scan_file,
                    DataKeys.LOGITS)
                sorted_subgraphs = sorted_subgraphs[:args.return_top_k]

            # other annotations:
            # extract score patterns for each node (to viz later)
            # TODO also edges?
            annotate_subgraphs_with_pwm_scores(
                sorted_subgraphs,
                args.scan_file,
                DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM)
            
            # debug
            for i in xrange(len(sorted_subgraphs)):
                subgraph = sorted_subgraphs[i]
                print [node.name for node in subgraph.nodes], len(subgraph.attrs["examples"]), subgraph.attrs["logit_max"]

            # for each subgraph, produce:
            # 1) gml file that keeps the network structure
            # 3) BED file of regions to check functional enrichment
            colors = sns.color_palette("hls", len(sorted_subgraphs)).as_hex()
            for subgraph_idx in xrange(len(sorted_subgraphs)):
                subgraph = sorted_subgraphs[subgraph_idx]

                # adjust edge color
                for edge in subgraph.edges:
                    subgraph.edge_attrs[edge.name]["graphics"] = {
                        "fill": colors[subgraph_idx].upper()}
                
                grammar_prefix = "grammar-{}".format(subgraph_idx)
                grammar_file_prefix = "{}/{}.{}-{}.{}".format(
                    args.out_dir, args.prefix, target_key, index, grammar_prefix)
                grammar_gml = "{}.gml".format(grammar_file_prefix)
                subgraph.write_gml(grammar_gml)
                grammar_bed = "{}.bed".format(grammar_file_prefix)
                subgraph.write_bed(grammar_bed, merge=True)

                # todo save out pwm scores attached to nodes too

                
                
    return
