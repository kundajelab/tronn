# description: build grammars from scans

import h5py
import logging

import seaborn as sns

from tronn.interpretation.networks import build_co_occurrence_graph
from tronn.interpretation.networks import build_dmim_graph
from tronn.interpretation.networks import get_subgraphs_and_filter
from tronn.interpretation.networks import sort_subgraphs_by_output_strength
from tronn.interpretation.networks import build_subgraph_per_example_array
from tronn.interpretation.networks import add_graphics_theme_to_nx_graph

from tronn.util.utils import DataKeys


def run(args):
    """build grammars from scan (scanmotifs or dmim)
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Building grammars")

    # adjust for sig file as needed
    if args.sig_pwms_file is None:
        args.sig_pwms_file = args.scan_file
        
    # extract metadata
    with h5py.File(args.scan_file, "r") as hf:
        metadata = hf[DataKeys.SEQ_METADATA][:]
        
    # set up targets
    with h5py.File(args.sig_pwms_file, "r") as hf:
        target_keys = hf[args.sig_pwms_key].keys()

    for target_key in target_keys:
        
        # set up indices
        with h5py.File(args.sig_pwms_file, "r") as hf:
            indices = hf[args.sig_pwms_key][target_key].keys()
        indices = [i for i in indices if "pwms" not in i]
            
        for index in indices:

            # debug
            if int(index) == 1:
                continue

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

            # debug
            for i in xrange(len(sorted_subgraphs)):
                subgraph = sorted_subgraphs[i]
                print [node.name for node in subgraph.nodes], len(subgraph.attrs["examples"]), subgraph.attrs["logit_max"]

            # in the original scan file, now save out results {N, subgraph}
            # TODO change this key!
            if False:
                #grammar_labels_key = "{}/{}".format(subgroup_key, DataKeys.GRAMMAR_LABELS)
                build_subgraph_per_example_array(
                    args.scan_file,
                    sorted_subgraphs,
                    target_key,
                    grammar_labels_key)
            
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
                try:
                    subgraph.write_gml(grammar_gml)
                except:
                    import ipdb
                    ipdb.set_trace()
                grammar_bed = "{}.bed".format(grammar_file_prefix)
                subgraph.write_bed(grammar_bed, merge=True)
        
    return
