# description: build grammars from scans

import h5py
import logging

from tronn.interpretation.networks import build_co_occurrence_graph
from tronn.interpretation.networks import get_subgraphs_and_filter
from tronn.interpretation.networks import sort_subgraphs_by_output_strength

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
            graph = build_co_occurrence_graph(
                args.scan_file,
                target_key,
                int(index),
                args.sig_pwms_file,
                sig_pwms_key,
                min_positive=args.min_positive_tasks,
                min_co_occurring_num=args.min_support)

            # get subgraphs
            subgraphs = get_subgraphs_and_filter(
                graph,
                args.subgraph_max_k,
                args.min_support, # TODO these two numbers should be one, since just doing intersects here and no unions
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
                print [node.name for node in subgraph.nodes], len(subgraph.attrs["examples"]), subgraph.attrs["logit_mean"]
                
            # summaries: summary metadata file
            # do I need this summary?
            summary_metadata_file = "{}/{}.{}-{}.summary.txt".format(
                args.out_dir, args.prefix, target_key, index)

            # for each subgraph, produce:
            # 1) gml file that keeps the network structure
            # 2) h5 file that tracks all extra information (also global indices, in case merging)
            # 3) BED file of regions to check functional enrichment

            



            # summary gml file - use a separate merge script that uses h5 file and gml file
            
            # figure out for dmim, what information required for dmim
            # for dmim, requires example_metadata, labels, motif indices


            # separate script for functional enriching and grouping results
                
            import ipdb
            ipdb.set_trace()
        
    return
