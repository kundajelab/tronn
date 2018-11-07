# description: build grammars from scans

import h5py
import logging




from tronn.interpretation.networks import build_co_occurrence_graph
from tronn.interpretation.networks import get_subgraphs_and_filter



def run(args):
    """build grammars from scan (scanmotifs or dmim)
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running motif scan")

    # get group keys from the sig pwms file
    with h5py.File(args.sig_pwms_file, "r") as hf:

        # todo standardize with results in the original h5 file
        group_keys = hf["cor_filt"].keys()

        # better set up: cor_filt/[target_key]/index/
        
        # need to extract the indices also....
        print group_keys

    # go through each group
    for key in group_keys:
        print key

        target_idx = int(key.split("-")[1])

        # build graph
        # here, either select co-occurrence or dmim
        graph = build_co_occurrence_graph(
            args.scan_file,
            "cor_filt/{}/pwms.sig".format(key),
            "TRAJ_LABELS",
            target_idx,
            sig_pwms_h5_file=args.sig_pwms_file,
            min_co_occurring_num=500)

        # get subgraphs
        subgraphs = get_subgraphs_and_filter(
            graph,
            3,
            500,
            500,
            #int(0.25*graph.attrs["min_edge_support"]),
            #int(0.5*graph.attrs["min_edge_support"]),
            0.3)

        all_nodes = []
        for subgraph in subgraphs:
            node_names = [node.name for node in subgraph.nodes]
            if len(node_names) > 1:
                print node_names
                all_nodes += node_names

        all_nodes = list(set(all_nodes))
        print all_nodes
        print len(all_nodes)

        import ipdb
        ipdb.set_trace()

        # save out
        # 1) summary and summary gml
        # 2) BED file
        # 3) new h5 file for each subset
        
        
        
    return
