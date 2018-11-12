#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import glob
import argparse

import pandas as pd
import networkx as nx

from tronn.interpretation.networks import get_bed_from_nx_graph
from tronn.interpretation.networks import stringize_nx_graph


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # required args
    parser.add_argument(
        "--grammars_dir",
        help="directory of gml files")
    parser.add_argument(
        "--annotation_type", default="great",
        help="annotation type (great, etc etc)")

    # out
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    parser.add_argument(
        "--prefix",
        help="prefix to output files")
    
    
    # parse args
    args = parser.parse_args()

    return args


def _run_great(bed_file, out_dir):
    """run great
    """
    run_cmd = "run_rgreat.R {} {}/{}".format(
        bed_file,
        out_dir,
        os.path.basename(bed_file).split(".bed")[0])
    print run_cmd
    os.system('GREPDB="{}"; /bin/bash -c "$GREPDB"'.format(run_cmd))
    
    return None


def _add_go_terms_to_graph(nx_graph, great_file, top_k=None):
    """add go terms into graph
    """
    enrichments = pd.read_table(great_file)

    # add in
    nx_graph.graph["GOids"] = enrichments["ID"].values.tolist()
    nx_graph.graph["GOnames"] = enrichments["name"].values.tolist()
    nx_graph.graph["GOqvals"] = enrichments["hyper_q_vals"].values.tolist()
    
    return None


def main():
    """run annotation
    """
    # set up args
    args = parse_args()

    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))

    # get all gml files and load in
    grammar_files = glob.glob("{}/*gml".format(args.grammars_dir))
    grammars = [nx.read_gml(grammar_file) for grammar_file in grammar_files]

    # and adjust examples
    for grammar in grammars:
        grammar.graph["examples"] = set(
            grammar.graph["examples"].split(","))
        for node_name in grammar.nodes():
            grammar.node[node_name]["examples"] = set(
                grammar.node[node_name]["examples"].split(","))
        for edge_name in grammar.edges():
            grammar.edge[edge_name]["examples"] = set(
                grammar.edge[edge_name]["examples"].split(","))

    # annotate
    if args.annotation_type == "great":
        # tricky part here is to use the BED files and then link back to gml files...
        # maybe just use the gml files and produce BED files on the fly
        for grammar_idx in xrange(len(grammars)):
            # make bed
            bed_file = "{}/{}.bed".format(
                args.out_dir,
                os.path.basename(grammar_files[grammar_idx]).split(".gml")[0])
            get_bed_from_nx_graph(grammars[grammar_idx], bed_file)
            
            # run great
            _run_great(bed_file, args.out_dir)
            
            # get terms back and save into file
            biol_terms_file = "{}.GO_Biological_Process.txt".format(bed_file.split(".bed")[0])
            _add_go_terms_to_graph(grammars[grammar_idx], biol_terms_file)

            # and write a new file out
            graph_file = "{}/{}.great_annotated.gml".format(
                args.out_dir,
                os.path.basename(grammar_files[grammar_idx]).split(".gml")[0])
            nx.write_gml(stringize_nx_graph(grammars[grammar_idx]), graph_file, stringizer=str)
            
    else:
        raise ValueError, "annotation type not implemented"
    
    return None



if __name__ == "__main__":
    main()
