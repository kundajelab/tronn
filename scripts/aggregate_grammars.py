#!/usr/bin/env python

"""
description: script to use to aggregate
grammars (subgraphs) that have matching features.

This can be:
  node(s) - shared node(s)
  edge(s) - shared edge(s)
  attribute(s) - shared graph attributes

will update example sets accordingly
"""

import os
import h5py
import glob
import argparse

import networkx as nx

from networkx.drawing.nx_agraph import graphviz_layout

from tronn.interpretation.networks import stringize_nx_graph
from tronn.interpretation.networks import add_graphics_theme_to_nx_graph
from tronn.interpretation.networks import apply_graphics_to_subgraphs


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="aggregate grammars")

    # required args
    parser.add_argument(
        "--grammar_dir",
        help="directory with grammar files (gml)")
    parser.add_argument(
        "--merge_type", default="graph",
        help="what type of merge - node, edge, graph")
    parser.add_argument(
        "--merge_attr",
        help="attribute key for merging")
    parser.add_argument(
        "--merge_expr", nargs="+",
        help="string to look for when merging")
    parser.add_argument(
        "--merge_all", action="store_true",
        help="merge all grammars in directory")
    parser.add_argument(
        "--filename_filter", default="",
        help="string to use for filtering filenames")
    
    # outputs
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="outputs directory")
    parser.add_argument(
        "--out_file", default="grammars.merged.gml",
        help="out file name")
    
    args = parser.parse_args()
    
    return args

# TODO track runs


def merge_graphs(graphs):
    """take multiple networkx graphs and merge them
    """
    # merge graph attributes
    graph_attrs = {
        "names": [],
        "examples": set(),
        "numexamples": 0}
    for graph in graphs:
        # update names, examples, num examples
        graph_attrs["names"].append(graph.graph["name"])
        graph_attrs["examples"] = graph_attrs["examples"].union(
            graph.graph["examples"])
        graph_attrs["numexamples"] = len(graph_attrs["examples"])
    
    # merge nodes
    all_nodes = {}
    for graph in graphs:
        for node_name, node_attrs in graph.nodes(data=True):
            if all_nodes.get(node_name) is not None:
                # update examples and numexamples
                all_nodes[node_name]["examples"] = all_nodes[node_name]["examples"].union(
                    node_attrs["examples"])
                all_nodes[node_name]["numexamples"] = len(
                    all_nodes[node_name]["examples"])
            else:
                all_nodes[node_name] = node_attrs

    # convert to node list
    node_list = [(node_name, all_nodes[node_name])
                 for node_name in all_nodes.keys()]

    # DO NOT MERGE EDGES
    edge_list = []
    for graph in graphs:
        edge_list += [edge for edge in graph.edges.data()]

    # produce new graph
    nx_graph = nx.MultiDiGraph(**graph_attrs)
    nx_graph.add_nodes_from(node_list)
    nx_graph.add_edges_from(edge_list)
    
    return nx_graph


def track_runs(args):
    """track command and github commit
    """
    # keeps track of restores (or different commands) in folder
    subcommand_name = "aggregate_grammars"
    num_restores = len(glob.glob('{0}/{1}.command*'.format(args.out_dir, subcommand_name)))
    logging_file = '{0}/{1}.command_{2}.log'.format(args.out_dir, subcommand_name, num_restores)
    
    # track github commit
    git_repo_path = os.path.dirname(os.path.realpath(__file__))
    os.system('echo "commit:" > {0}'.format(logging_file))
    os.system('git --git-dir={0}/.git rev-parse HEAD >> {1}'.format(
        git_repo_path.split("/scripts")[0], logging_file))
    os.system('echo "" >> {0}'.format(logging_file))
    
    # write out the command
    with open(logging_file, 'a') as f:
        f.write(' '.join(sys.argv)+'\n\n')
    
    return logging_file


def _setup_logs(args):
    """set up logging
    """
    logging_file = track_runs(args)
    reload(logging)
    logging.basicConfig(
        filename=logging_file,
        level=logging.DEBUG, # TODO ADJUST BEFORE RELEASE
        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    for arg in sorted(vars(args)):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    logging.info("")

    return


def main():
    """run merging
    """
    # set up args
    args = parse_args()

    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))

    args.merge_expr = " ".join(args.merge_expr)
    
    # get all gml files and load in
    grammar_files = glob.glob("{}/*{}*gml".format(args.grammar_dir, args.filename_filter))
    grammars = [nx.read_gml(grammar_file) for grammar_file in grammar_files]

    # and adjust examples
    for grammar in grammars:
        grammar.graph["examples"] = set(
            grammar.graph["examples"].split(","))
        for node_name in grammar.nodes():
            grammar.node[node_name]["examples"] = set(
                grammar.node[node_name]["examples"].split(","))
        for edge_name in grammar.edges():
            for edge_idx in xrange(len(grammar[edge_name[0]][edge_name[1]])):
                grammar[edge_name[0]][edge_name[1]][edge_idx]["examples"] = set(
                    grammar[edge_name[0]][edge_name[1]][edge_idx]["examples"].split(","))
    
    # select which ones to keep
    if args.merge_all:
        grammars_filt = grammars
    else:
        grammars_filt = []
        for grammar in grammars:
            # check which merge type and collect attributes
            if args.merge_type == "graph":
                attr_vals = [grammar.graph[args.merge_attr]]
            elif args.merge_type == "node":
                attr_vals = nx.get_node_attributes(grammar, args.merge_attr).values()
            elif args.merge_type == "edge":
                attr_vals = nx.get_edge_attributes(grammar, args.merge_attr).values()
            else:
                raise ValueError, "merge type requested not implemented"

            # check through the vals
            expr_present = False
            for attr_val in attr_vals:
                if args.merge_expr in attr_val:
                    expr_present = True

            # and save out if expr is present
            if expr_present:
                grammars_filt.append(grammar)

    # now merge
    merged_grammar = merge_graphs(grammars_filt)

    # get iniitial positions and add into grammar
    pos = graphviz_layout(merged_grammar, prog="dot")
    for key in pos.keys():
        coords = pos[key]
        pos[key] = {"x": coords[0], "y": -coords[1]}
    nx.set_node_attributes(merged_grammar, pos, "graphics") # note this is diff from v1 to v2 in networkx
    add_graphics_theme_to_nx_graph(merged_grammar)
    
    # write gml
    out_file = "{}/{}".format(args.out_dir, os.path.basename(args.out_file))
    nx.write_gml(stringize_nx_graph(merged_grammar), out_file, stringizer=str)

    
    # and then update the subgraphs with positions and write them out to new files too
    # ^ this is a separate function. given a master file of
    # master positions, apply those positions to other gml files.
    # needs to write out new ones to go along with the master gml file
    grammars_w_graphics = apply_graphics_to_subgraphs(merged_grammar, grammars_filt)

    for grammar in grammars_w_graphics:
        # save out
        out_file = "{}/{}.fixed_positions.gml".format(
            args.out_dir,
            grammar.graph["name"])
        nx.write_gml(stringize_nx_graph(grammar), out_file, stringizer=str)
    
    return


if __name__ == "__main__":
    main()
