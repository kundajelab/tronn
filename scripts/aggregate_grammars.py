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

import h5py
import glob
import argparse

import networkx as nx


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
        "--merge_expr",
        help="string to look for when merging")
    parser.add_argument(
        "--merge_all", action="store_true",
        help="merge all grammars in directory")

    # outputs
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="outputs directory")
    
    args = parser.parse_args()
    
    return args

# TODO track runs


def merge_graphs(graphs):
    """take multiple networkx graphs and merge them
    """
    # merge graph attributes
    graph_attrs = {
        "names": "",
        "examples": set(),
        "numexamples": 0}
    for graph in graphs:
        # update names, examples, num examples
        graph_attrs["names"] += graph.graph["name"]
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
    
    # then merge edges
    all_edges = {}
    for graph in graphs:
        for edge_name, edge_attrs in graph.edges(data=True):
            if all_edges.get(edge_name) is not None:
                # update examples and numexamples
                all_edges[edge_name]["examples"] = all_edges[edge_name]["examples"].union(
                    edge_attrs["examples"])
                all_edges[edge_name]["numexamples"] = len(
                    all_edges[edge_name]["examples"])
            else:
                all_edges[edge_name] = edge_attrs

    # convert to edge list
    edge_list = [
        (all_edges[edge_name]["start_node"],
         all_edges[edge_name]["end_node"],
         all_edges[edge_name]) for edge_name in all_edges.keys()]

    # produce new graph
    nx_graph = nx.MultiDiGraph(**graph_attrs)
    nx_graph.add_nodes_from(node_list)
    nx_graph.add_edges_from(edge_list)
    
    return nx_graph



def main():
    """run merging
    """
    # set up args
    args = parse_args()

    # get all gml files and load in
    grammar_files = glob.glob("{}/*gml".format(args.grammar_dir))
    grammars = [nx.read_gml(grammar_file) for grammar_file in grammar_files]

    # and adjust examples
    for grammar in grammars:
        grammar.graph["examples"] = set(
            grammar.graph["examples"].split(","))
        for node_name in grammar.nodes():
            print node_name
            grammar.node[node_name]["examples"] = set(
                grammar.node[node_name]["examples"].split(","))
        for edge_name in grammar.edges():
            print edge_name
            grammar.edge[edge_name]["examples"] = set(
                grammar.edge[edge_name]["examples"].split(","))
    
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
                    
    import ipdb
    ipdb.set_trace()

    # now merge
    # merge nodes: take two lists of nodes. if any are the same,
    # merge those and add up examples (update examples, num examples)
    merged_grammar = merge_graphs(grammars_filt)

    # TODO adjust attributes before writing out!


    # TODO use a layout to set initial good positions
    
    
    # produces ONE gml.
    nx.write_gml(merged_grammar, "out_file.gml", stringizer=str)


    # and then update the subgraphs with positions and write them out to new files too
    # ^ this is a separate function. given a master file of
    # master positions, apply those positions to other gml files.

    
    import ipdb
    ipdb.set_trace()
    
    
    return


if __name__ == "__main__":
    main()
