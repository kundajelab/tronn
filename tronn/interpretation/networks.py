# description: code for doing network analyses

import os
import re
import h5py
import copy
import glob
import logging
import itertools

import numpy as np
import pandas as pd
import networkx as nx

from tronn.util.h5_utils import AttrKeys
from tronn.util.h5_utils import copy_h5_dataset_slices
from tronn.util.utils import DataKeys


def get_bed_from_nx_graph(graph, bed_file, interval_key="active", merge=True):
    """get BED file from nx examples
    """
    examples = list(graph.graph["examples"])

    with open(bed_file, "w") as fp:
        for region_metadata in examples:
            interval_types = region_metadata.split(";")
            interval_types = dict([
                interval_type.split("=")[0:2]
                for interval_type in interval_types])
            interval_string = interval_types[interval_key]

            chrom = interval_string.split(":")[0]
            start = interval_string.split(":")[1].split("-")[0]
            stop = interval_string.split("-")[1]
            fp.write("{}\t{}\t{}\n".format(chrom, start, stop))

    if merge:
        tmp_bed_file = "{}.tmp.bed".format(bed_file.split(".bed")[0])
        os.system("mv {} {}".format(bed_file, tmp_bed_file))
        os.system("cat {} | sort -k1,1 -k2,2n | bedtools merge -i stdin > {}".format(
            tmp_bed_file, bed_file))
        os.system("rm {}".format(tmp_bed_file))
    
    return None


def _build_nodes(
        all_sig_mut_outputs,
        example_metadata,
        pwm_indices,
        pwm_names,
        sig_reduce_type="any"):
    """build nodes. when building, build with following attributes:
    1) examples
    2) global_idx (based on pwmvector)
    """
    if sig_reduce_type == "all":
        reduce_fn = np.all
    elif sig_reduce_type == "any":
        reduce_fn = np.any
    else:
        raise ValueError("reduce fn is not valid!")

    nodes = []
    for pwm_i in range(len(pwm_indices)):
        pwm_global_idx = pwm_indices[pwm_i]
        pwm_name = pwm_names[pwm_global_idx]
        pwm_sig_mut_outputs = all_sig_mut_outputs[:,pwm_i] # {N, logit}
        pwm_sig_mut_outputs_reduced = reduce_fn(pwm_sig_mut_outputs != 0, axis=1) # {N}
        example_indices = np.where(pwm_sig_mut_outputs_reduced)[0]
        node_examples = set(example_metadata[example_indices])
        node_attrs = {
            "examples": node_examples,
            "numexamples": len(node_examples),
            "mutidx": pwm_i,
            "pwmidx": pwm_global_idx}
        nodes.append((pwm_name, node_attrs))
        logging.debug("{} has {} sig mutant examples".format(pwm_name, len(node_examples)))
    logging.info("created {} nodes".format(len(nodes)))
        
    return nodes


def _build_co_occurrence_edges(
        graph,
        min_region_num,
        sig_reduce_type="any",
        keep_grammars=[]):
    """build co-occurrence edges
    """
    nodes = list(graph)
    edges = []
    for node1_idx in range(len(nodes)):
        node1_name = nodes[node1_idx]
        for node2_idx in range(len(nodes)):
            node2_name = nodes[node2_idx]
            if node1_idx >= node2_idx:
                continue
            
            # check co-occurrence
            node1_examples = graph.nodes[node1_name]["examples"]
            node2_examples = graph.nodes[node2_name]["examples"]
            intersect = node1_examples.intersection(node2_examples)

            # check if should keep
            keep_edge = False
            for grammar in keep_grammars:
                if set([node1_name, node2_name]).issubset(set(grammar)):
                    keep_edge = True
                    logging.info("from keep_grammars, keeping {},{} (size {})".format(
                        node1_name, node2_name, len(intersect)))
            
            # add an edge if passes min region num OR passes exception num
            if (len(intersect) > min_region_num) or keep_edge:
                # make an edge
                edge_attrs = {
                    "examples": intersect,
                    "numexamples": len(intersect),
                    "edgetype": "co_occurrence"}
                edge = (node1_name, node2_name, edge_attrs)
                edges.append(edge)

    logging.info("created {} co-occurrence edges".format(len(edges)))

    return edges


def _build_synergy_edges(
        graph,
        dmim_scores,
        example_metadata,
        min_region_num,
        sig_reduce_type="any"):
    """get synergy edges
    """
    if sig_reduce_type == "all":
        reduce_fn = np.all
    elif sig_reduce_type == "any":
        reduce_fn = np.any
    else:
        raise ValueError("reduce fn is not valid!")
    
    nodes = list(graph)
    edges = []
    for driver_node_idx in range(len(nodes)):
        driver_node_name = nodes[driver_node_idx]
        for responder_node_idx in range(len(nodes)):
            responder_node_name = nodes[responder_node_idx]

            # for now, ignore self loops
            if driver_node_idx == responder_node_idx:
                continue
            
            # check synergy
            driver_mut_idx = graph.nodes[driver_node_name]["mutidx"]
            responder_global_idx = graph.nodes[responder_node_name]["pwmidx"]
            dmim_subset = dmim_scores[:,driver_mut_idx,:,responder_global_idx]
            dmim_results = reduce_fn(dmim_subset != 0, axis=1)
            example_indices = np.where(dmim_results)[0]
            examples = set(example_metadata[example_indices])

            if len(examples) > min_region_num:
                # make an edge
                edge_attrs = {
                    "examples": examples,
                    "numexamples": len(examples),
                    "edgetype": "directed"}
                edge = (driver_node_name, responder_node_name, edge_attrs)
                edges.append(edge)

    logging.info("created {} synergy edges".format(len(edges)))
        
    return edges


def build_full_graph(
        dmim_h5_file,
        sig_pwms_indices,
        sig_mut_logits_key=DataKeys.MUT_MOTIF_LOGITS_SIG,
        dmim_scores_key=DataKeys.DMIM_SCORES_SIG,
        sig_responses=DataKeys.FEATURES,
        min_positive_tasks=2,
        min_region_num=1000,
        keep_grammars=[]):
    """build full graph with co-occurrence and synergy edges
    """
    # instantiate empty graph
    graph = nx.MultiDiGraph()

    # 1) build nodes
    # first extract sig mut logits to make nodes
    with h5py.File(dmim_h5_file, "r") as hf:
        example_metadata = hf[DataKeys.SEQ_METADATA][:,0]
        sig_mut_outputs = hf[sig_mut_logits_key][:]
        pwm_names = hf[DataKeys.FEATURES].attrs[AttrKeys.PWM_NAMES]
    assert sig_mut_outputs.shape[1] == len(sig_pwms_indices)
    
    # build nodes
    nodes = _build_nodes(
        sig_mut_outputs,
        example_metadata,
        sig_pwms_indices,
        pwm_names,
        sig_reduce_type="any")
    graph.add_nodes_from(nodes)
    
    # 2) build co-occurrence edges
    co_occurrence_edges = _build_co_occurrence_edges(
        graph, min_region_num, sig_reduce_type="any",
        keep_grammars=keep_grammars)
    graph.add_edges_from(co_occurrence_edges)

    # 3) build synergy edges
    # TODO somewhere before this do a signifiance test
    with h5py.File(dmim_h5_file, "r") as hf:
        dmim_scores = hf[dmim_scores_key][:]
    synergy_edges = _build_synergy_edges(
        graph,
        dmim_scores,
        example_metadata,
        min_region_num,
        sig_reduce_type="any")
    graph.add_edges_from(synergy_edges)
    
    return graph


def make_graph_examples_consistent(graph):
    """make sure all the examples are consistent
    """
    node_names = list(graph)
    start_node = node_names[0]
    examples = graph.nodes[start_node]["examples"]

    # intersect all nodes
    for node_name, node_examples in graph.nodes(data="examples"):
        examples = examples.intersection(node_examples)

    # intersect all edges
    for start_node_name, end_node_name, edge_examples in graph.edges(data="examples"):
        examples = examples.intersection(edge_examples)
    num_examples = len(examples)
        
    # replace
    for node, data in graph.nodes(data=True):
        data["examples"] = examples
        data["numexamples"] = num_examples

    for start_node, end_node, data in graph.edges(data=True):
        data["examples"] = examples
        data["numexamples"] = num_examples

    graph.graph["examples"] = examples
    graph.graph["numexamples"] = num_examples

    return graph


def remove_edge_and_check_consistent(graph, min_region_count):
    """remove edge from graph, check if meets all criteria
    """
    graph_edges = list(graph.edges)
    new_graphs = []
    for start_node, end_node, _ in graph_edges:
        new_graph = graph.copy()
        new_graph.remove_edge(start_node, end_node)
        if not nx.is_weakly_connected(new_graph):
            continue
        new_graph = make_graph_examples_consistent(new_graph)
        if new_graph.graph["numexamples"] == graph.graph["numexamples"]:
            continue
        if new_graph.graph["numexamples"] > min_region_count:
            new_graphs.append(new_graph)
        new_graphs += remove_edge_and_check_consistent(new_graph, min_region_count)
        
    return new_graphs


def get_size_k_subgraphs(graph, min_region_count, k=1, keep_grammars=[]):
    """for size k, get all unique subgraphs
    """
    total = 0
    subgraphs = []
    current_subgraph_len = 0
    for node_subset in itertools.combinations(graph, k):
        subgraph = graph.subgraph(node_subset)

        # check if all connected
        if nx.is_weakly_connected(subgraph):

            # check num examples
            subgraph = make_graph_examples_consistent(subgraph.copy())
            if subgraph.graph["numexamples"] > min_region_count:
                subgraphs.append(subgraph)
            else:
                # check keep grammars
                keep_subgraph = False
                for keep_grammar in keep_grammars:
                    if len(set(subgraph.nodes).difference(set(keep_grammar))) == 0:
                        keep_subgraph = True
                if keep_subgraph:
                    logging.info("from keep grammars, keeping subgraph {}".format(subgraph.nodes))
                    subgraphs.append(subgraph)
                
            # now recursively remove edges and check subgraphs for any new ones
            other_subgraphs = remove_edge_and_check_consistent(
                subgraph.copy(), min_region_count)
            subgraphs += other_subgraphs

        total += 1
    
    return subgraphs


def get_maxsize_k_subgraphs(graph, min_region_count, k=3, keep_grammars=[]):
    """get all subgraphs of size k
    """
    subgraphs = []
    for i in range(k):
        subgraphs += get_size_k_subgraphs(graph, min_region_count, i+1, keep_grammars)
        logging.info("Found {} subgraphs for size {}".format(len(subgraphs), i+1))
        
    return subgraphs


def attach_data_summary(subgraphs, dmim_h5_file, key):
    """attach relevant details to subgraphs
    """
    # extract the data given the key
    with h5py.File(dmim_h5_file, "r") as hf:
        data = hf[key][:] # {N, logit/signal/label}
        example_metadata = hf[DataKeys.SEQ_METADATA][:,0]

    # for each subgraph, attach to graph
    gml_key = key.replace("-", "").replace(".", "").replace("_", "")
    for subgraph in subgraphs:
        examples = sorted(list(subgraph.graph["examples"]))
        example_indices = np.where(np.isin(example_metadata, examples))[0]
        subgraph_data = np.mean(data[example_indices], axis=0)
        subgraph.graph[gml_key] = subgraph_data

    return subgraphs


def attach_mut_logits(subgraphs, dmim_h5_file):
    """attach delta logits to subgraphs
    """
    # delta logits
    with h5py.File(dmim_h5_file, "r") as hf:
        logits = np.expand_dims(hf["logits.norm"][:], axis=1)
        mut_logits = hf[DataKeys.MUT_MOTIF_LOGITS][:]
        delta_logits = np.subtract(mut_logits, logits)
        example_metadata = hf[DataKeys.SEQ_METADATA][:,0]

    # for each subgraph, attach to nodes
    for subgraph in subgraphs:
        examples = sorted(list(subgraph.graph["examples"]))
        example_indices = np.where(np.isin(example_metadata, examples))[0]
        subgraph_delta_logits = delta_logits[example_indices]

        for node, data in subgraph.nodes(data=True):
            data["deltalogits"] = np.median(subgraph_delta_logits[:,data["mutidx"]], axis=0)

    return subgraphs


def write_bed_from_graph(graph, bed_file, interval_key="active", merge=True):
    """take the examples and write out a BED file
    """
    grammar_examples = list(graph.graph["examples"])
    with open(bed_file, "w") as fp:
        for region_metadata in grammar_examples:
            interval_types = region_metadata.split(";")
            interval_types = dict([
                interval_type.split("=")[0:2]
                for interval_type in interval_types])
            interval_string = interval_types[interval_key]

            chrom = interval_string.split(":")[0]
            start = interval_string.split(":")[1].split("-")[0]
            stop = interval_string.split("-")[1]
            fp.write("{}\t{}\t{}\n".format(chrom, start, stop))

    if merge:
        tmp_bed_file = "{}.tmp.bed".format(bed_file.split(".bed")[0])
        os.system("mv {} {}".format(bed_file, tmp_bed_file))
        os.system("cat {} | sort -k1,1 -k2,2n | bedtools merge -i stdin > {}".format(
            tmp_bed_file, bed_file))
        os.system("rm {}".format(tmp_bed_file))

    return None


def stringize_nx_graph(nx_graph):
    """preparatory function for writing out to gml
    """
    # graph attributes
    for key in nx_graph.graph.keys():
        if isinstance(nx_graph.graph[key], (list, set, np.ndarray)):
            nx_graph.graph[key] = ",".join([
                str(val) for val in list(nx_graph.graph[key])])

    # node attributes
    for node_name, node_attrs in nx_graph.nodes(data=True):
        for key in node_attrs.keys():
            if isinstance(nx_graph.nodes[node_name][key], (list, set, np.ndarray)):
                nx_graph.nodes[node_name][key] = ",".join([
                    str(val) for val in nx_graph.nodes[node_name][key]])
        # adjust node name for nice output in cytoscape
        new_node_name = re.sub(r"HCLUST.\d+_", "", node_name)
        new_node_name = new_node_name.replace(".UNK.0.A", "")
        nx_graph.nodes[node_name]["name"] = new_node_name
                
    # edge attributes
    for start_node, end_node in nx_graph.edges():
        for edge_idx in xrange(len(nx_graph[start_node][end_node])):
            edge_attrs = nx_graph[start_node][end_node][edge_idx]
            for key in edge_attrs.keys():
                if isinstance(edge_attrs[key], (list, set, np.ndarray)):
                    nx_graph[start_node][end_node][edge_idx][key] = ",".join([
                        str(val) for val in nx_graph[start_node][end_node][edge_idx][key]])
                    
    return nx_graph


def add_graphics_theme_to_nx_graph(
        nx_graph,
        edge_color=None,
        node_size_factor=50,
        edge_size_factor=500):
    """adjust nodes and edges
    """
    # node size, stroke
    for node_name, node_attrs in nx_graph.nodes(data=True):

        node_size = nx_graph.nodes[node_name]["numexamples"] / float(node_size_factor)
        
        #node_size = nx_graph.nodes[node_name]["numexamples"] / float(nx_graph.graph["numexamples"])
        #node_size *= node_size_factor
        
        graphics = {
            "type": "ellipse",
            "w": node_size,
            "h": node_size,
            "fill": "#FFFFFF",
            "outline": "#000000",
            "width": 1.0,
            "fontSize": 14
        }

        if nx_graph.nodes[node_name].get("graphics") is not None:
            nx_graph.nodes[node_name]["graphics"].update(graphics)
        else:
            nx_graph.nodes[node_name]["graphics"] = graphics

    # edges
    for start_node, end_node in nx_graph.edges():
        for edge_idx in xrange(len(nx_graph[start_node][end_node])):

            edge_width = nx_graph[start_node][end_node][edge_idx]["numexamples"] / float(
                edge_size_factor)
            
            #edge_width = nx_graph[start_node][end_node][edge_idx]["numexamples"] / float(
            #    nx_graph.graph["numexamples"])
            #edge_width *= edge_size_factor 

            graphics = {
                "type": "arc",
                "width": edge_width,
                "targetArrow": "delta"
            }
            
            if edge_color is not None:
                graphics["fill"] = edge_color

            if nx_graph[start_node][end_node][edge_idx].get("graphics") is not None:
                nx_graph[start_node][end_node][edge_idx]["graphics"].update(graphics)
            else:
                nx_graph[start_node][end_node][edge_idx]["graphics"] = graphics

        
            
    return None





def apply_graphics_to_subgraphs(master_graph, other_graphs):
    """assuming loaded graphs (in networkx format)
    get positions from the master graph and apply to the other graphs
    """
    # get graphics
    node_graphics = nx.get_node_attributes(master_graph, "graphics")
    #edge_graphics = nx.get_edge_attributes(master_graph, "graphics")

    #print edge_graphics
    
    # apply
    for other_graph in other_graphs:
        nx.set_node_attributes(other_graph, node_graphics, "graphics")
        #nx.set_edge_attributes(other_graph, edge_graphics, "graphics")
        
    return other_graphs



















# SUPER OLD








def get_clean_pwm_name(pwm_name):
    """convenience function
    """
    pwm_name = re.sub(r"HCLUST.\d+_", "", pwm_name)
    pwm_name = pwm_name.replace(".UNK.0.A", "")
    pwm_name = re.sub(r"\d+$", "", pwm_name)
    
    return pwm_name


def write_summary_metadata_file(subgraphs, metadata_file):
    """
    """
    with open(summary_metadata_file, "w") as fp:
        fp.write("grammar_index\t{}\tedges\n".format(
            "\t".join([node.name for node in graph.nodes])))
    for i in xrange(len(sorted_subgraphs)):
        with open(summary_metadata_file, "w") as fp:
            # write code to make a node vector
            # code to make edge string
            pass

    return



def merge_subgraphs_and_write_gml(subgraphs, gml_file, ignore_singles=True):
    """merge the list of subgraphs and write out a gml
    """
    # nodes: go through and get all nodes and update example set
    node_dict = {}
    for subgraph in subgraphs:
        if ignore_singles:
            if len(subgraph.nodes) <= 1:
                continue
        for node_name in subgraph.node_names:
            clean_node_name = get_clean_pwm_name(node_name)
            node_examples = set(subgraph.node_attrs[node_name]["examples"])
            if clean_node_name in node_dict.keys():
                # adjust numbers
                node_dict[clean_node_name][1]["examples"] = node_examples.union(
                    node_dict[clean_node_name][1]["examples"])
                node_dict[clean_node_name][1]["num_examples"] = len(
                    node_dict[clean_node_name][1]["examples"])
            else:
                # add new
                node_dict[clean_node_name] = (
                    clean_node_name,
                    copy.deepcopy(subgraph.node_attrs[node_name]))
                
    # then adjust here
    nodes = []
    for node_name in node_dict.keys():
        node_attrs = node_dict[node_name][1]
        clean_attrs = {}
        for key in node_attrs.keys():
            new_key = key.replace("_", "") # for networkx gml specs
            if not isinstance(node_attrs[key], (list, set, dict)):
                clean_attrs[new_key] = node_attrs[key]
        nodes.append((node_name, clean_attrs))
    
    # edges: collect all and add an attribute based on which subgraph
    edges = []
    for subgraph in subgraphs:
        for edge in subgraph.edges:
            edge_attrs = subgraph.edge_attrs[edge.name]
            clean_start_node_name = get_clean_pwm_name(edge.start_node_name)
            clean_end_node_name = get_clean_pwm_name(edge.end_node_name)
            clean_attrs = {"subgraph": subgraph.name}
            for key in edge_attrs.keys():
                new_key = key.replace("_", "") # for networkx gml specs
                if not isinstance(edge_attrs[key], (list, set, dict)):
                    clean_attrs[new_key] = edge_attrs[key]
            edges.append(
                (clean_start_node_name,
                 clean_end_node_name,
                 clean_attrs))
            
    # make graph
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    # write out
    nx.write_gml(graph, gml_file, stringizer=str)
    
    return None

  
