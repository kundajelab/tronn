# description: code for doing network analyses

import os
import re
import h5py
import copy
import glob
import logging

import numpy as np
import pandas as pd
import networkx as nx

from tronn.util.h5_utils import AttrKeys
from tronn.util.h5_utils import copy_h5_dataset_slices
from tronn.util.utils import DataKeys


class Node(object):
    """node class"""

    def __init__(self, name=0, attrs={}):
        self.name = name
        self.attrs = attrs
        if "examples" in self.attrs.keys():
            self.attrs["num_examples"] = len(self.attrs["examples"])

            
    def get_tuple(self):
        # important for conversion to gml
        return (self.name, self.attrs)

    
        
class DirectedEdge(object):
    """edge class"""
    
    def __init__(self, start_node_name, end_node_name, attrs={}):
        self.name = "{}_{}".format(start_node_name, end_node_name)
        self.start_node_name = start_node_name
        self.end_node_name = end_node_name
        self.attrs = attrs
        if "examples" in self.attrs.keys():
            self.attrs["num_examples"] = len(self.attrs["examples"])

        
    def get_tuple(self):
        # important for conversion to gml
        return (self.start_node_name, self.end_node_name, self.attrs)

    

class MotifGraph(object):
    """network class for managing a directed motif network"""
    
    def __init__(self, nodes=[], edges=[], name=None, propagated=False):
        """initialize
        note that the nodes and edges are SHALLOW copies
        """
        self.name = name
        self.nodes = list(nodes)
        self.node_names = [node.name for node in self.nodes]
        self.edges = list(edges)
        self.propagated = propagated
        self.attrs = {} # graph attributes

        # update attrs with initial example set across nodes and edges
        self.attrs["examples"] = nodes[0].attrs.get("examples", set())
        for node in self.nodes:
            self.attrs["examples"] = self.attrs["examples"].intersection(
                node.attrs["examples"])
        for edge in self.edges:
            self.attrs["examples"] = self.attrs["examples"].intersection(
                edge.attrs["examples"])
        self.attrs["num_examples"] = len(self.attrs["examples"])
            
        # keep these! update local copies of examples,
        # that way when combining can just sum each.
        self.node_attrs = {} # "local" copy of attributes to adjust as needed
        self.edge_attrs = {} # "local" copy of attributes to adjust as needed
        
        # copy node attrs from nodes
        for node in self.nodes:
            self.node_attrs[node.name] = copy.deepcopy(node.attrs)
            self.node_attrs[node.name]["num_examples"] = len(self.attrs["examples"])
        # copy edge attrs from edges
        for edge in self.edges:
            self.edge_attrs[edge.name] = copy.deepcopy(edge.attrs)
            self.edge_attrs[edge.name]["num_examples"] = len(self.attrs["examples"])
            
            
    def deepcopy(self):
        """make a deep copy of the graph
        note that this only deep copies the ATTRIBUTES not the
        node and edge objects
        """
        new_net = MotifGraph(nodes=self.nodes, edges=self.edges)
        new_net.attrs = copy.deepcopy(self.attrs)
        new_net.node_attrs = copy.deepcopy(self.node_attrs)
        new_net.edge_attrs = copy.deepcopy(self.edge_attrs)
        return new_net

    
    def add_node(self, node):
        """first deep copy, then add the node to the new object
        """
        new_net = self.deepcopy()
        new_net.nodes.append(node)
        new_net.node_names.append(node.name)
        new_net.node_attrs[node.name] = copy.deepcopy(node.attrs)
        
        return new_net

    
    def add_edge(self, edge):
        """add edge (shallow copy), makes a new graph object
        """
        new_net = self.deepcopy()
        new_net.edges.append(edge)
        new_net.edge_attrs[edge.name] = copy.deepcopy(edge.attrs)
        
        return new_net

    
    def equals(self, other_net):
        """check if other net matches. checks nodes and edges
        """
        for node in self.nodes:
            if node not in other_net.nodes:
                return False
        for node in other_net.nodes:
            if node not in self.nodes:
                return False
        for edge in self.edges:
            if edge not in other_net.edges:
                return False
        for edge in other_net.edges:
            if edge not in self.edges:
                return False
        
        return True

    
    def get_node_by_id(self, node_name):
        """get the node object by id
        """
        for node in self.nodes:
            if node_name == node.name:
                return node

            
    def get_edge_by_id(self, edge_name):
        """get the node object by id
        """
        for edge in self.edges:
            if edge_name == edge.name:
                return edge

            
    def get_node_out_edges(self, node_name):
        """return the edges
        """
        node_edges = []
        for edge in self.edges:
            if node_name in edge.start_node_name:
                node_edges.append(edge)

        return node_edges

    
    def get_node_in_edges(self, node_name):
        """return edges coming into node
        """
        node_edges = []
        for edge in self.edges:
            if node_name in edge.end_node_name:
                node_edges.append(edge)

        return node_edges


    def update_examples(self, examples):
        """update across graph
        """
        # update the graph
        self.attrs["examples"] = examples
        self.attrs["num_examples"] = len(examples)

        # update node attrs
        for node_name in self.node_attrs.keys():
            self.node_attrs[node_name]["examples"] = examples
            self.node_attrs[node_name]["num_examples"] = len(examples)
            
        # update edge attrs
        for edge_name in self.edge_attrs.keys():
            self.edge_attrs[edge_name]["examples"] = examples
            self.edge_attrs[edge_name]["num_examples"] = len(examples)

        return None
    

    def write_gml(self, gml_file):
        """write out the gml file
        """
        # cleaned graph attributes
        clean_attrs = {}
        for key in self.attrs.keys():
            clean_key = key.replace("_", "")
            if not isinstance(self.attrs[key], (list, set, dict, np.ndarray)):
                clean_attrs[clean_key] = self.attrs[key]
        
        # instantiate graph with attributes
        nx_graph = nx.MultiDiGraph(**clean_attrs)
        
        # nodes with cleaned up attrs
        node_list = []
        for node_name in self.node_attrs.keys():
            clean_attrs = {}
            for key in self.node_attrs[node_name].keys():
                clean_key = key.replace("_", "")
                if not isinstance(self.node_attrs[node_name][key], (list, set, dict, np.ndarray)):
                    clean_attrs[clean_key] = self.node_attrs[node_name][key]
                    #if isinstance(clean_attrs[key], basestring):
                    #    clean_attrs[key] = clean_attrs[key].replace("_", "-")
            node_list.append((node_name, clean_attrs))

        # edges with cleaned up attrs
        edge_list = []
        for edge_name in self.edge_attrs.keys():
            clean_attrs = {}
            for key in self.edge_attrs[edge_name].keys():
                clean_key = key.replace("_", "")
                if not isinstance(self.edge_attrs[edge_name][key], (list, set, dict, np.ndarray)):
                    clean_attrs[clean_key] = self.edge_attrs[edge_name][key]
                    #if isinstance(clean_attrs[key], basestring):
                    #    clean_attrs[key] = clean_attrs[key].replace("_", "-")
            edge = self.get_edge_by_id(edge_name)
            edge_list.append(
                (edge.start_node_name, edge.end_node_name, clean_attrs))
            
        # add them in
        nx_graph.add_nodes_from(node_list)
        nx_graph.add_edges_from(edge_list)

        # write
        nx.write_gml(nx_graph, gml_file, stringizer=str)
        
        return None


def write_subgraph_to_bed(
        subgraph,
        example_metadata,
        bed_file,
        interval_key="active",
        merge=True):
    """convenience function to take indices, convert to regions,
    and save out as a bed file.

    NOTE THAT INDICES MUST MATCH THE EXAMPLE METADATA ORDER
    """
    subgraph_examples = sorted(list(subgraph.attrs["examples"]))
    grammar_metadata = example_metadata[subgraph_examples,0].tolist()
    with open(bed_file, "w") as fp:
        for region_metadata in grammar_metadata:
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
    

def build_nodes(
        sig_pwms,
        pwm_names,
        nodes=[],
        effects_mat=None,
        original_indices=None,
        indexing_key="driver_idx"):
    """helper function to build a nodes list
    use to build perturb nodes and response nodes
    
    Args:
      sig_pwms: np bool vector of whether the pwm is significant or not {M}
      pwm_names: list of the pwm names
      effects_mat: np array of effects per example {N, M}
      nodes: node list, if it already exists
    
    Returns:
      nodes: updated list of nodes
    """
    assert sig_pwms.shape[0] == len(pwm_names)
    
    # go through pwms
    pwm_indices = np.where(sig_pwms !=0)[0]
    for pwm_idx in pwm_indices:
        pwm_name = pwm_names[pwm_idx]
        
        # first create node or find it as needed
        if pwm_name not in [node.name for node in nodes]:
            node = Node(pwm_name, {indexing_key: pwm_idx})
            nodes.append(node)
        else:
            for old_node_idx in xrange(len(nodes)):
                if nodes[old_node_idx].name == pwm_name:
                    node = nodes[old_node_idx]
                    break
                
        # update node
        node.attrs[indexing_key] = pwm_idx
        if effects_mat is not None:
            indices_from_effects = np.where(effects_mat[:,pwm_idx] != 0)[0]
            true_indices = original_indices[indices_from_effects]
            node.attrs["examples"] = set(true_indices.tolist())

    logging.info("Built {} nodes".format(len(nodes)))

    return nodes


def build_edges_by_example_intersect(
        nodes,
        min_co_occurring_num):
    """build co-occurrence edges
    """
    edges = []
    for driver_node in nodes:
        if driver_node.attrs.get("driver_idx") is None:
            continue
        for responder_node in nodes:
            if responder_node.attrs.get("responder_idx") is None:
                continue

            # do not add if the reverse already exists (undirected graph)
            edge_exists = False
            for edge in edges:
                if "{}_{}".format(responder_node.name, driver_node.name) in edge.name:
                    edge_exists = True
            if edge_exists:
                continue

            # do not add if same node
            if driver_node.name == responder_node.name:
                continue
                
            # get overlap
            co_occurring_examples = driver_node.attrs["examples"].intersection(
                responder_node.attrs["examples"])

            # check that conditions met
            if len(co_occurring_examples) < min_co_occurring_num:
                continue
            
            # build edge with co-occurrence info
            edge = DirectedEdge(
                driver_node.name,
                responder_node.name,
                attrs={"examples": co_occurring_examples})
            edges.append(edge)
            
    logging.info("Built {} edges with min support {} regions each".format(
        len(edges), min_co_occurring_num))

    return edges


def build_co_occurrence_graph(
        h5_file,
        targets_key,
        target_idx,
        sig_pwms_h5_file,
        sig_pwms_key,
        data_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM,
        sig_only=True,
        min_positive=2,
        min_co_occurring_num=500):
    """build co-occurrence based graph
    """
    # load in sig pwms vector
    with h5py.File(sig_pwms_h5_file, "r") as hf:
        sig_pwms = hf[sig_pwms_key][:] # {M}
        
    # load in scores
    with h5py.File(h5_file, "r") as hf:
        scores = hf[data_key][:] # {N, task, M}
        targets = hf[targets_key][:] # {N, task}
        pwm_names = hf[data_key].attrs[AttrKeys.PWM_NAMES] # {M}
    
    # set up effects mat and original indices
    original_indices = np.arange(scores.shape[0])
    score_indices = np.where(targets[:,target_idx] > 0)[0]
    scores = scores[score_indices]
    original_indices = original_indices[score_indices]
    effects_mat = np.sum(scores > 0, axis=1) >= min_positive # {N}
    
    # first set up driver nodes
    num_drivers = np.sum(sig_pwms)
    nodes = build_nodes(
        sig_pwms,
        pwm_names,
        nodes=[],
        effects_mat=effects_mat,
        original_indices=original_indices,
        indexing_key="driver_idx")
    
    # then set up response nodes
    num_responders = np.sum(sig_pwms)
    if sig_only:
        nodes = build_nodes(
            sig_pwms,
            pwm_names,
            nodes=nodes,
            indexing_key="responder_idx")
    else:
        raise Exception, "not yet implemented"

    # build edges
    edges = build_edges_by_example_intersect(
        nodes,
        min_co_occurring_num)
    
    # and put together into a graph
    graph = MotifGraph(nodes, edges)

    return graph


def build_dmim_graph(
        h5_file,
        targets_key,
        target_idx,
        sig_pwms_h5_file,
        sig_pwms_key,
        sig_dmim_key,
        data_key=DataKeys.FEATURES,
        outputs_key=DataKeys.MUT_MOTIF_LOGITS,
        sig_only=True,
        min_positive=2,
        min_co_occurring_num=500):
    """using dmim outputs, build graph
    """
    # first determine what the global pwm vector was
    sig_pwm_root = sig_pwms_key.replace("/{}".format(DataKeys.PWM_SIG_ROOT), "")
    with h5py.File(sig_pwms_h5_file, "r") as hf:
        sig_pwm_keys = hf[sig_pwm_root].keys()
        for i in xrange(len(sig_pwm_keys)):
            key = sig_pwm_keys[i]
            if i == 0:
                sig_pwms = hf[sig_pwm_root][key][:]
            else:
                sig_pwms += hf[sig_pwm_root][key][:]
    global_sig_pwms = sig_pwms != 0

    # and set up correct subset of pwms
    with h5py.File(sig_pwms_h5_file, "r") as hf:
        sig_pwms = hf[sig_pwms_key][:] # {M}
    sig_pwms = sig_pwms[np.where(global_sig_pwms)[0]]
    keep_pwms = np.where(sig_pwms != 0)[0] # use this to filter on mutM

    sig_pwms = sig_pwms[keep_pwms]
    
    import ipdb
    ipdb.set_trace()

    # need to load in mut logits (nodes), scores (edges), adjacence mat (edges)
    # load in scores
    with h5py.File(h5_file, "r") as hf:
        scores = hf[data_key][:] # {N, mutM, task, M}
        outputs = hf[outputs_key][:] # {N, mutM, logit}
        targets = hf[targets_key][:] # {N, task}
        adjacency = hf[sig_dmim_key][:] # {mutM, responseM} <- square, but eventually not
        #pwm_names = hf[data_key].attrs[AttrKeys.PWM_NAMES] # {M}

    # now adjust for the specific task desired
    original_indices = np.arange(scores.shape[0])
    select_indices = np.where(targets[:,target_idx] > 0)[0]

    original_indices = original_indices[select_indices]
    scores = scores[select_indices]
    outputs = outputs[select_indices]

    # nodes: make nodes based on whether the motif had an effect
    # note! need to make this directional maybe
    effect_mat = np.any(outputs < 0, axis=2) # {N, mutM}

    # first set up driver nodes
    import ipdb
    ipdb.set_trace()
    
    num_drivers = np.sum(sig_pwms) # <- make sure correct pwms get read out here
    nodes = build_nodes(
        sig_pwms,
        pwm_names,
        nodes=[],
        effects_mat=effects_mat,
        original_indices=original_indices,
        indexing_key="driver_idx")
    
    # then set up response nodes
    num_responders = np.sum(sig_pwms)
    if sig_only:
        nodes = build_nodes(
            sig_pwms,
            pwm_names,
            nodes=nodes,
            indexing_key="responder_idx")
    else:
        raise Exception, "not yet implemented"

    

    #  edges: create edges based on adjacency {mutM, responseM}
    # and put examples on the edges if there was an effect



    return


    
def get_subgraphs(
        graph,
        subgraph,
        k=1,
        min_region_count=50,
        update_type="synergistic"):
    """start from a specific node and get all subgraphs
    that are size k (or less)

    NOTE: everything here is intersect (because any example in the graph
    must fulfill all criteria - node effects and edge effects)
    """
    # ignore empty graphs
    if k == 1:
        return []

    # get edges that can connect to subgraph but do not yet exist
    edges = []
    for edge in graph.edges:
        if edge in subgraph.edges:
            continue
        if edge.start_node_name in subgraph.node_names:
            edges.append(edge)
        elif edge.end_node_name in subgraph.node_names:
            edges.append(edge)
            
    # go through each edge and add
    new_subgraphs = []
    for edge in edges:

        # extract examples and intersect
        edge_examples = edge.attrs["examples"]
        intersected_examples = subgraph.attrs["examples"].intersection(
            edge_examples)

        # check conditions
        if len(intersected_examples) < min_region_count:
            continue
        
        # check if internal and not seen already
        if (edge.start_node_name in subgraph.node_names) and (edge.end_node_name in subgraph.node_names):
            new_subgraph = subgraph.add_edge(edge)
            new_subgraph.update_examples(intersected_examples)
            new_subgraphs.append(new_subgraph)
                
        # check if edge adds new child node into subgraph
        elif (edge.end_node_name not in subgraph.node_names):
            new_subgraph = subgraph.add_node(
                graph.get_node_by_id(edge.end_node_name))
            new_subgraph = new_subgraph.add_edge(edge)
            new_subgraph.update_examples(intersected_examples)
            new_subgraphs.append(new_subgraph)
            
            new_subgraphs += get_subgraphs(
                graph, new_subgraph, k-1, min_region_count=min_region_count)

        # check if edge adds new parent node into subgraph
        elif (edge.start_node_name not in subgraph.node_names):
            new_subgraph = subgraph.add_node(
                graph.get_node_by_id(edge.start_node_name))
            new_subgraph = new_subgraph.add_edge(edge)
            new_subgraph.update_examples(intersected_examples)
            new_subgraphs.append(new_subgraph)
            
            new_subgraphs += get_subgraphs(
                graph, new_subgraph, k-1, min_region_count=min_region_count)

        # something went wrong otherwise
        else:
            print "something wrong!"
            import ipdb
            ipdb.set_trace()
    
    # now check to make sure they pass criteria
    filtered_subgraphs = []
    for subgraph in new_subgraphs:
        region_num = len(subgraph.attrs["examples"])
        if region_num > min_region_count:
            filtered_subgraphs.append(subgraph)

    return filtered_subgraphs


def get_subgraphs_and_filter(
        graph,
        max_subgraph_size,
        forward_pass_min_region_count,
        min_region_count,
        max_subgraph_overlap):
    """get subgraphs and filter to get unique subgraphs
    """
    # collect subgraphs
    subgraphs = []
    print len(graph.nodes)
    for node in graph.nodes:
        print node.name
        seed_subgraph = MotifGraph(nodes=[node], edges=[])
        subgraphs.append(seed_subgraph)
        subgraphs += get_subgraphs(
            graph,
            seed_subgraph,
            k=max_subgraph_size,
            min_region_count=forward_pass_min_region_count)
    print "Initial subgraphs with k={} and min size {}: {}".format(
        max_subgraph_size, forward_pass_min_region_count, len(subgraphs))

    # 1) and remove redundant
    filtered_subgraphs = []
    for i in xrange(len(subgraphs)):
        exists = False
        for j in xrange(len(filtered_subgraphs)):
            if subgraphs[i].equals(filtered_subgraphs[j]):
                exists = True
        if not exists:
            filtered_subgraphs.append(subgraphs[i])
    print "After removing redundant subgraphs: {}".format(len(filtered_subgraphs))
    subgraphs = filtered_subgraphs

    # 2) now more stringent region number check
    filtered_subgraphs = []
    for subgraph in subgraphs:
        region_num = len(subgraph.attrs["examples"])
        if region_num > min_region_count:
            filtered_subgraphs.append(subgraph)
    print "After second size filter with size > {}: {}".format(
        min_region_count, len(filtered_subgraphs))
    subgraphs = filtered_subgraphs
    
    # 3) and remove those that basically are the same (in terms of region overlap)
    filtered_subgraphs = []
    for i in xrange(len(subgraphs)):
        differs = True
        for j in xrange(len(filtered_subgraphs)):
            # get examples
            examples_i = subgraphs[i].attrs["examples"]
            examples_j = filtered_subgraphs[j].attrs["examples"]
            # calculate jaccard index
            intersect = examples_i.intersection(examples_j)
            union = examples_i.union(examples_j)
            fract_overlap = len(intersect) / float(len(union))
            # if high overlap, keep set with more edges (more specific)
            if fract_overlap > max_subgraph_overlap:
                if len(subgraphs[i].edges) > len(filtered_subgraphs[j].edges):
                    filtered_subgraphs[j] = subgraphs[i]
                differs = False
                break
        if differs:
            filtered_subgraphs.append(subgraphs[i])
    print "After removing high overlap subgraphs: {}".format(len(filtered_subgraphs))
    subgraphs = filtered_subgraphs

    # and name them
    for subgraph_idx in xrange(len(subgraphs)):
        subgraphs[subgraph_idx].attrs["name"] = "grammar-{}".format(subgraph_idx)

    return subgraphs


def sort_subgraphs_by_output_strength(subgraphs, h5_file, target_key):
    """given a subgraph set, re-rank the order based on summed output strength
    this gives a nice way to cut the subgraphs to only keep a manageable set
    
    extract the MAX val.

    """
    summed_outputs = np.zeros((len(subgraphs)))
    
    with h5py.File(h5_file, "r") as hf:
        outputs = hf[target_key][:] # {N, task}
    
    for i in xrange(len(subgraphs)):
        subgraph_examples = sorted(list(subgraphs[i].attrs["examples"]))
        subgraph_outputs = outputs[subgraph_examples]
        summed_outputs[i] = np.mean(np.amax(subgraph_outputs, axis=1))
        subgraphs[i].attrs["logit_max"] = summed_outputs[i]

        # separately (consider whether to factor this out),
        # also attach the pattern to the subgraph (to visualize later)
        subgraphs[i].attrs["logits"] = np.mean(subgraph_outputs, axis=0)
        subgraphs[i].attrs["logits_str"] = ";".join(
            [str(val) for val in np.mean(subgraph_outputs, axis=0).tolist()])
        
        
    sort_indices = np.argsort(-summed_outputs) # negative to reverse the sort
    sorted_subgraphs = np.array(subgraphs)[sort_indices].tolist()

    # rename
    for subgraph_idx in xrange(len(sorted_subgraphs)):
        sorted_subgraphs[subgraph_idx].attrs["name"] = "grammar-{}".format(subgraph_idx)
    
    return sorted_subgraphs


def build_subgraph_per_example_array(
        h5_file,
        subgraphs,
        target_key,
        out_key):
    """
    """
    # use h5 file to determine how many examples there were originally
    with h5py.File(h5_file, "r") as hf:
        num_examples = hf[target_key].shape[0]

    # set up array
    subgraph_present = np.zeros((num_examples, len(subgraphs)))
    
    # go through subgraphs
    for subgraph_idx in xrange(len(subgraphs)):
        subgraph = subgraphs[subgraph_idx]
        examples = sorted(list(subgraph.attrs["examples"]))
        subgraph_present[examples,subgraph_idx] = 1

    # and save it back into the h5 file
    with h5py.File(h5_file, "a") as hf:
        if hf.get(out_key) is not None:
            del hf[out_key]
        hf.create_dataset(out_key, data=subgraph_present)
    
    return




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

  

def get_motif_hierarchies(
        h5_file,
        adjacency_mat_key,
        data_key,
        metadata_key=DataKeys.SEQ_METADATA,
        sig_pwms_key=DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL,
        logits_key=DataKeys.LOGITS,
        mut_logits_key=DataKeys.MUT_MOTIF_LOGITS,
        pwm_names_attr_key=AttrKeys.PWM_NAMES,
        extra_keys=[],
        num_sig_tasks=3,
        max_subgraph_size=3,
        max_subgraph_overlap=0.3,
        forward_pass_min_region_count=100, # this is too manual still, need to adjust
        min_region_count=500): # 100/500 for mid, 50/400 for early, 
    """extract motif hierarchies
    """
    # load in necessary data
    with h5py.File(h5_file, "r") as hf:
        adjacency_matrix = hf[adjacency_mat_key][:] # for edges
        data = hf[data_key][:] # node mutational results
        example_metadata = hf[metadata_key][:] # for BED files
        sig_pwms = hf[sig_pwms_key][:]
        
        # TODO need to have mut names attr and pwm names attr
        mut_pwm_names = hf[adjacency_mat_key].attrs[pwm_names_attr_key] 
        logits = np.expand_dims( hf[logits_key][:], axis=1)
        mut_logits = hf[mut_logits_key][:]

        # set up delta logits
        delta_logits = np.subtract(mut_logits, logits)
    
    # TODO - change these later
    data = data[:,:,:,np.where(sig_pwms > 0)[0]]
    response_pwm_names = mut_pwm_names

    # TODO - seperate the below into a separate function?
    # set up the mut nodes
    nodes = []
    for name_idx in xrange(len(mut_pwm_names)):
        pwm_name = mut_pwm_names[name_idx]
        had_effect = set(np.where(np.sum(delta_logits[:,name_idx,:] < 0, axis=1) > num_sig_tasks)[0].tolist())
        node = Node(pwm_name, {"mut_idx": name_idx, "examples": had_effect})
        nodes.append(node)

    # set up response nodes
    for name_idx in xrange(len(response_pwm_names)):
        pwm_name = response_pwm_names[name_idx]
        if pwm_name not in [node.name for node in nodes]:
            node = Node(pwm_name, {"response_idx": name_idx})
            nodes.append(node)
        else:
            for old_node_idx in xrange(len(nodes)):
                if nodes[old_node_idx].name == pwm_name:
                    nodes[old_node_idx].attrs["response_idx"] = name_idx

    # set up the edges
    num_mut_motifs = adjacency_matrix.shape[0]
    num_response_motifs = adjacency_matrix.shape[2]
    edges = []
    for mut_idx in xrange(num_mut_motifs):
        for response_idx in xrange(num_response_motifs):
            # first require effects existed in at least num_sig_tasks
            if np.sum(adjacency_matrix[mut_idx, :, response_idx] != 0) < num_sig_tasks:
                continue
            
            # set up an edge
            start_node = mut_pwm_names[mut_idx]
            end_node = response_pwm_names[response_idx]

            # determine direction
            overall_edge_effect = np.sum(data[:,mut_idx,:,response_idx])
            if overall_edge_effect < 0:
                sig_in_tasks = np.sum(data[:,mut_idx,:,response_idx] < 0, axis=1)
            else:
                sig_in_tasks = np.sum(data[:,mut_idx,:,response_idx] > 0, axis=1)
                
            # get subset and attach to attrs
            edge_examples = np.where(sig_in_tasks >= num_sig_tasks)[0].tolist()
            attr_dict = {"examples": set(edge_examples)}            

            # attach the list of examples with both 
            edge = DirectedEdge(start_node, end_node, attrs=attr_dict)
            edges.append(edge)

    # put into network
    graph = MotifGraph(nodes, edges)
    
    # get subgraphs
    subgraphs = get_subgraphs_and_filter(
        graph,
        max_subgraph_size,
        forward_pass_min_region_count,
        min_region_count,
        max_subgraph_overlap)


    # currently HERE in refactoring
    
    # save out subgraphs
    grammar_file = "{}.grammars.txt".format(
        h5_file.split(".h5")[0])
    with open(grammar_file, "w") as fp:
        fp.write(
            "grammar_idx\t{}\tedges\n".format(
                "\t".join([node.name for node in graph.nodes])))
    grammar_example_masks = np.zeros((data.shape[0], len(subgraphs)))
    
    grammar_idx = 0
    for subgraph in subgraphs:
        subgraph.name = "grammar-{}".format(grammar_idx)

        # TODO see if this is obsolete now (probably not)
        # 1) save out a table of nodes and edges to plot
        subgraph_node_vector = []
        for node in graph.nodes:
            if node in subgraph.nodes:
                subgraph_node_vector.append("1")
            else:
                subgraph_node_vector.append("0")
                
        edge_list = []
        for edge in subgraph.edges:
            for start_node_idx in xrange(len(graph.nodes)):
                for end_node_idx in xrange(len(graph.nodes)):
                    interaction_name = "{}_{}".format(
                        graph.nodes[start_node_idx].name,
                        graph.nodes[end_node_idx].name)
                    if edge.name == interaction_name:
                        out_name = "{},{}".format(
                            graph.nodes[start_node_idx].name,
                            graph.nodes[end_node_idx].name)
                        edge_list.append(out_name)
                        break
                    
        with open(grammar_file, "a") as fp:
            fp.write(
                "{}\t{}\t{}\n".format(
                    grammar_idx,
                    "\t".join(subgraph_node_vector),
                    ";".join(edge_list)))

        # 2) also write out the bed file of examples
        bed_file = "{}.grammar-{}.bed".format(
            h5_file.split(".h5")[0],
            grammar_idx)
        grammar_examples = sorted(list(subgraph.get_covered_region_set()[1]))
        grammar_metadata = example_metadata[grammar_examples,0].tolist()
        with open(bed_file, "w") as fp:
            for region_metadata in grammar_metadata:
                region = region_metadata.split(";")[1].split("=")[1] # TODO adjust here - just active region
                chrom = region.split(":")[0]
                start = region.split(":")[1].split("-")[0]
                stop = region.split("-")[1]
                fp.write("{}\t{}\t{}\n".format(chrom, start, stop))

        # 3) write out h5 file of subset, to go into downstream analyses
        # should i actually just put this into dmim file?
        grammar_example_masks[grammar_examples, grammar_idx] = 1

        # 4) also need to write out manifold...
        # use subgraph nodes and mut idx
        sig_pwm_indices = np.where(sig_pwms != 0)[0]
        grammar_indices = []
        for node in subgraph.nodes:
            grammar_indices.append(node.attrs["mut_idx"])
        grammar_indices = sorted(grammar_indices)
        grammar_indices = sig_pwm_indices[grammar_indices]
        grammar_pwm_mask = np.zeros(sig_pwms.shape)
        grammar_pwm_mask[grammar_indices] = 1
        
        grammar_manifold_file = "{}.grammar-{}.manifold.h5".format(
            h5_file.split(".h5")[0],
            grammar_idx)
        with h5py.File(grammar_manifold_file, "w") as out:
            out.create_dataset(sig_pwms_key, data=grammar_pwm_mask)
            out[sig_pwms_key].attrs[pwm_names_attr_key] = mut_pwm_names

        # update idx
        grammar_idx += 1

    # save out grammar mask to h5 results file
    with h5py.File(h5_file, "a") as hf:
        if hf.get(DataKeys.GRAMMAR_LABELS) is not None:
            del hf[DataKeys.GRAMMAR_LABELS]
        hf.create_dataset(
            DataKeys.GRAMMAR_LABELS, data=grammar_example_masks)
    
    # finally, write out merged gml results
    global_gml = "{}.global.gml".format(
        h5_file.split(".h5")[0])
    merge_subgraphs_and_write_gml(subgraphs, global_gml, ignore_singles=True)

    # get subsets
    terms = [
        "keratinocyte differentiation",
        "epithelial cell proliferation",
        "epidermal cell division",
        "epidermis development",
        "proliferation",
        "cell cycle arrest",
        "negative regulation of cell cycle",
        "stem cell proliferation",
        "hair",
        "junction assembly",
        "regulation of cell motility",
        "extracellular matrix",
        "stem cell"]
    
    subsets = subset_by_functional_enrichments(
        subgraphs, terms, os.path.dirname(h5_file), "{}/great".format(os.path.dirname(h5_file)))
    
    # and save out
    for term in terms:
        if len(subsets[term]) == 0:
            continue
        out_file = "{}.{}.gml".format(
            h5_file.split(".h5")[0],
            term.replace(" ", "_").replace(".", "_"))
        merge_subgraphs_and_write_gml(subsets[term], out_file, ignore_singles=True)
        
    import ipdb
    ipdb.set_trace()

    return subgraphs
        

def subset_by_functional_enrichments(subgraphs, terms, bed_dir, out_dir):
    """get subsets by functional enrichment
    """
    import ipdb
    ipdb.set_trace()
    
    # run functional enrichment tool
    # ie, rgreat
    if False:
        bed_files = sorted(glob.glob("{}/*grammar*bed".format(bed_dir)))
        for bed_file in bed_files:
            run_cmd = "run_rgreat.R {} {}/{}".format(
                bed_file,
                out_dir,
                os.path.basename(bed_file).split(".bed")[0])
            print run_cmd
            os.system(run_cmd)

    # then, for each subgraph, check enrichments
    enrichment_files = glob.glob("{}/*Biol*txt".format(out_dir))
    subsets = {}
    for term in terms:
        subsets[term] = []
        for subgraph in subgraphs:
            for enrichment_file in enrichment_files:
                if ".{}.".format(subgraph.name) in enrichment_file:
                    # read
                    with open(enrichment_file, "r") as fp:
                        data = " ".join(fp.readlines()[0:11])
                    if term in data:
                        subsets[term].append(subgraph)
                        
    import ipdb
    ipdb.set_trace()
    
    return subsets
