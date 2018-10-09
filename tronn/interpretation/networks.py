# description: code for doing network analyses

import re
import h5py
import copy

import numpy as np
import pandas as pd
import networkx as nx

from tronn.stats.nonparametric import run_delta_permutation_test

from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys


class Node(object):
    """node class"""

    def __init__(self, name=0, attrs={}):
        self.name = name
        self.attrs = attrs
        if "examples" in self.attrs.keys():
            self.attrs["num_examples"] = len(self.attrs["examples"])

        
    def get_tuple(self):
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
        return (self.start_node_name, self.end_node_name, self.attrs)

    

class MotifGraph(object):
    """network class for managing a directed motif network"""
    
    def __init__(self, nodes=[], edges=[], name=None, propagated=False):
        """initialize
        note that the nodes and edges are SHALLOW copies
        """
        # TODO consider also saving as dict? or should it be a set?
        self.name = name
        self.nodes = list(nodes)
        self.nodes_dict = dict([(node.name, node) for node in self.nodes])
        self.node_names = [node.name for node in self.nodes]
        
        self.edges = list(edges)
        self.edges_dict = dict([(edge.name, edge) for edge in self.edges])
        
        self.propagated = propagated
        self.attrs = {} # graph attributes
        self.node_attrs = {} # "local" copy of attributes to adjust as needed
        self.edge_attrs = {} # "local" copy of attributes to adjust as needed

        # copy node attrs from nodes
        for node in self.nodes:
            self.node_attrs[node.name] = copy.deepcopy(node.attrs)
            
        # copy edge attrs from edges
        for edge in self.edges:
            self.edge_attrs[edge.name] = copy.deepcopy(edge.attrs)

            
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

    
    def get_node_tuple_DEPRECATE(self, node_name):
        """get tuple for networkx
        """
        cleaned_attrs = {}
        #for key in self.node_attrs[node_name]:
        #    if not isinstance(self.node_attrs[node_name][key], (list, set, dict)):
        #        cleaned_attrs[key] = self.node_attrs[node_name][key]
                
        return (node_name, self.node_attrs[node_name])
    
    
    def add_edge(self, edge):
        """add edge (shallow copy), makes a new graph object
        """
        new_net = self.deepcopy()
        new_net.edges.append(edge)
        new_net.edge_attrs[edge.name] = copy.deepcopy(edge.attrs)
        
        return new_net

    
    def get_edge_tuple_DEPRECATE(self, edge_name):
        """for networkx
        """
        cleaned_attrs = {}
        #for key in self.edge_attrs[edge_name]:
        #    if not isinstance(self.edge_attrs[edge_name][key], (list, set, dict)):
        #        cleaned_attrs[key] = self.edge_attrs[edge_name][key]
        
        return (edge_name, self.edge_attrs[edge_name])

    
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

    
    def merge_DEPRECATE(self, other_net):
        """merge two instances 
        NOTE: this is a copy, but not a deep copy
        """
        nodes = list(set(self.nodes + other_net.nodes))
        edges = list(set(self.edges + other_net.edges))
        new_net = MotifNet(nodes, edges)
        return new_net

    
    def get_node_by_id(self, node_name):
        """get the node object by id
        """
        return self.nodes_dict[node_name]
        #for node in self.nodes:
        #    if node.name == node_name:
        #        return node

            
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


    def get_leaf_nodes(self):
        """extract all nodes with no OUT edges
        """
        leaf_node_names = []
        for node in self.nodes:
            edges = self.get_node_out_edges(node.name)
            if len(edges) == 0:
                leaf_node_names.append(node.name)
                
        return leaf_node_names


    def get_covered_region_set(self):
        """for the corrected (all info fully propagated) graph,
        go to leafs and collect all regions
        """
        leaf_node_names = self.get_leaf_nodes()
        region_set = set([])
        for node_name in leaf_node_names:
            region_set = region_set.union(
                self.node_attrs[node_name]["examples"])

        region_num = len(region_set)
            
        return region_num, region_set


    def update_node_examples(self, node_name, examples):
        """update node attrs
        """
        try:
            self.node_attrs[node_name]["examples"] = examples
            self.node_attrs[node_name]["num_examples"] = len(examples)
        except:
            import ipdb
            ipdb.set_trace()

        return None

    
    def update_edge_examples(self, edge_name, examples):
        """update node attrs
        """
        self.edge_attrs[edge_name]["examples"] = examples
        self.edge_attrs[edge_name]["num_examples"] = len(examples)

        return None

    
    def propagate_up(
            self, node_name, transform_fn, seen_nodes=[]):
        """from a starting node, propagate up and
        adjust graph attributes
        """
        # get parent node ids
        parent_node_names = []
        in_edges = self.get_node_in_edges(node_name)
        for edge in in_edges:
            parent_node_names.append(edge.start_node_name)

        for parent_node_name in parent_node_names:
            if parent_node_name not in seen_nodes:
                # note that trasnform fn must only take node name
                transform_fn(parent_node_name)
                seen_nodes.append(parent_node_name)
                self.propagate_up(
                    parent_node_name,
                    transform_fn,
                    seen_nodes=seen_nodes)

        return None

    
    def propagate_down(
            self, node_name, transform_fn, seen_nodes=[]):
        """from a starting node, propagate up and
        adjust each node using transform fn
        """
        # get parent node ids
        child_node_names = []
        out_edges = self.get_node_out_edges(node_name)
        for edge in out_edges:
            child_node_names.append(edge.end_node_name)

        for child_node_name in child_node_names:
            if child_node_name not in seen_nodes:
                # note that trasnform fn must only take node name
                transform_fn(child_node_name)
                seen_nodes.append(child_node_name)
                self.propagate_down(
                    child_node_name,
                    transform_fn,
                    seen_nodes=seen_nodes)
                
        return None


    def update_examples_and_propagate(self, edge, edge_type="internal_edge"):
        """update the graph based on addition of internal edge
        """
        assert edge.start_node_name in self.node_names
        assert edge.end_node_name in self.node_names
        
        # setup - get examples that have edge and both (orig) node effects
        # from original node/edge sets
        edge_examples = edge.attrs["examples"]
        edge_examples = edge_examples.intersection(
            self.get_node_by_id(edge.start_node_name).attrs["examples"])
        edge_examples = edge_examples.intersection(
            self.get_node_by_id(edge.end_node_name).attrs["examples"])

        # get subgraph example sets
        start_node_examples = self.node_attrs[edge.start_node_name]["examples"]
        end_node_examples = self.node_attrs[edge.end_node_name]["examples"]
        
        def get_union(set1, set2):
            return set1.union(set2)

        def get_intersect(set1, set2):
            return set1.intersection(set2)

        if edge_type == "internal_edge":
            # internal edge - union on both
            upstream_fn = get_union
            downstream_fn = get_union
            
        elif edge_type == "frontier_edge":
            upstream_fn = get_intersect
            downstream_fn = get_intersect

        elif edge_type == "out_edge":
            upstream_fn = get_union
            downstream_fn = get_intersect
            
        elif edge_type == "in_edge":
            upstream_fn = get_intersect
            downstream_fn = get_union

        else:
            print "WRONG"
            quit()
            
        # update upstream node
        new_upstream_examples = upstream_fn(edge_examples, start_node_examples)
        self.update_node_examples(edge.start_node_name, new_upstream_examples)
        
        # and propagate
        def transform_fn(node_name):
            self.update_node_examples(node_name, new_upstream_examples)
            return None
        self.propagate_up(edge.start_node_name, transform_fn)

        # after updating upstream, update downstream accordingly
        new_downstream_examples = new_upstream_examples.intersection(edge_examples)

        # TODO is this where I update the edge examples?
        self.update_edge_examples(edge.name, new_downstream_examples)
        
        # update downstream node - union
        #new_downstream_examples = downstream_fn(edge_examples, end_node_examples)
        new_downstream_examples = downstream_fn(new_downstream_examples, end_node_examples)
        self.update_node_examples(edge.end_node_name, new_downstream_examples)
        
        # and propagate
        def transform_fn(node_name):
            self.update_node_examples(node_name, new_downstream_examples)
            return None
        self.propagate_down(edge.end_node_name, transform_fn)

        # check
        if False:
            print [node.name for node in self.nodes]
            for node in self.nodes:
                print "{}\t{}".format(node.name, len(self.attrs["node_examples"][node.name]))
            region_num, _ = self.get_covered_region_set()
            print region_num
        
        return None


    def write_gml(self, gml_file):
        """write out the gml file
        """
        nx_graph = nx.MultiDiGraph()

        # nodes with cleaned up attrs
        node_list = []
        for node in self.nodes:
            clean_attrs = {}
            for key in node.attrs.keys():
                if not isinstance(node.attrs[key], (list, set, dict)):
                    clean_attrs[key] = node.attrs[key]
            node_list.append((node.name, clean_attrs))

        # edges with cleaned up attrs
        edge_list = []
        for edge in self.edges:
            clean_attrs = {}
            for key in edge.attrs.keys():
                if not isinstance(edge.attrs[key], (list, set, dict)):
                    clean_attrs[key] = node.attrs[key]
            edge_list.append(
                (edge.start_node_name, edge.end_node_name, clean_attrs))

        # add them in
        nx_graph.add_nodes_from(node_list)
        nx_graph.add_edges_from(edge_list)

        # write
        nx.write_gml(nx_graph, gml_file, stringizer=str)
        
        return None

    
    def get_adjacency_matrix(self, ordered_node_ids):
        """
        """

        return
    

def get_subgraphs(
        graph,
        subgraph,
        k=1,
        min_region_count=50):
    """start from a specific node and get all subgraphs
    that are size k (or less?)

    NOTE: everything here is intersect (because any example in the graph
    must fulfill all criteria - node effects and edge effects)
    """
    # ignore empty graphs
    if k == 1:
        return []

    # get relevant edges
    edges = []
    for edge in graph.edges:
        if edge in subgraph.edges:
            continue
        if edge.start_node_name in subgraph.node_names:
            edges.append(edge)
        elif edge.end_node_name in subgraph.node_names:
            edges.append(edge)
    
    # consider all edges at the same time
    new_subgraphs = []
    for edge in edges:
        
        # get edge examples out in prep
        edge_examples = edge.attrs["examples"]
        
        # check if internal and not seen already
        if (edge.start_node_name in subgraph.node_names) and (edge.end_node_name in subgraph.node_names):
            new_subgraph = subgraph.add_edge(edge)
            new_subgraph.update_examples_and_propagate(edge, edge_type="frontier_edge")
            new_subgraphs.append(new_subgraph)
                
        # check if edge adds new child node into subgraph
        elif (edge.end_node_name not in subgraph.node_names):
            new_subgraph = subgraph.add_node(
                graph.get_node_by_id(edge.end_node_name))
            new_subgraph = new_subgraph.add_edge(edge)
            new_subgraph.update_examples_and_propagate(edge, edge_type="frontier_edge")

            new_subgraphs.append(new_subgraph)
            new_subgraphs += get_subgraphs(
                graph, new_subgraph, k-1, min_region_count=min_region_count)

        # check if edge adds new parent node into subgraph
        elif (edge.start_node_name not in subgraph.node_names):
            new_subgraph = subgraph.add_node(
                graph.get_node_by_id(edge.start_node_name))
            new_subgraph = new_subgraph.add_edge(edge)
            new_subgraph.update_examples_and_propagate(edge, edge_type="frontier_edge")
            
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
        region_num, _ = subgraph.get_covered_region_set()
        if region_num > min_region_count:
            filtered_subgraphs.append(subgraph)

    return filtered_subgraphs


def merge_subgraphs_and_write_gml(subgraphs, gml_file):
    """merge the list of subgraphs and write out a gml
    """
    import ipdb
    ipdb.set_trace()

    # nodes: go through and get all nodes and update
    # the num regions attribute
    node_dict = {}
    for subgraph in subgraphs:
        for node_name in subgraph.node_names:
            #node_name = re.sub(r"HCLUST.\d+_", "", node_name)
            
            node_examples = subgraph.node_attrs[node_name]["examples"]
            if node_name in node_dict.keys():
                # adjust numbers
                # TODO this is wrong, need to do union on examples
                node_dict[node_name][1]["examples"] = node_examples.union(
                    node_dict[node_name][1]["examples"])
                node_dict[node_name][1]["num_examples"] = len(
                    node_dict[node_name][1]["examples"])
            else:
                # add new
                node_dict[node_name] = (node_name, subgraph.node_attrs[node_name])
                
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
                
    import ipdb
    ipdb.set_trace()
    
    # edges: collect all and add an attribute based on which subgraph
    edges = []
    for subgraph in subgraphs:
        for edge in subgraph.edges:
            edge_attrs = subgraph.edge_attrs[edge.name]
            clean_attrs = {"subgraph": subgraph.name}
            for key in edge_attrs.keys():
                new_key = key.replace("_", "") # for networkx gml specs
                if not isinstance(edge_attrs[key], (list, set, dict)):
                    clean_attrs[new_key] = edge_attrs[key]
            edges.append(
                (edge.start_node_name,
                 edge.end_node_name,
                 clean_attrs))
            
    import ipdb
    ipdb.set_trace()
            
    # make graph
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    # write out
    nx.write_gml(graph, gml_file, stringizer=str)
    
    return None

    
def get_path_dmim_effects(path, net, data):
    """extract the mut effects, multiplicatively
    """
    effects = np.ones((data.shape[0], data.shape[2])) # {N, task}
    for i in xrange(1, len(path)):
        mut_node_idx = net.get_node_by_id(path[i-1]).attrs["mut_idx"]
        response_node_idx = net.get_node_by_id(path[i]).attrs["response_idx"]
        edge_effects = data[:,mut_node_idx,:,response_node_idx] # {N, task}
        effects *= np.abs(edge_effects)
    effects = np.mean(effects, axis=0) # {task}
    
    return effects


def get_path_logit_effects(path, net, data):
    """extract the logit effects, multiplicatively
    """
    effects = np.ones((data.shape[0], data.shape[2]))
    for i in xrange(len(path)):
        mut_node_idx = net.get_node_by_id(path[i]).attrs["mut_idx"]
        effects *= np.abs(data[:,mut_node_idx,:]) # {N, task}
    effects = np.mean(effects, axis=0) # {task}

    return effects



def get_path_example_logits(path, net, logits, mut_logits):
    """go through the path and get the mut results and orig logits
    """
    task_idx = 0
    effects = np.zeros((logits.shape[0], 1+len(path)))
    effects[:,0] = logits[:,task_idx]
    for i in xrange(len(path)):
        mut_node_idx = net.get_node_by_id(path[i]).attrs["mut_idx"]
        effects[:,i+1] = mut_logits[:,mut_node_idx,task_idx]

    return effects



def get_mut_effect_results(path, net, delta_logits, out_file, task_idx=0):
    """get a matrix of effects
    """
    effects_df = pd.DataFrame()
    for i in xrange(len(path)):
        mut_node_idx = net.get_node_by_id(path[i]).attrs["mut_idx"]
        effects = delta_logits[:,mut_node_idx,task_idx] # {N}

        # save out
        effects_df[path[i]] = effects

    effects_df.to_csv(out_file, sep="\t")

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
        forward_pass_min_region_count=400,
        min_region_count=600): # 400/600 for early/mid, 200/400 for late
    """extract motif hierarchies
    """
    # load in datasets
    with h5py.File(h5_file, "r") as hf:
        adjacency_matrix = hf[adjacency_mat_key][:]
        data = hf[data_key][:]
        example_metadata = hf[metadata_key][:]
        sig_pwms = hf[sig_pwms_key][:]
        # TODO need to have mut names attr and pwm names attr
        mut_pwm_names = hf[adjacency_mat_key].attrs[pwm_names_attr_key] 
        logits = np.expand_dims( hf[logits_key][:], axis=1)
        mut_logits = hf[mut_logits_key][:]
        signal = np.expand_dims(hf[logits_key][:], axis=1) # vs ATAC_SIGNAL
        signal_indices = [0,1,2,3,4,5,6,9,10] #12
        
        # also track actual ATAC signal and H3K27ac signal
        # only allow 2D or 1D matrices
        extra_datasets = []
        for key in extra_keys:
            extra_data = hf[key][:]
            assert len(extra_data.shape) == 2
            extra_datasets.append(extra_data)

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

            # TODO fix this (in relation to previous analysis)
            if np.sum(adjacency_matrix[mut_idx, :, response_idx] != 0) < 3:
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
                
            # get subset
            edge_examples = np.where(sig_in_tasks >= num_sig_tasks)[0].tolist() # keep as indices until end

            # TODO - also filter for mutational effect?
            # how to do this, this is likely a node attribute to filter on?
            attr_dict = {"examples": set(edge_examples)}            
            
            # TODO don't add the edge if there isn't mutational output effect on both sides
            # for the subset?
            # Note that this is dealt with elsewhere (at the nodes)
            if False:
                edge_mut_logits = np.subtract(
                    mut_logits[edge_examples,response_idx],
                    logits[edge_examples]) 
                sig_logits = run_delta_permutation_test(edge_mut_logits)

                if np.any(sig_logits == False):
                    print "here"
                    import ipdb
                    ipdb.set_trace()
            
            # attach the list of examples with both 
            edge = DirectedEdge(start_node, end_node, attrs=attr_dict)
            edges.append(edge)

    # put into network
    graph = MotifGraph(nodes, edges)

    # summary of heuristics used below:
    # k size subgraphs - 3 <- this one is ok
    # min region count (init) - 300 <- this one is about compute time, and back-of-envelope says this is fine
    # min region count (final) - 800 <- this one maybe should be adjusted?
    # min overlap - 0.5 <- how to choose this... maybe don't do this?
    # probably just do a small param search
    k = 3
    init_min_count = 50
    final_min_count = 300 # 300
    max_overlap = 0.3
    
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
            k=k,
            min_region_count=init_min_count)
    print len(subgraphs)

    # and remove redundant
    filtered_subgraphs = []
    for i in xrange(len(subgraphs)):
        exists = False
        for j in xrange(len(filtered_subgraphs)):
            if subgraphs[i].equals(filtered_subgraphs[j]):
                exists = True
        if not exists:
            filtered_subgraphs.append(subgraphs[i])
    print len(filtered_subgraphs)
    subgraphs = filtered_subgraphs

    # now more stringent region number check
    filtered_subgraphs = []
    for subgraph in subgraphs:
        region_num, _ = subgraph.get_covered_region_set()
        if region_num > final_min_count:
            filtered_subgraphs.append(subgraph)
    print len(filtered_subgraphs)
    subgraphs = filtered_subgraphs
    
    # and remove those that basically are the same (in terms of region overlap?)
    # make sure the logic here makes sense
    filtered_subgraphs = []
    for i in xrange(len(subgraphs)):
        differs = True
        for j in xrange(len(filtered_subgraphs)):
            _, examples_i = subgraphs[i].get_covered_region_set()
            _, examples_j = filtered_subgraphs[j].get_covered_region_set()

            # calculate jaccard index
            intersect = examples_i.intersection(examples_j)
            union = examples_i.union(examples_j)
            fract_overlap = len(intersect) / float(len(union))

            if fract_overlap > max_overlap:
                # keep the more specific set
                if len(subgraphs[i].edges) > len(filtered_subgraphs[j].edges):
                #if len(examples_i) < len(examples_j):
                    filtered_subgraphs[j] = subgraphs[i]
                differs = False
                break
            
        if differs:
            filtered_subgraphs.append(subgraphs[i])
    print len(filtered_subgraphs)
    subgraphs = filtered_subgraphs

    # check
    if False:
        for subgraph in filtered_subgraphs:
            print [node.name for node in subgraph.nodes]
            #for edge in subgraph.edges:
            #    print ">>>", edge.get_tuple()[0:2]
            
            if len(subgraph.edges) > 2:
                print "^more edges"

        print ""

        # upstream centric view
        print "UPSTREAM"
        for node in graph.nodes:

            node_centric_subgraphs = []
            for subgraph in filtered_subgraphs:
                if node in subgraph.nodes:
                    for edge in subgraph.edges:
                        if node.name in edge.get_tuple()[0]:
                            node_centric_subgraphs.append(subgraph)
            if len(node_centric_subgraphs) == 0:
                continue
            print node.name
            for subgraph in node_centric_subgraphs:
                #print " ".join([node.name for node in subgraph.nodes])
                for edge in subgraph.edges:
                    print ">>>", " ".join(edge.get_tuple()[0:2])
            print ""
        
        # downstream centric view
        print "DOWNSTREAM"
        for node in graph.nodes:

            node_centric_subgraphs = []
            for subgraph in filtered_subgraphs:
                if node in subgraph.nodes:
                    for edge in subgraph.edges:
                        if node.name in edge.get_tuple()[1]:
                            node_centric_subgraphs.append(subgraph)
            if len(node_centric_subgraphs) == 0:
                continue
            print node.name
            for subgraph in node_centric_subgraphs:
                #print " ".join([node.name for node in subgraph.nodes])
                for edge in subgraph.edges:
                    print ">>>", " ".join(edge.get_tuple()[0:2])
            print ""

        import ipdb
        ipdb.set_trace()
            
        # node centric view
        for node in graph.nodes:

            node_centric_subgraphs = []
            for subgraph in filtered_subgraphs:
                if node in subgraph.nodes:
                    node_centric_subgraphs.append(subgraph)

            if len(node_centric_subgraphs) == 0:
                continue
                    
            print node.name
            for subgraph in node_centric_subgraphs:
                #print " ".join([node.name for node in subgraph.nodes])
                print "..."
                for edge in subgraph.edges:
                    print ">>>", " ".join(edge.get_tuple()[0:2])
            print ""

    # save out subgraphs
    grammar_file = "{}.grammars.txt".format(
        h5_file.split(".h5")[0])
    with open(grammar_file, "w") as fp:
        fp.write(
            "grammar_idx\t{}\tedges\n".format(
                "\t".join([node.name for node in graph.nodes])))

    grammar_idx = 0
    for subgraph in subgraphs:
        subgraph.name = "grammar-{}".format(grammar_idx)

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
                        #edge_list.append("{}-{}".format(start_node_idx, end_node_idx))
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
        grammar_examples = sorted(list(subgraph.get_covered_region_set()[1]))
        bed_file = "{}.grammar-{}.bed".format(
            h5_file.split(".h5")[0],
            grammar_idx)
        grammar_metadata = example_metadata[grammar_examples,0].tolist()
        with open(bed_file, "w") as fp:
            for region_metadata in grammar_metadata:
                region = region_metadata.split(";")[1].split("=")[1] # TODO adjust here - just active region
                chrom = region.split(":")[0]
                start = region.split(":")[1].split("-")[0]
                stop = region.split("-")[1]
                fp.write("{}\t{}\t{}\n".format(chrom, start, stop))

        # 3) write out the delta effects, need to write out the PAIRWISE
        # since the tests will have to be pairwise
        
        
        grammar_idx += 1

    # finally, write out merged gml results
    merge_subgraphs_and_write_gml(subgraphs, "test.gml")
        
    import ipdb
    ipdb.set_trace()

    quit()
    return None
        


def old():
    # TODO make this a seperate function?
    # run the network to get hierarchical paths
    # is each path a tuple ([path], [examples])?
    # TODO keep paths as indices?
    # TODO also keep edges as indices?
    paths = []
    for node in net.nodes:

        # add in singletons and where they exist
        node_idx = node.attrs["mut_idx"]
        #node_mut_effects = np.sum(delta_logits[:,node_idx,:] != 0, axis=1)
        #node_examples = set(np.where(node_mut_effects > 0)[0].tolist())

        node_examples = set(node.attrs["examples"])
        
        current_path = ([node.name], node_examples)

        if len(node_examples) > min_region_count:
            paths.append(current_path)
        
        #all_examples = set(range(example_metadata.shape[0]))
        #all_examples = set(example_metadata[:,0].tolist())
        paths += net.get_paths(
            node.name,
            forward_pass_min_region_count,
            current_path=current_path,
            checked_nodes=[node.name])

    # TODO saving paths out - separate function out?
    # for each path, want to write out a BED file (and write a metadata file)
    metadata_file = "{}.hierarchies.metadata.txt".format(
        h5_file.split(".h5")[0])
    with open(metadata_file, "w") as fp:
        fp.write(
            "path_idx\tpath_hierarchy\tnum_regions\tdmim\tdelta_logit\t{}\n".format(
                "\t".join(extra_keys)))

    timepoints_file = "{}.hierarchies.LOGITS.txt".format(
        h5_file.split(".h5")[0])

    global_path_id = 0
    for path_idx in xrange(len(paths)):
        path = paths[path_idx]
        path_hierarchy = "->".join(path[0])
        path_indices = sorted(path[1]) # TODO save this out in the h5 file
        path_data = data[path_indices]
        path_logits = logits[path_indices]
        path_mut_logits = mut_logits[path_indices]
        path_delta_logits = delta_logits[path_indices]

        if len(path_indices) < min_region_count:
            # do not write out
            continue
        
        # get the aggregate effects over the path
        # TODO check the sign on the multiplications
        path_dmim_effects = get_path_dmim_effects(path[0], net, path_data)
        path_delta_logits_effects = get_path_logit_effects(path[0], net, path_delta_logits)

        # testing
        mut_effects_file = "{}.path-{}.effects_per_example.txt".format(
            h5_file.split(".h5")[0],
            global_path_id)
        get_mut_effect_results(path[0], net, path_delta_logits, mut_effects_file, task_idx=0)

        # TODO just choose max val?
        # TODO save out dy/dx
        path_dmim_max = path_dmim_effects[
            np.argmax(np.abs(path_dmim_effects))]
        path_delta_logits_max = path_delta_logits_effects[
            np.argmax(np.abs(path_delta_logits_effects))]

        # for key in extra keys
        path_extra_results = []
        path_extra_results_examples = []
        for key_idx in xrange(len(extra_keys)):
            key = extra_keys[key_idx]
            extra_data = extra_datasets[key_idx][path_indices]
            #extra_data_summary = np.mean(np.amax(extra_data, axis=1) > 0.5)
            #print extra_data_summary.shape
            extra_data_summary = np.median(np.amax(extra_data, axis=1))
            path_extra_results.append(extra_data_summary)
            path_extra_results_examples.append(np.amax(extra_data, axis=1))
            
        # write out to file
        with open(metadata_file, "a") as fp:
            fp.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                global_path_id,
                path_hierarchy,
                len(path[1]),
                path_dmim_max,
                path_delta_logits_max,
                "\t".join([str(val)
                           for val in path_extra_results])))
        
        bed_file = "{}.path-{}.bed".format(
            h5_file.split(".h5")[0],
            global_path_id)

        # TODO for each region, keep track of extra information
        # ie ATAC signal, H3K27ac, dmim score, etc
        # this is for filtering regions later
        path_metadata = example_metadata[sorted(path[1]),0].tolist()
        with open(bed_file, "w") as fp:
            for region_metadata in path_metadata:
                region = region_metadata.split(";")[1].split("=")[1] # TODO adjust here - just active region
                chrom = region.split(":")[0]
                start = region.split(":")[1].split("-")[0]
                stop = region.split("-")[1]
                fp.write("{}\t{}\t{}\n".format(chrom, start, stop))

        # write out the logit trajectory too
        path_signal = signal[path_indices]
        path_logits_mean = np.mean(
            path_signal[:,0,signal_indices],
            axis=0).tolist()

        with open(timepoints_file, "a") as fp:
            fp.write("{}\t{}\t{}\n".format(
                global_path_id,
                path_hierarchy,
                "\t".join([str(val) for val in path_logits_mean])))
                
        global_path_id += 1


    # TODO - now do a merge_paths run through? to find things that work together....
    total_new = 0
    for path1_idx in xrange(len(paths)):
        path1 = paths[path1_idx]
        path1_hierarchy = path1[0] # list of names
        path1_indices = path1[1]

        if len(path1_hierarchy) <= 1:
            continue

        for subpath1_idx in xrange(len(path1_hierarchy)-1, 0, -1):
            subpath1 = path1_hierarchy[subpath1_idx:]
            
            # starting at BOTTOM of path, see if any other paths have same bottom path
            for path2_idx in xrange(len(paths)):
                if path2_idx <= path1_idx:
                    continue
                path2 = paths[path2_idx]
                path2_hierarchy = path2[0] # list of names
                #print path2_hierarchy

                if len(path2_hierarchy) <= 1:
                    continue
                
                for subpath2_idx in xrange(len(path2_hierarchy)-1, 0, -1):
                    subpath2 = path2_hierarchy[subpath2_idx:]

                    if subpath1 == subpath2:
                        # merge paths!
                        path2_indices = path2[1]
                        merged_path_indices = sorted(list(
                            set(path1_indices).union(set(path2_indices))))

                        if len(merged_path_indices) < min_region_count:
                            continue
                        
                        path1_start = path1_hierarchy[:subpath1_idx]
                        path2_start = path2_hierarchy[:subpath2_idx]
                        #print path1_hierarchy
                        #print path2_hierarchy
                        path_hierarchy = "({};{})>{}".format(
                            "->".join(path1_start),
                            "->".join(path2_start),
                            "->".join(subpath2))
                        print ">>>", path_hierarchy, len(merged_path_indices)
                        total_new += 1

                        # write out to file
                        with open(metadata_file, "a") as fp:
                            fp.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                global_path_id,
                                path_hierarchy,
                                len(merged_path_indices),
                                path_dmim_max,
                                path_delta_logits_max,
                                "\t".join([str(val)
                                           for val in path_extra_results])))

                        bed_file = "{}.path-{}.bed".format(
                            h5_file.split(".h5")[0],
                            global_path_id)

                        # TODO for each region, keep track of extra information
                        # ie ATAC signal, H3K27ac, dmim score, etc
                        # this is for filtering regions later
                        path_metadata = example_metadata[sorted(merged_path_indices),0].tolist()
                        with open(bed_file, "w") as fp:
                            for region_metadata in path_metadata:
                                region = region_metadata.split(";")[1].split("=")[1] # TODO adjust here - just active region
                                chrom = region.split(":")[0]
                                start = region.split(":")[1].split("-")[0]
                                stop = region.split("-")[1]
                                fp.write("{}\t{}\t{}\n".format(chrom, start, stop))

                        # write out the logit trajectory too
                        path_signal = signal[merged_path_indices]
                        path_logits_mean = np.mean(
                            path_signal[:,0,signal_indices],
                            axis=0).tolist()

                        with open(timepoints_file, "a") as fp:
                            fp.write("{}\t{}\t{}\n".format(
                                global_path_id,
                                path_hierarchy,
                                "\t".join([str(val) for val in path_logits_mean])))

                        # check out the mutational effects
                        path_delta_logits = delta_logits[sorted(path1_indices)]
                        mut_effects_file = "{}.path-{}.effects_per_example.txt".format(
                            h5_file.split(".h5")[0],
                            global_path_id)
                        get_mut_effect_results(path1_hierarchy, net, path_delta_logits, mut_effects_file, task_idx=0)

                        global_path_id += 1

    print total_new
    print global_path_id

    import ipdb
    ipdb.set_trace()

    if False:
        # get path logits
        # TODO add in filters as needed
        # TODO split this out into separate function
        #keep_indices = np.where(path_extra_results_examples[1] > 0.7)[0]
        # TODO remove this, not being used
        example_logits = get_path_example_logits(
            path[0],
            net,
            path_logits[:,0,:], # TODO change the task here
            path_mut_logits)
            #path_logits[keep_indices,0],
            #path_mut_logits[keep_indices])

        #from scipy.stats import describe
        #print describe(example_logits[:,0])
        #print describe(example_logits[:,1])
        #print describe(example_logits[:,2])
        #import ipdb
        #ipdb.set_trace()
        
    # and resort metadata
    metadata = pd.read_table(metadata_file)
    #metadata_sorted = metadata.sort_values(["dmim"], ascending=False)
    metadata_sorted = metadata.sort_values([extra_keys[1]], ascending=False)
    metadata_sorted_file = "{}.sorted.txt".format(metadata_file.split(".txt")[0])
    metadata_sorted.to_csv(metadata_sorted_file, sep='\t', index=False)
        
    return paths
