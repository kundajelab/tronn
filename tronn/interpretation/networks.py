# description: code for doing network analyses

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
        self.attrs = dict(attrs)

    def get_tuple(self):
        return (self.name, self.attrs)
        
        
class DirectedEdge(object):
    """edge class"""
    
    def __init__(self, start_node_id, end_node_id, attrs={}):
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.attrs = dict(attrs)
        self.name = "{}_to_{}".format(start_node_id, end_node_id)

    def get_tuple(self):
        return (self.start_node_id, self.end_node_id, self.attrs)


class MotifGraph(object):
    """network class for managing a directed motif network"""
    
    def __init__(self, nodes=[], edges=[], propagated=False):
        """initialize
        note that the nodes and edges are SHALLOW copies
        """
        self.nodes = list(nodes)
        self.node_ids = [node.name for node in self.nodes]
        self.edges = list(edges)
        self.propagated = propagated

        self.attrs = {}
        
        # maintain a set of examples in graph that
        # can be adjusted for every new copy of graph
        self.attrs["node_examples"] = {}
        for node in self.nodes:
            self.attrs["node_examples"][node.name] = node.attrs["examples"]

        
    def add_node(self, node):
        """add node
        NOTE: does not copy node, but does make a new net object
        """
        new_net = MotifGraph(nodes=self.nodes, edges=self.edges)
        new_net.attrs = copy.deepcopy(self.attrs)
        self.propagated = False
        new_net.nodes.append(node)
        new_net.attrs["node_examples"][node.name] = node.attrs["examples"]
        return new_net
        

    def add_edge(self, edge):
        """add edge
        NOTE: does not copy edge, but does make a new net object
        """
        new_net = MotifGraph(nodes=self.nodes, edges=self.edges)
        new_net.attrs = copy.deepcopy(self.attrs)
        self.propagated = False
        new_net.edges.append(edge)
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

    
    def merge(self, other_net):
        """merge two instances of MotifNet
        NOTE: this is a copy, but not a deep copy
        """
        nodes = list(set(self.nodes + other_net.nodes))
        edges = list(set(self.edges + other_net.edges))
        new_net = MotifNet(nodes, edges)
        return new_net

    
    def get_node_by_id(self, node_id):
        """get the node object by id
        """
        for node in self.nodes:
            if node.name == node_id:
                return node
        
    def get_node_out_edges(self, node_id):
        """return the edges
        """
        node_edges = []
        for edge in self.edges:
            if node_id in edge.start_node_id:
                node_edges.append(edge)

        return node_edges

    def get_node_in_edges(self, node_id):
        """return edges coming into node
        """
        node_edges = []
        for edge in self.edges:
            if node_id in edge.end_node_id:
                node_edges.append(edge)

        return node_edges


    def get_leaf_nodes(self):
        """extract all nodes with no OUT edges
        """
        leaf_node_ids = []
        for node in self.nodes:
            edges = self.get_node_out_edges(node.name)
            if len(edges) == 0:
                leaf_node_ids.append(node.name)
                
        return leaf_node_ids


    def get_covered_region_set(self):
        """for the corrected (all info fully propagated) graph,
        go to leafs and collect all regions
        """
        leaf_node_ids = self.get_leaf_nodes()

        region_set = set([])
        for node_id in leaf_node_ids:
            region_set = region_set.union(
                self.attrs["node_examples"][node_id])

        region_num = len(region_set)
            
        return region_num, region_set
    

    
    def propagate_up(
            self, node_id, transform_fn, seen_nodes=[]):
        """from a starting node, propagate up and
        adjust graph attributes
        """
        # get parent node ids
        parent_node_ids = []
        in_edges = self.get_node_in_edges(node_id)
        for edge in in_edges:
            parent_node_ids.append(edge.start_node_id)

        for parent_node_id in parent_node_ids:
            if parent_node_id not in seen_nodes:
                # note that trasnform fn must adjust the graph attributes
                self.attrs = transform_fn(parent_node_id, self.attrs)
                seen_nodes.append(parent_node_id)            
                self.propagate_up(
                    parent_node_id,
                    transform_fn,
                    seen_nodes=seen_nodes)

        return None

    
    def propagate_down(
            self, node_id, transform_fn, seen_nodes=[]):
        """from a starting node, propagate up and
        adjust each node using transform fn
        """
        # get parent node ids
        child_node_ids = []
        out_edges = self.get_node_out_edges(node_id)
        for edge in out_edges:
            child_node_ids.append(edge.end_node_id)

        for child_node_id in child_node_ids:
            if child_node_id not in seen_nodes:
                # note that trasnform fn must adjust the graph attributes
                self.attrs = transform_fn(child_node_id, self.attrs)
                seen_nodes.append(child_node_id)
                self.propagate_down(
                    child_node_id,
                    transform_fn,
                    seen_nodes=seen_nodes)
                
        return None


    def update_examples(self, edge, edge_type="internal_edge"):
        """update the graph based on addition of internal edge
        """
        assert edge.start_node_id in self.node_ids
        assert edge.end_node_id in self.node_ids
        
        # setup - get examples that have edge and both (orig) node effects
        edge_examples = edge.attrs["examples"]
        edge_examples = edge_examples.intersection(
            self.get_node_by_id(edge.start_node_id).attrs["examples"])
        edge_examples = edge_examples.intersection(
            self.get_node_by_id(edge.end_node_id).attrs["examples"])

        start_node_examples = self.attrs["node_examples"][edge.start_node_id]
        end_node_examples = self.attrs["node_examples"][edge.end_node_id]
        
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
        self.attrs["node_examples"][edge.start_node_id] = new_upstream_examples
        
        # and propagate
        def transform_fn(node_id, attrs):
            attrs["node_examples"][node_id] = new_upstream_examples
            return attrs
        self.propagate_up(edge.start_node_id, transform_fn)

        # after updating upstream, update downstream accordingly
        new_downstream_examples = new_upstream_examples.intersection(edge_examples)
        
        # update downstream node - union
        #new_downstream_examples = downstream_fn(edge_examples, end_node_examples)
        new_downstream_examples = downstream_fn(new_downstream_examples, end_node_examples)
        self.attrs["node_examples"][edge.end_node_id] = new_downstream_examples
        
        # and propagate
        def transform_fn(node_id, attrs):
            attrs["node_examples"][node_id] = new_downstream_examples
            return attrs
        self.propagate_down(edge.end_node_id, transform_fn)

        # check
        if False:
            print [node.name for node in self.nodes]
            for node in self.nodes:
                print "{}\t{}".format(node.name, len(self.attrs["node_examples"][node.name]))
            region_num, _ = self.get_covered_region_set()
            print region_num
        
        return None

    
    def determine_region_set(self, node_ids, seen_edges=[]):
        """given a node list and connections between them
        determine the regions that fall into this set
        """
        upstream_set = set([])
        downstream_set = set([])

        # get first node info
        node_ids = list(node_ids)
        root_node = node_ids[0]
        node_ids.remove(root_node)
        node_set = set(
            self.get_node_by_id(root_node).attrs["examples"]) # TODO this needs to be considered

        if len(node_ids) == 0:
            upstream_set = set(node_set)
            downstream_set = set(node_set)
        
        for i in xrange(len(node_ids)):
            node_id = node_ids[i]

            if True:
                # DOWNSTREAM
                edges = self.get_node_out_edges(root_node)
                for edge in edges:

                    # get other node
                    other_node = edge.end_node

                    # TODO need to figure out what to do with self feedback nodes
                    
                    # TODO change this to see whether edge was already checked?
                    # first check whether other node is in set desired
                    if other_node not in node_ids:
                        continue

                    # get the examples on the edge, these get adjusted
                    edge_examples = edge.attrs["examples"]

                    # recurse
                    other_node_upstream_set, other_node_downstream_set = self.determine_region_set(node_ids)

                    # intersect with the downstream set, and then add to total
                    edge_downstream_set = edge_examples.intersection(other_node_downstream_set)
                    downstream_set = downstream_set.union(edge_downstream_set)

                    # union with the upstream set?
                    edge_upstream_set = edge_examples.union(other_node_upstream_set)
                    upstream_set = upstream_set.union(edge_upstream_set)
                    
                # UPSTREAM                        
                edges = self.get_node_in_edges(root_node)
                for edge in edges:

                    # get other node
                    other_node = edge.start_node
                    
                    # first check whether other node is in set desired
                    if other_node not in node_ids:
                        continue

                    # intersection (edge)
                    edge_examples = edge.attrs["examples"]

                    # and recurse
                    other_node_upstream_set, other_node_downstream_set = self.determine_region_set(node_ids)

                    # union with the downstream set, and then add to total
                    edge_downstream_set = edge_examples.union(other_node_downstream_set)
                    downstream_set = downstream_set.union(edge_downstream_set)

                    # intersection with the upstream set
                    edge_upstream_set = edge_examples.intersection(other_node_upstream_set)
                    upstream_set = upstream_set.union(edge_upstream_set)

                    
        return upstream_set, downstream_set


    
    def get_adjacency_matrix(self):


        return
    

    def passes_minimum_cutoff(self, subgraph):
        """
        """


        return

    
    
    def get_paths(
            self,
            initial_node_id,
            min_region_count,
            current_path=([],[]),
            checked_nodes=[]):
        """recursive function to find path
        """
        paths = []
        
        # get node's edges
        edges = self.get_node_out_edges(initial_node_id)
        for edge in edges:
            
            # get other node
            other_node = edge.end_node
            other_node_examples = self.get_node_by_id(other_node).attrs["examples"]

            if True:
                if edge.name in checked_nodes:
                    continue
                #checked_nodes.append(edge.name)
            
            # if already seen, continue unless it was the last one (to get self loops)
            if False:
                if (other_node in checked_nodes) and (other_node in current_path[0][:-1]):
                    continue
                #if other_node not in current_path[:-1]:
                #    continue
                checked_nodes.append(other_node)
                
            if False:
                # check if already looked at
                if other_node in checked_nodes:
                    continue
                checked_nodes.append(other_node)
            
                # check for loops, continue if you hit a loop
                if other_node in current_path[:-1]:
                    continue

            # get intersect
            edge_examples = edge.attrs["examples"]
            shared_examples = current_path[1].intersection(edge_examples)

            # also intersect with downstream node
            # this means that all examples here have to have a delta logit effect
            # in the right direction as well as delta mutational effect
            # in the right direction
            if True:
                shared_examples = shared_examples.intersection(other_node_examples)
            
            # if intersect is good, append node, save out, and keep going down
            if len(shared_examples) > min_region_count:
                new_path = (current_path[0] + [other_node], shared_examples)
                paths.append(new_path)

                edges_seen = checked_nodes + [edge.name]
                
                edge_paths = self.get_paths(
                    other_node,
                    min_region_count,
                    current_path=new_path,
                    checked_nodes=edges_seen)#checked_nodes)
                paths += edge_paths
                
        return paths



def get_subgraphs_OLD(
        graph,
        subgraph,
        k=1,
        min_region_count=100):
    """start from a specific node and get all subgraphs
    that are size k (or less?)
    """
    # ignore empty graphs
    if k == 1:
        return []

    # get relevant edges
    edges = []
    for edge in graph.edges:
        if edge in subgraph.edges:
            continue
        if edge.start_node_id in subgraph.node_ids:
            edges.append(edge)
        elif edge.end_node_id in subgraph.node_ids:
            edges.append(edge)
    
    # consider all edges at the same time
    new_subgraphs = []
    for edge in edges:

        # get edge examples out in prep
        edge_examples = edge.attrs["examples"]            
        
        # check if internal and not seen already
        if (edge.start_node_id in subgraph.node_ids) and (edge.end_node_id in subgraph.node_ids):
            if edge not in subgraph.edges:
                new_subgraph = subgraph.add_edge(edge)
                new_subgraph.update_examples(edge, edge_type="internal_edge")
                new_subgraphs.append(new_subgraph)

        # check if first edge
        #elif len(subgraph.nodes) == 1:
        #    new_subgraph = subgraph.add_node(
        #        graph.get_node_by_id(edge.end_node_id)) # TODO - fix, this is not always true
        #    new_subgraph = new_subgraph.add_edge(edge)
        #    new_subgraph.update_examples(edge, edge_type="in_edge")
        #    new_subgraphs.append(new_subgraph)
        #    new_subgraphs += get_subgraphs(
        #        graph, new_subgraph, k-1, min_region_count=min_region_count)
                
        # check if edge adds new child node into subgraph
        elif (edge.end_node_id not in subgraph.node_ids):
            new_subgraph = subgraph.add_node(
                graph.get_node_by_id(edge.end_node_id))
            new_subgraph = new_subgraph.add_edge(edge)
            
            if len(new_subgraph.get_node_out_edges(edge.start_node_id)) == 1:
                new_subgraph.update_examples(edge, edge_type="frontier_edge")
            else:
                new_subgraph.update_examples(edge, edge_type="out_edge")

            new_subgraphs.append(new_subgraph)
            new_subgraphs += get_subgraphs(
                graph, new_subgraph, k-1, min_region_count=min_region_count)

        # check if edge adds new parent node into subgraph
        elif (edge.start_node_id not in subgraph.node_ids):
            new_subgraph = subgraph.add_node(
                graph.get_node_by_id(edge.start_node_id))
            new_subgraph = new_subgraph.add_edge(edge)
            
            if len(new_subgraph.get_node_in_edges(edge.end_node_id)) == 1:
                new_subgraph.update_examples(edge, edge_type="frontier_edge")
            else:
                new_subgraph.update_examples(edge, edge_type="in_edge")
            
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
        if edge.start_node_id in subgraph.node_ids:
            edges.append(edge)
        elif edge.end_node_id in subgraph.node_ids:
            edges.append(edge)
    
    # consider all edges at the same time
    new_subgraphs = []
    for edge in edges:

        # get edge examples out in prep
        edge_examples = edge.attrs["examples"]            
        
        # check if internal and not seen already
        if (edge.start_node_id in subgraph.node_ids) and (edge.end_node_id in subgraph.node_ids):
            if edge not in subgraph.edges:
                new_subgraph = subgraph.add_edge(edge)
                new_subgraph.update_examples(edge, edge_type="frontier_edge")
                new_subgraphs.append(new_subgraph)
                
        # check if edge adds new child node into subgraph
        elif (edge.end_node_id not in subgraph.node_ids):
            new_subgraph = subgraph.add_node(
                graph.get_node_by_id(edge.end_node_id))
            new_subgraph = new_subgraph.add_edge(edge)
            new_subgraph.update_examples(edge, edge_type="frontier_edge")

            new_subgraphs.append(new_subgraph)
            new_subgraphs += get_subgraphs(
                graph, new_subgraph, k-1, min_region_count=min_region_count)

        # check if edge adds new parent node into subgraph
        elif (edge.start_node_id not in subgraph.node_ids):
            new_subgraph = subgraph.add_node(
                graph.get_node_by_id(edge.start_node_id))
            new_subgraph = new_subgraph.add_edge(edge)
            new_subgraph.update_examples(edge, edge_type="frontier_edge")
            
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

    # TODO - seperate the below into a seperate function?
    # set up the mut nodes
    nodes = []
    for name_idx in xrange(len(mut_pwm_names)):
        pwm_name = mut_pwm_names[name_idx]
        had_effect = np.where(np.sum(delta_logits[:,name_idx,:] < 0, axis=1) > num_sig_tasks)[0].tolist()
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
    final_min_count = 300
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
                differs = False
                break
            
        if differs:
            filtered_subgraphs.append(subgraphs[i])
    print len(filtered_subgraphs)
    subgraphs = filtered_subgraphs

    import ipdb
    ipdb.set_trace()
    
    if True:
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

                    
        
                
    import ipdb
    ipdb.set_trace()

    quit()


    
    # additional stricter filter here
    print len(subgraphs)
    min_region_count = 400
    filtered_subgraphs = []
    for subgraph in subgraphs:
        upstream_set, downstream_set = subgraph.determine_region_set(
            [node.name for node in subgraph.nodes])
        region_set = upstream_set.intersection(downstream_set)
        print ";".join([node.name for node in subgraph.nodes]),
        print len(region_set)

        
        if len(region_set) > min_region_count:
            filtered_subgraphs.append(subgraph)
        
    print len(filtered_subgraphs)

    for subgraph in filtered_subgraphs:
        print ";".join([node.name for node in subgraph.nodes])

    import ipdb
    ipdb.set_trace()

    quit()

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
