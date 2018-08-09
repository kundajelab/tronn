# description: code for doing network analyses

import h5py

import numpy as np
import networkx as nx

from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys


# to consider - what other needs will there be for this network?
class Node(object):
    """node class"""

    def __init__(self, name=1, attrs={}):
        self.name = name
        self.attrs = attrs

    def get_tuple(self):
        return (self.name, self.attrs)
        
        
class DirectedEdge(object):
    """edge class"""
    
    def __init__(self, start_node, end_node, attrs={}):
        self.start_node = start_node
        self.end_node = end_node
        self.attrs = attrs

    def get_tuple(self):
        return (self.start_node, self.end_node, self.attrs)


class MotifNet(object):
    """network class for managing a directed motif network"""
    
    def __init__(self, nodes=[], edges=[]):
        """initialize
        """
        self.nodes = nodes
        self.edges = edges
                
    def add_node(self, node):
        """add node
        """
        self.nodes.append(node)

    def add_edge(self, edge):
        """
        """
        self.edges.append(edge)

    def get_node_out_edges(self, node_id):
        """return the edges
        """
        node_edges = []
        for edge in self.edges:
            if node_id in edge.get_tuple()[0]:
                node_edges.append(edge)

        return node_edges
        
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

            # check if already looked at
            if other_node in checked_nodes:
                continue
            checked_nodes.append(other_node)
            
            # check for loops, continue if you hit a loop
            if other_node in current_path:
                continue

            # get intersect
            edge_examples = edge.attrs["examples"]
            shared_examples = current_path[1].intersection(edge_examples)
            
            # if intersect is good, append node, save out, and keep going down
            if len(shared_examples) > min_region_count:
                new_path = (current_path[0] + [other_node], shared_examples)
                paths.append(new_path)
                edge_paths = self.get_paths(
                    other_node,
                    min_region_count,
                    current_path=new_path,
                    checked_nodes=checked_nodes)
                paths += edge_paths
                
        return paths
    


def get_motif_hierarchies(
        h5_file,
        adjacency_mat_key,
        data_key,
        metadata_key=DataKeys.SEQ_METADATA,
        sig_pwms_key=DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL,
        pwm_names_attr_key=AttrKeys.PWM_NAMES):
    """extract motif hierarchies
    """
    with h5py.File(h5_file, "r") as hf:
        adjacency_matrix = hf[adjacency_mat_key][:]
        data = hf[data_key][:]
        example_metadata = hf[metadata_key][:]
        sig_pwms = hf[sig_pwms_key][:]
        pwm_names = hf[adjacency_mat_key].attrs[pwm_names_attr_key]

    # TODO - change this later
    data = data[:,:,:,np.where(sig_pwms > 0)[0]]

    # set up the nodes
    # TODO note that there may be different sets in mut and response,
    # so need to save out both
    nodes = []
    for pwm_name in pwm_names:
        node = Node(pwm_name)
        nodes.append(node)

    # set up the edges
    num_mut_motifs = adjacency_matrix.shape[0]
    num_response_motifs = adjacency_matrix.shape[2]
    edges = []
    for mut_idx in xrange(num_mut_motifs):
        for response_idx in xrange(num_response_motifs):

            if np.sum(adjacency_matrix[mut_idx, :, response_idx] != 0) < 3:
                continue
            
            # set up an edge
            start_node = pwm_names[mut_idx]
            end_node = pwm_names[response_idx]

            # get subset
            sig_in_tasks = np.sum(data[:,mut_idx,:,response_idx] != 0, axis=1)
            edge_examples = example_metadata[np.where(sig_in_tasks >= 2)[0],0].tolist()
            attr_dict = {"examples": set(edge_examples)}
            
            # attach the list of examples with both 
            edge = DirectedEdge(start_node, end_node, attrs=attr_dict)
            edges.append(edge)

    # put into network
    net = MotifNet(nodes, edges)
    
    # run the network to get hierarchical paths
    # is each path a tuple ([path], [examples])?
    min_region_count = 600
    paths = []
    for node in net.nodes:
        all_examples = set(example_metadata[:,0].tolist())
        paths += net.get_paths(
            node.name,
            min_region_count,
            current_path=([node.name], all_examples),
            checked_nodes=[node.name])

    # for each path, want to write out a BED file (and write a metadata file)
    metadata_file = "{}.hierarchies.metadata.txt".format(
        h5_file.split(".h5")[0])
    for path_idx in xrange(len(paths)):
        path = paths[path_idx]
        path_hierarchy = ";".join(path[0])
        with open(metadata_file, "a") as fp:
            fp.write("{}\t{}\n".format(path_idx, path_hierarchy))
        
        bed_file = "{}.path-{}.bed.gz".format(
            h5_file.split(".h5")[0],
            path_idx)

        
    
    # and also save the metadata somewhere
    # use pandas

    import ipdb
    ipdb.set_trace()
        
    return paths
