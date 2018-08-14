# description: code for doing network analyses

import h5py

import numpy as np
import pandas as pd
import networkx as nx

from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys


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


def get_motif_hierarchies(
        h5_file,
        adjacency_mat_key,
        data_key,
        metadata_key=DataKeys.SEQ_METADATA,
        sig_pwms_key=DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL,
        logits_key=DataKeys.LOGITS,
        mut_logits_key=DataKeys.MUT_MOTIF_LOGITS,
        pwm_names_attr_key=AttrKeys.PWM_NAMES):
    """extract motif hierarchies
    """
    with h5py.File(h5_file, "r") as hf:
        adjacency_matrix = hf[adjacency_mat_key][:]
        data = hf[data_key][:]
        example_metadata = hf[metadata_key][:]
        sig_pwms = hf[sig_pwms_key][:]
        mut_pwm_names = hf[adjacency_mat_key].attrs[pwm_names_attr_key] # TODO need to have mut names attr and pwm names attr
        logits = hf[logits_key][:]
        mut_logits = hf[mut_logits_key][:]

    # TODO - change these later
    data = data[:,:,:,np.where(sig_pwms > 0)[0]]
    response_pwm_names = mut_pwm_names
    
    # set up the mut nodes
    nodes = []
    for name_idx in xrange(len(mut_pwm_names)):
        pwm_name = mut_pwm_names[name_idx]
        node = Node(pwm_name, {"mut_idx": name_idx})
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

            if np.sum(adjacency_matrix[mut_idx, :, response_idx] != 0) < 3:
                continue
            
            # set up an edge
            start_node = mut_pwm_names[mut_idx]
            end_node = response_pwm_names[response_idx]
            
            # get subset
            sig_in_tasks = np.sum(data[:,mut_idx,:,response_idx] != 0, axis=1)
            #edge_examples = example_metadata[np.where(sig_in_tasks >= 2)[0],0].tolist()
            edge_examples = np.where(sig_in_tasks >= 2)[0].tolist() # keep as indices until end
            attr_dict = {"examples": set(edge_examples)}
            
            # attach the list of examples with both 
            edge = DirectedEdge(start_node, end_node, attrs=attr_dict)
            edges.append(edge)

    # put into network
    net = MotifNet(nodes, edges)
    
    # run the network to get hierarchical paths
    # is each path a tuple ([path], [examples])?
    # TODO keep paths as indices?
    # TODO also keep edges as indices?
    min_region_count = 600
    paths = []
    for node in net.nodes:
        all_examples = set(range(example_metadata.shape[0]))
        #all_examples = set(example_metadata[:,0].tolist())
        paths += net.get_paths(
            node.name,
            min_region_count,
            current_path=([node.name], all_examples),
            checked_nodes=[node.name])

    # for each path, want to write out a BED file (and write a metadata file)
    metadata_file = "{}.hierarchies.metadata.txt".format(
        h5_file.split(".h5")[0])
    with open(metadata_file, "w") as fp:
        fp.write("path_idx\tpath_hierarchy\tnum_regions\tdmim\tdelta_logit\n")
        
    for path_idx in xrange(len(paths)):
        path = paths[path_idx]
        path_hierarchy = ";".join(path[0])
        path_indices = sorted(path[1])
        path_data = data[path_indices]
        path_logits = np.expand_dims(logits[path_indices], axis=1)
        path_mut_logits = mut_logits[path_indices]
        path_delta_logits = np.subtract(path_mut_logits, path_logits)
        
        # get the aggregate effects over the path
        # TODO check the sign on the multiplications
        path_dmim_effects = get_path_dmim_effects(path[0], net, path_data)
        path_delta_logits_effects = get_path_logit_effects(path[0], net, path_delta_logits)

        # TODO just choose max val?
        path_dmim_max = path_dmim_effects[
            np.argmax(np.abs(path_dmim_effects))]
        path_delta_logits_max = path_delta_logits_effects[
            np.argmax(np.abs(path_delta_logits_effects))]
            
        # write out to file
        with open(metadata_file, "a") as fp:
            fp.write("{}\t{}\t{}\t{}\t{}\n".format(
                path_idx,
                path_hierarchy,
                len(path[1]),
                path_dmim_max,
                path_delta_logits_max))
        
        bed_file = "{}.path-{}.bed".format(
            h5_file.split(".h5")[0],
            path_idx)

        path_metadata = example_metadata[sorted(path[1]),0].tolist()
        with open(bed_file, "w") as fp:
            for region_metadata in path_metadata:
                region = region_metadata.split(";")[1].split("=")[1] # TODO adjust here - just active region
                chrom = region.split(":")[0]
                start = region.split(":")[1].split("-")[0]
                stop = region.split("-")[1]
                fp.write("{}\t{}\t{}\n".format(chrom, start, stop))
    
    # and resort metadata
    metadata = pd.read_table(metadata_file)
    #metadata_sorted = metadata.sort_values(["delta_logit"], ascending=False)
    metadata_sorted = metadata.sort_values(["dmim"], ascending=False)
    metadata_sorted_file = "{}.sorted.txt".format(metadata_file.split(".txt")[0])
    metadata_sorted.to_csv(metadata_sorted_file, sep='\t', index=False)
    
    import ipdb
    ipdb.set_trace()
        
    return paths
