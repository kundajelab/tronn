# description: code for finding grammars

import matplotlib
matplotlib.use("Agg")

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import read_pwm_file

from tronn.interpretation.clustering import sd_cutoff

import phenograph

import networkx as nx
from networkx.drawing.nx_pydot import pydot_layout

from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from scipy.special import expit

def read_grammar_file(grammar_file, pwm_file, as_dict=False):
    """Read in grammar and pwm file to set up grammars
    """
    if as_dict:
        grammars = {}
    else:
        grammars = []

    # open pwm files
    pwms = read_pwm_file(pwm_file)
    pwm_dict = read_pwm_file(pwm_file, as_dict=True)

    # open grammar file
    with open(grammar_file, "r") as fp:
        line = fp.readline().strip()
        while True:
            if line == "":
                break
            # get header
            header = line.strip(">").strip()
            
            # get params
            line = fp.readline().strip()
            param_string = line.strip("#params").strip()
            motifspace_dict = {}
            node_dict = {}
            edge_dict = {}

            line = fp.readline().strip()
            
            # get motifspace vector
            assert line.startswith("#motifspace")
            motifspace_param_string = line.strip().split()[1]
            while True:
                line = fp.readline().strip()
                if line.startswith("#nodes") or line == "": break

                fields = line.split()
                motifspace_dict[fields[0]] = (float(fields[1]), float(fields[2]))
            
            # go through nodes TODO check for presence in pwm file?
            assert line.startswith("#nodes")
            while True:
                line = fp.readline().strip()
                if line.startswith("#edges") or line == "": break

                fields = line.split()
                node_dict[fields[0]] = (float(fields[1]), float(fields[2]))

            # go through edges
            assert line.startswith("#edges")
            while True:
                line = fp.readline().strip()
                if line.startswith(">") or line == "": break

                fields = line.split()
                edge_dict[(fields[0], fields[1])] = float(fields[2])

            grammar = Grammar(
                pwms,
                node_dict,
                edge_dict,
                param_string,
                motifspace_dict=motifspace_dict,
                motifspace_param_string=motifspace_param_string,
                name=header)

            if as_dict:
                grammars[header] = grammar
            else:
                grammars.append(grammar)

    return grammars


class Grammar(object):
    """This grammar is a linear model with pairwise interactions.
    """

    def __init__(
            self,
            pwms,
            node_dict,
            edge_dict,
            param_string,
            motifspace_dict=None,
            motifspace_param_string="",
            name=None,
            threshold=None):
        self.name = name
        self.pwms = pwms
        self.param_string = param_string
        self.nodes = node_dict
        self.edges = edge_dict
        self.motifspace_dict = motifspace_dict
        self.motifspace_param_string = motifspace_param_string

        # set up motifspace vector
        if self.motifspace_dict is not None:
            self.motifspace_vector = np.zeros((len(self.pwms)))
            self.motifspace_weights = np.zeros((len(self.pwms)))
            for pwm_idx in xrange(len(self.pwms)):
                pwm = self.pwms[pwm_idx]
                try:
                    val, weight = self.motifspace_dict[pwm.name]
                    self.motifspace_vector[pwm_idx] = val
                    self.motifspace_weights[pwm_idx] = weight
                except:
                    pass
            motifspace_params = dict(
                item.split("=")
                for item in self.motifspace_param_string.split(";"))
            self.motifspace_threshold = float(motifspace_params["threshold"])
        else:
            self.motifspace_vector = None
            self.motifspace_threshold = None
            
        # set up 1D pwm vector with thresholds
        # TODO
        self.pwm_vector = np.zeros((len(self.pwms)))
        self.pwm_thresholds = np.zeros((len(self.pwms)))
        for pwm_idx in xrange(len(self.pwms)):
            pwm = self.pwms[pwm_idx]
            try:
                threshold, coef = self.nodes[pwm.name]
                self.pwm_vector[pwm_idx] = coef
                self.pwm_thresholds[pwm_idx] = threshold
            except:
                pass
        
        # set up 2D numpy array of adjacencies
        self.adjacency_matrix = np.zeros((len(self.pwms), len(self.pwms)))
        for pwm1_idx in xrange(len(self.pwms)):
            for pwm2_idx in xrange(len(self.pwms)):
                edge_name = (self.pwms[pwm1_idx].name, self.pwms[pwm2_idx].name)
                try:
                    val = self.edges[edge_name]
                    self.adjacency_matrix[pwm1_idx, pwm2_idx] = val
                except:
                    pass
        
        return

    
    def add_node(self, node_id):
        """
        """

        return

    
    def delete_node(self, node_id):
        """
        """
        del self.nodes[node_id]
        for edge_key in self.edges.keys():
            if node_id == edge_key[0]:
                del self.edges[edge_key]
            elif node_id == edge_key[1]:
                del self.edges[edge_key]
        # adjust 1D pwm vector

        
        return

    def to_file(self, filename, filemode="a"):
        """Write out grammar to file
        """
        with open(filename, filemode) as out:
            out.write(">{}\n".format(self.name))
            out.write("#params {}\n".format(self.param_string))

            # write out motif space
            out.write("#motifspace\t{}\n".format(self.motifspace_param_string))
            for node in self.motifspace_dict.keys():
                out.write("{0}\t{1}\t{2}\n".format(
                    node,
                    self.motifspace_dict[node][0],
                    self.motifspace_dict[node][1]))
            
            # write out nodes
            out.write("#nodes\n")
            for node in self.nodes.keys():
                out.write("{0}\t{1}\t{2}\n".format(
                    node, self.nodes[node][0], self.nodes[node][1]))
                
            # then write out edges
            out.write("#edges\n")
            for edge in self.edges.keys():
                out.write("{0}\t{1}\t{2}\n".format(
                    edge[0], edge[1], self.edges[edge]))
            
        return None
    
    def plot(self, plot_file, positions=None):
        """Plot out the grammar. Use positions if given
        """
        # build a networkx graph

        
        # build a networkx graph
        links = self.adjacency_matrix.stack().reset_index()
        links.columns = ["var1", "var2", "value"]
        
        # remove self correlation and zeros
        links_filtered = links.loc[ (links["value"] > corr_thresh) & (links["var1"] !=  links["var2"]) ]
        links_filtered = links_filtered.loc[links_filtered["var1"] != "UNK"]
        links_filtered = links_filtered.loc[links_filtered["var2"] != "UNK"]

        # make graph
        #G = nx.from_pandas_dataframe(links_filtered, "var1", "var2")
        G = nx.from_pandas_edgelist(links_filtered, "var1", "var2", edge_attr="value")


        
        #G = get_networkx_graph(corr_mat, corr_thresh=corr_thresh)

        # set up node sizes
        if node_size_dict is None:
            node_size = 50
        else:
            node_size = [node_size_dict[node]*0.5 for node in G.nodes]
        print node_size

        # set up edge weights
        edge_to_weight = nx.get_edge_attributes(G, "value")
        edge_weights = [edge_to_weight[edge]*0.01 for edge in G.edges]
        print edge_weights

        # plot
        f = plt.figure()
        nx.draw(
            G,
            pos=positions,
            with_labels=True,
            node_color="orange",
            node_size=node_size,
            edge_color="black",
            linewidths=edge_weights,
            font_size=3)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        f.savefig("{}.network.pdf".format(prefix))
        

        
        

        
        return
    
    

def get_significant_correlations(motif_mat, corr_method="pearson", pval_thresh=0.05, corr_min=0.4):
    """Given a matrix, calculate pearson correlation for each pair of 
    columns and look at p-val. If p-val is above threshold, keep 
    otherwise leave as 0.
    """
    num_columns = motif_mat.shape[1]
    correlation_vals = np.zeros((num_columns, num_columns))
    correlation_pvals = np.zeros((num_columns, num_columns))
    for i in xrange(num_columns):
        for j in xrange(num_columns):

            if corr_method == "pearson":
                cor_val, pval = pearsonr(motif_mat[:,i], motif_mat[:,j])
                if pval < pval_thresh and cor_val > corr_min:
                    correlation_vals[i,j] = cor_val
                    correlation_pvals[i,j] = pval
            elif corr_method == "continuous_jaccard":
                min_vals = np.minimum(motif_mat[:,i], motif_mat[:,j])
                max_vals = np.maximum(motif_mat[:,i], motif_mat[:,j])
                similarity = np.sum(min_vals) / np.sum(max_vals)
                correlation_vals[i,j] = similarity
                correlation_pvals[i,j] = similarity
            elif corr_method == "intersection_size":
                intersect = np.sum(
                    np.logical_and(motif_mat[:,i] > 0, motif_mat[:,j] > 0))
                #intersect_fract = float(intersect) / motif_mat.shape[0]
                intersect_fract = intersect
                correlation_vals[i,j] = intersect_fract
                correlation_pvals[i,j] = intersect_fract
                
    return correlation_vals, correlation_pvals

def reduce_corr_mat_by_motif_similarity(
        corr_mat,
        pwm_dict,
        signal_dict,
        edge_cor_thresh=0.4,
        cor_thresh=0.6,
        ncor_thresh=0.4):
    """Given the correlation matrix and pwm list, reduce.
    """

    corr_mat_tmp = pd.DataFrame(corr_mat)
    
    while True:

        # stack
        edges = corr_mat_tmp.stack().reset_index()
        edges.columns = ["var1", "var2", "value"]

        # remove self correlation and zeros
        edges_filtered = edges.loc[edges["value"] > edge_cor_thresh]
        edges_filtered = edges_filtered.loc[edges_filtered["var1"] != edges_filtered["var2"]]
        edges_filtered = edges_filtered.loc[edges_filtered["var1"] != "UNK"]
        edges_filtered = edges_filtered.loc[edges_filtered["var2"] != "UNK"]

        # sort
        edges_filtered = edges_filtered.sort_values("value", ascending=False)

        # for each edge, check ncor
        total_checked = 0
        for edge_idx in xrange(edges_filtered.shape[0]):
            node1 = edges_filtered["var1"].iloc[edge_idx]
            node2 = edges_filtered["var2"].iloc[edge_idx]
            ncor = pwm_dict[node1].rsat_cor(pwm_dict[node2], ncor=True)
            cor = pwm_dict[node1].rsat_cor(pwm_dict[node2], ncor=False)

            # if passes, choose max signal row and keep, delete the other
            if ncor > ncor_thresh and cor > cor_thresh:
                node1_signal = signal_dict[node1]
                node2_signal = signal_dict[node2]
                
                if node1_signal > node2_signal:
                    # keep node1
                    corr_mat_tmp.drop(node2, axis=0, inplace=True)
                    corr_mat_tmp.drop(node2, axis=1, inplace=True)
                else:
                    # keep node2
                    corr_mat_tmp.drop(node1, axis=0, inplace=True)
                    corr_mat_tmp.drop(node1, axis=1, inplace=True)
                    
                # found a match, start from beginning
                break

            total_checked += 1
            
        # if make it here (ie, nothing passes and merges) break.
        if total_checked == edges_filtered.shape[0]:
            break
    
    return corr_mat_tmp



def get_networkx_graph(corr_mat, corr_thresh=0.0):
    """preprocessing to get a clean graph object
    """
    links = corr_mat.stack().reset_index()
    links.columns = ["var1", "var2", "value"]

    # remove self correlation and zeros
    links_filtered = links.loc[ (links["value"] > corr_thresh) & (links["var1"] !=  links["var2"]) ]
    links_filtered = links_filtered.loc[links_filtered["var1"] != "UNK"]
    links_filtered = links_filtered.loc[links_filtered["var2"] != "UNK"]

    # make graph
    #G = nx.from_pandas_dataframe(links_filtered, "var1", "var2")
    G = nx.from_pandas_edgelist(links_filtered, "var1", "var2", edge_attr="value")

    return G


def plot_corr_as_network(corr_mat, prefix):
    """plot out network
    """
    links = corr_mat.stack().reset_index()
    links.columns = ["var1", "var2", "value"]

    # remove self correlation and zeros
    links_filtered = links.loc[ (links["value"] > 0.0) & (links["var1"] !=  links["var2"]) ]
    links_filtered = links_filtered.loc[links["var1"] != "UNK"]
    links_filtered = links_filtered.loc[links["var2"] != "UNK"]

    # make graph
    #G = nx.from_pandas_dataframe(links_filtered, "var1", "var2")
    G = nx.from_pandas_edgelist(links_filtered, "var1", "var2")

    # plot
    f = plt.figure()
    nx.draw(
        G,
        pos=nx.spring_layout(G),
        #pos=nx.nx_pydot.graphviz_layout(G, prog="neato"),
        #pos=pydot_layout(G),
        with_labels=True,
        node_color="orange",
        node_size=200,
        edge_color="black",
        linewidths=1,
        font_size=5)
    f.savefig("{}.network.pdf".format(prefix))
    
    return G


def plot_corr_on_fixed_graph(corr_mat, positions, prefix, corr_thresh=0.5, node_size_dict=None):
    """Probably need some assertions to make sure that the
    same nodes are represented?
    """
    G = get_networkx_graph(corr_mat, corr_thresh=corr_thresh)

    # set up node sizes
    if node_size_dict is None:
        node_size = 50
    else:
        node_size = [node_size_dict[node]*0.5 for node in G.nodes]
    print node_size

    # set up edge weights
    edge_to_weight = nx.get_edge_attributes(G, "value")
    edge_weights = [edge_to_weight[edge]*0.01 for edge in G.edges]
    print edge_weights
    
    # plot
    f = plt.figure()
    nx.draw(
        G,
        pos=positions,
        #pos=nx.nx_pydot.graphviz_layout(G, prog="neato"),
        #pos=pydot_layout(G),
        with_labels=True,
        node_color="orange",
        node_size=node_size,
        edge_color="black",
        linewidths=edge_weights,
        #linewidths=1,
        font_size=3)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    f.savefig("{}.network.pdf".format(prefix))
    
    return G


def get_significant_motifs(motif_mat, colnames, prefix, num_shuffles=99, pval_thresh=0.05):
    """Given a motif matrix, shuffle within the rows to get bootstraps
    and determine whether the motif is significantly enriched against the
    null background
    """

    # first calculate scores
    true_scores = np.sum(motif_mat, axis=0)

    # set up results array
    score_array = np.zeros((num_shuffles+1, motif_mat.shape[1]))
    score_array[0,:] = true_scores

    # go through shuffles
    for i in xrange(num_shuffles):

        if i % 10 == 0: print i

        # shuffle
        random_ranks = np.random.random(motif_mat.shape)
        idx = np.argsort(random_ranks, axis=1)
        shuffled_array = motif_mat[np.arange(motif_mat.shape[0])[:, None], idx]

        # calculate scores and add to mat
        shuffled_scores = np.sum(shuffled_array, axis=0)
        score_array[i+1,:] = shuffled_scores

    # now rank order to determine significance
    scores_file = "{}.permutations.txt".format(prefix)
    scores_df = pd.DataFrame(data=score_array, columns=colnames)
    scores_df.to_csv(scores_file, sep="\t")

    fdr_file = "{}.fdr.txt".format(prefix)
    score_ranks_df = scores_df.rank(ascending=False, pct=True)
    fdr_vals = score_ranks_df.iloc[0,:]
    fdr_vals.to_csv(fdr_file, sep="\t")
    
    fdr_thresholded_file = "{}.cutoff.txt".format(fdr_file.split(".txt")[0])
    fdr_thresholded = fdr_vals.ix[fdr_vals < pval_thresh]
    fdr_thresholded.to_csv(fdr_thresholded_file, sep="\t")

    return fdr_thresholded



def get_significant_delta_motifs(h5_file, mut_key, pwm_score_key, pwm_list, pwm_dict, cutoff=0.0002):
    """given an (example, delta_score) matrix, determine which columns
    are significant by extracting the t score and seeing if 0 in range.
    """
    from scipy.stats import ttest_1samp
    from tronn.interpretation.motifs import reduce_pwms_by_signal_similarity
    from tronn.interpretation.motifs import reduce_pwms
    
    with h5py.File(h5_file, "r") as hf:
        dmim = hf[mut_key] # {N, mut, all_motifs}
        pwm_scores = hf[pwm_score_key]

        # filter 1 - first only use motifs with high importance (loosely)
        #pwm_importance_vector = reduce_pwms(pwm_scores, pwm_list, pwm_dict, std_thresh=2)
        from tronn.interpretation.clustering import sd_cutoff
        pwm_importance_vector = sd_cutoff(pwm_scores, std_thresh=2)
        

        print "after importance scores"
        indices = np.where(pwm_importance_vector > 0)[0].tolist()
        print [pwm_list[k].name for k in indices]
        pwm_vector = np.zeros((len(pwm_list)))

        # for each mutational dataset:
        for i in xrange(dmim.shape[1]):
            
            dmim_single_mut = dmim[:,i,:] # {N, all_motifs}

            # calculate t test for each column and mark as 1 if passes test
            ttest_results = ttest_1samp(dmim_single_mut, 0)

            # cutoff and only keep those that have high importance and pval < cutoff
            keep = pwm_importance_vector * (ttest_results[1] < cutoff)
            print np.sum(keep > 0)

            print "after t test"
            indices = np.where(keep > 0)[0].tolist()
            print [pwm_list[k].name for k in indices]

            if True:
                # reduce pwm similarity?
                # revisit here, may be blocking on specific motifs
                # TODO - want to keep the ones that are marked as important by importance vector?
                signal_filtered = reduce_pwms_by_signal_similarity(
                    np.abs(dmim_single_mut), pwm_list, pwm_dict)

                keep = keep * signal_filtered
            
            #keep = reduce_pwms(np.abs(dmim_single_mut), pwm_list, pwm_dict)
            
            # ignore long pwms
            if True:
                current_indices = np.where(keep > 0)[0].tolist()
                for idx in current_indices:
                    if pwm_list[idx].weights.shape[1] > 15:
                        keep[idx] = 0
            
            # and add to pwm_vector
            pwm_vector += keep
            
            print "after similarity"
            indices = np.where(pwm_vector > 0)[0].tolist()
            print [pwm_list[k].name for k in indices]
            
            #import ipdb
            #ipdb.set_trace()
            

    # return pwm vector
    pwm_vector = pwm_vector.astype(int)
    
    return pwm_vector


def generate_networks(h5_file, pwm_vector_key, task_indices, pwm_list, pwm_dict):
    """create directed networks
    """
    from scipy.stats import ttest_1samp
        
    pwm_score_prefix = "pwm-scores"
    mut_score_prefix = "dmim-scores"

    node_size_scaling = 10000
    edge_weight_scaling = 1000

    cutoff = 0.0002

    with h5py.File(h5_file, "r") as hf:

        # get the pwm vector
        pwm_vector = hf[pwm_vector_key][:]
        indices = np.where(pwm_vector > 0)[0].tolist()
        pwm_names = [pwm_list[k].name for k in indices]
        pwm_names = [name.split(".")[0].split("_")[1] for name in pwm_names]

        delta_logits = hf["delta_logits"][:]
    
        # first generate the positions by using global mutational scores
        for i in xrange(len(task_indices)):

            task_idx = task_indices[i]
            task_mut_key = "{}.taskidx-{}".format(mut_score_prefix, task_idx)
            task_mut_scores = np.amax(np.abs(hf[task_mut_key]), axis=1) # {N, M}
                
            if i == 0:
                global_mut_scores = task_mut_scores
            else:
                global_mut_scores = np.maximum(global_mut_scores, task_mut_scores)

        # generate layout

        # then use pwm vector to select out the desired scores and reduce to adjacency matrix
        for i in xrange(len(task_indices)):
            print i

            task_idx = task_indices[i]
            task_pwm_key = "{}.taskidx-{}".format(pwm_score_prefix, task_idx)
            task_mut_key = "{}.taskidx-{}".format(mut_score_prefix, task_idx)

            mut_scores = hf[task_mut_key][:]
            
            # TODO make a cutoff mask per mutational state
            for mut_i in xrange(hf[task_mut_key].shape[1]):
                ttest_results = ttest_1samp(hf[task_mut_key][:,mut_i,:], 0) # {N, all_motifs}
                keep = pwm_vector * (ttest_results[1] < cutoff)
                mut_scores[:,mut_i,:] = (keep > 0) * mut_scores[:,mut_i,:]

            # set up scores
            pwm_scores = hf[task_pwm_key][:,pwm_vector > 0]
            mut_scores = mut_scores[:,:,pwm_vector > 0]
            mut_names = hf[task_mut_key].attrs["pwm_mut_names"]

            # calculate means
            pwm_means = np.mean(pwm_scores, axis=0) # {motifs}
            mut_means = np.mean(mut_scores, axis=0) # {mut, motifs}

            # node list
            node_list = []
            for node_idx in xrange(len(pwm_names)):
                node = (pwm_names[node_idx], {"size": node_size_scaling * pwm_means[node_idx]})
                node_list.append(node)
                
            # TRY: Nodes with delta logits
            delta_logit_scores = np.mean(np.abs(delta_logits[:,i,:]), axis=0) # {mut}
            node_list = []
            for node_idx in xrange(delta_logit_scores.shape[0]):
                if mut_names[node_idx] in pwm_names:
                    node = (mut_names[node_idx], {"size": 10*delta_logit_scores[node_idx]})
                    node_list.append(node)
                
            # make the edge list
            # edge list is a dictionary 3-tuple, (from_node, to_node, attrs)
            edge_list = []
            for mut_i in xrange(mut_means.shape[0]):

                # this limits which nodes are in the network
                if mut_names[mut_i] not in pwm_names:
                    continue
                
                for response_j in xrange(mut_means.shape[1]):
                    # TODO need to threshold out the non-passing values

                    if pwm_names[response_j] not in mut_names:
                        continue
                    
                    if mut_means[mut_i, response_j] == 0:
                        continue
                    
                    from_node = mut_names[mut_i]
                    to_node = pwm_names[response_j]
                    edge_attrs = {"weight": edge_weight_scaling * abs(mut_means[mut_i,response_j])}
                    edge = (from_node, to_node, edge_attrs)
                    edge_list.append(edge)

            # generate a graph
            task_graph = nx.MultiDiGraph()
            task_graph.add_nodes_from(node_list)
            task_graph.add_edges_from(edge_list)

            #print task_graph.nodes()
            #print task_graph.edges()
            
            # plot graph
            node_size_dict = nx.get_node_attributes(task_graph, "size")
            edge_weight_dict = nx.get_edge_attributes(task_graph, "weight")
            #print node_size_dict
            #print edge_weight_dict

            ordered_node_sizes = [node_size_dict[node]
                                  for node in task_graph.nodes()]
            ordered_line_widths = [edge_weight_dict[(node1, node2, 0)]
                                   for node1, node2 in task_graph.edges()]
            
            f = plt.figure()
            nx.draw(
                task_graph,
                #pos=positions,
                pos=nx.drawing.nx_pydot.pydot_layout(task_graph, prog="dot"),
                #pos=pydot_layout(G),
                with_labels=True,
                node_color="orange",
                node_size=10000*(ordered_node_sizes/sum(ordered_node_sizes)),
                edge_color="black",
                linewidths=100*(ordered_line_widths/sum(ordered_line_widths)),
                font_size=3)
            #plt.xlim(-1,1)
            #plt.ylim(-1,1)
            f.savefig("{}.network.pdf".format(task_mut_key))

            nx.write_gml(task_graph, "{}.graph.xml".format(task_mut_key), stringizer=str)
            

    return None


def make_network_from_adjacency(adj_array, pwm_names, node_sizes, gml_file):
    """ given adjacency and pwm names, generate a networkx and gml file
    """
    node_scaling = 1.0
    edge_scaling = 1.0
    
    # node list
    nodes = dict(zip(pwm_names, node_sizes))
    node_list = []
    for node_name in nodes.keys():
        node = (node_name, {"size": node_scaling*nodes[node_name]})
        node_list.append(node)
        
    # edge list
    edge_list = []
    for i in xrange(adj_array.shape[0]):
        for j in xrange(adj_array.shape[1]):
            val = adj_array[i,j]
            if val != 0:
                from_node = pwm_names[i]
                to_node = pwm_names[j]
                edge_attrs = {"weight": edge_scaling * abs(val)}
                edge = (from_node, to_node, edge_attrs)
                edge_list.append(edge)

    # build graph
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node_list)
    graph.add_edges_from(edge_list)

    # write out
    nx.write_gml(graph, gml_file, stringizer=str)
    
    return None


def generate_grammars_from_dmim(results_h5_file, inference_tasks, pwm_list, cutoff=0.05):
    """given the h5 file results, extract the grammar results
    """
    from scipy.stats import ttest_1samp
    from tronn.interpretation.motifs import reduce_pwms
    from tronn.interpretation.motifs import correlate_pwms
    
    dmim_prefix = "dmim-scores"
    pwm_prefix = "pwm-scores"

    # prep by keeping hclust for pwms
    cor_filt_mat, distances = correlate_pwms(
        pwm_list,
        cor_thresh=0.3,
        ncor_thresh=0.2,
        num_threads=24)
    hclust = linkage(squareform(1 - distances), method="ward")
    
    # extract clusters
    with h5py.File(results_h5_file, "r") as hf:
        num_clusters = hf["manifold_clusters"].shape[1]
        cluster_by_example = range(num_clusters)
        #cluster_by_example = list(set(hf["manifold_clusters.onehot"][:].tolist()))
        #num_clusters = len(cluster_by_example)
        master_pwm_vector = hf["master_pwm_vector"][:]

    # set up output arrays
    # TODO clean this up!
    dmim_results = np.zeros((
        num_clusters,
        len(inference_tasks),
        np.sum(master_pwm_vector > 0),
        len(pwm_list)))

    pwm_results = np.zeros((
        num_clusters,
        len(inference_tasks),
        np.sum(master_pwm_vector > 0)))

    label_results = np.zeros((
        num_clusters,
        len(inference_tasks)))
    
    prob_results = np.zeros((
        num_clusters,
        len(inference_tasks)))
    
    logit_results = np.zeros((
        num_clusters,
        len(inference_tasks)))

    delta_prob_results = np.zeros((
        num_clusters,
        len(inference_tasks),
        np.sum(master_pwm_vector > 0)))

    # TODO keep track of individual pwm vectors for each cluster.
    cluster_pwms_results = np.zeros((
        num_clusters,
        np.sum(master_pwm_vector >0)))
    
    # for each cluster, for each task, extract subset
    for cluster_i in xrange(num_clusters):
        cluster = cluster_by_example[cluster_i]
        print cluster
        with h5py.File(results_h5_file, "r") as hf:
            #in_cluster = hf["manifold_clusters.onehot"][:] == cluster
            in_cluster = hf["manifold_clusters"][:,cluster_i] == 1
            delta_logits = hf["delta_logits"][:][in_cluster] # {N, logit, mut}
            labels = hf["labels"][:][in_cluster][:,inference_tasks]
            logits = hf["logits"][:][in_cluster][:,inference_tasks] # {N, logit}
            probs = hf["probs"][:][in_cluster][:,inference_tasks]
            metadata = hf["example_metadata"][:][in_cluster]

        # save out the metadata as a bed file
        metadata_file = "{0}.cluster-{1}.metadata.txt".format(dmim_prefix, cluster)
        print metadata_file
        np.savetxt(metadata_file, metadata, fmt="%s", delimiter="\t")
        
        # make bed from the active region
        metadata_bed = "{0}.cluster-{1}.metadata.bed".format(dmim_prefix, cluster)
        make_bed = (
            "cat {0} | "
            "awk -F ';' '{{ print $2 }}' | "
            "awk -F '=' '{{ print $2 }}' | "
            "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
            "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
            "sort -k1,1 -k2,2n | "
            "bedtools merge -i stdin "
            "> {1}").format(
                metadata_file, metadata_bed)
        os.system(make_bed)
        
        #cluster_pwms = np.zeros((np.sum(in_cluster > 0), master_pwm_vector.shape[0]))
        cluster_pwms = np.zeros(master_pwm_vector.shape)
        
        for task_j in xrange(len(inference_tasks)):

            task_idx = inference_tasks[task_j]
            task_dmim_prefix = "{}.taskidx-{}".format(dmim_prefix, task_idx)
            task_pwm_prefix = "{}.taskidx-{}".format(pwm_prefix, task_idx)
            
            with h5py.File(results_h5_file, "r") as hf:
                dmim_scores = hf[task_dmim_prefix][:][in_cluster]
                pwm_scores = hf[task_pwm_prefix][:][in_cluster]

            print dmim_scores.shape
            print pwm_scores.shape

            # keep dmim results (sum) and pwm vector of things that are above importance thresh
            dmim_results[cluster_i,task_j,:,:] = np.sum(dmim_scores, axis=0) # {mut, motif}
            pwm_vector = reduce_pwms(pwm_scores, hclust, pwm_list, std_thresh=1)
            cluster_pwms = np.maximum(cluster_pwms, np.sum(pwm_scores, axis=0)) # {motif}
            #cluster_pwms += pwm_scores
            indices = np.where(pwm_vector > 0)[0].tolist()
            print [pwm_list[k].name for k in indices]

            # for each single mutant in set, check which ones responded
            for mut_k in xrange(dmim_scores.shape[1]):
                mut_data = dmim_scores[:,mut_k,:]
                ttest_results = ttest_1samp(mut_data, 0)
                #keep = pwm_vector * (ttest_results[1] < cutoff)
                if True:
                    #keep = (ttest_results[1] < cutoff) * pwm_vector
                    keep = ttest_results[1] < cutoff
                    dmim_results[cluster_i,task_j,mut_k,:] = np.multiply(
                        keep, dmim_results[cluster_i,task_j,mut_k,:])
                if False:
                    task_pwm_indices = np.where(pwm_vector > 0)[0]
                    task_mut_indices = np.where(master_pwm_vector > 0)[0]
                    if task_mut_indices[mut_k] not in task_pwm_indices:
                        dmim_results[cluster_i,task_j,mut_k,:] = np.multiply(
                            [0], dmim_results[cluster_i,task_j,mut_k,:])

                # save out delta logits
                mut_indices = np.where(master_pwm_vector > 0)[0]
                mut_motif_present = np.where(pwm_scores[:,mut_indices[mut_k]] > 0)
                #delta_prob_results[cluster_i,task_j,mut_k] = np.mean(
                #        delta_logits[mut_motif_present,task_j,mut_k])

                probs_orig = np.expand_dims(expit(logits), axis=2)
                probs_mut = expit(np.expand_dims(logits, axis=2) + delta_logits)
                delta_probs = probs_mut - probs_orig
                delta_prob_results[cluster_i,task_j,mut_k] = np.mean(
                    delta_probs[mut_motif_present,task_j,mut_k])
                        
            # also save out pwm results
            pwm_results[cluster_i, task_j,:] = np.sum(pwm_scores, axis=0)[master_pwm_vector > 0]

            # TODO - for each motif, get the best global hit position
            # and save out a bed file centered on the motif position.
            
            
        # save out labels, logits, and delta logit results
        label_results[cluster_i,:] = np.mean(labels, axis=0)
        prob_results[cluster_i,:] = np.mean(probs, axis=0)
        logit_results[cluster_i,:] = np.mean(logits, axis=0)
            #delta_prob_results[cluster_i,:,:] = np.mean(
            #    expit(np.expand_dims(logits, axis=2) + delta_logits), axis=0)

            # TODO - ideally only average per mutation on regions that have that motif
            # in orig pwm scores
        #delta_prob_results[cluster_i,:,:] = np.mean(delta_logits, axis=0)

        # save out cluster pwms
        # TODO - do this using a max matrix across pwm scores
        #with h5py.File(results_h5_file, "r") as hf:
        #    in_cluster = hf["manifold_clusters"][:,cluster_i] == 1
        #    global_prefix = "{}.taskidx-{}".format(pwm_prefix, len(inference_tasks))
        #    print global_prefix
        #    cluster_pwms = hf[global_prefix][:][in_cluster]
        #cluster_pwms_results[cluster_i,:] = reduce_pwms(
        #    pwm_global_scores, hclust, pwm_list, std_thresh=1)[master_pwm_vector > 0]
        #cluster_pwms_results[cluster_i,:] = reduce_pwms(
        #    cluster_pwms, hclust, pwm_list, std_thresh=1)[master_pwm_vector > 0]

        # get best pwms
        nonzero_means = np.hstack((cluster_pwms, -cluster_pwms))
        mean_val = np.mean(nonzero_means)
        std_val = np.std(nonzero_means)
        sig_pwms = (cluster_pwms > (mean_val + (std_val * 2))).astype(int)
        
        #scores = np.sum(cluster_pwms, axis=0).tolist()
        #test = dict(zip([pwm.name for pwm in pwm_list], cluster_pwms))
        #for key in test.keys():
        #    if "TEAD" in key:
        #        print key, test[key]
        #    elif "CEBP" in key:
        #        print key, test[key]
        #    elif "RELA" in key:
        #        print key, test[key]
        #print test
        #sig_pwms = sd_cutoff(cluster_pwms, std_thresh=2)
        indices = np.where(sig_pwms > 0)[0].tolist()
        pwm_names = [pwm_list[k].name for k in indices]
        #import ipdb
        #ipdb.set_trace()
        
        cluster_pwms_results[cluster_i,:] = sig_pwms[master_pwm_vector > 0]
        #sd_cutoff(cluster_pwms, std_thresh=2)[master_pwm_vector > 0]

        for task_j in xrange(len(inference_tasks)):
            # edges
            adjacency_array = dmim_results[cluster_i,task_j,cluster_pwms_results[cluster_i,:]>0,:]
            adjacency_array = adjacency_array[:,master_pwm_vector>0]
            adjacency_array = adjacency_array[:,cluster_pwms_results[cluster_i,:]>0]
            print adjacency_array.shape
            
            # nodes
            node_sizes = pwm_results[cluster_i,task_j,cluster_pwms_results[cluster_i,:] > 0]
            print node_sizes

            # node names
            mut_indices = np.where(master_pwm_vector > 0)[0]
            mut_indices = mut_indices[cluster_pwms_results[cluster_i,:] > 0]
            pwm_names = [pwm_list[k].name.split("_")[1].split(".")[0] for k in mut_indices]
            print pwm_names

            gml_file = "{}.cluster_{}.task_{}.gml".format(results_h5_file.split(".h5")[0], cluster_i, task_j)
            make_network_from_adjacency(adjacency_array, pwm_names, node_sizes, gml_file)
        
    # save out dmim results
    dataset_key = "{}.merged".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=dmim_results)

    # get ordering
    flattened = np.reshape(
        np.transpose(dmim_results, axes=[2, 0, 1, 3]),
        [dmim_results.shape[2], -1])
    hclust_dmim = linkage(flattened, method="ward")
    ordered_indices = leaves_list(hclust_dmim)
    
    # get subset (mutated set) and reorder
    dmim_results_master = dmim_results[:,:,:,master_pwm_vector > 0]
    dmim_results_master = dmim_results_master[:,:,ordered_indices,:]
    dmim_results_master = dmim_results_master[:,:,:,ordered_indices]

    mut_indices = np.where(master_pwm_vector > 0)[0]
    mut_indices = mut_indices[ordered_indices]
    pwm_names = [pwm_list[i].name for i in mut_indices]
    
    dataset_key = "{}.merged.master".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=dmim_results_master)
        hf[dataset_key].attrs["pwm_names"] = pwm_names

    # save out pwm results
    dataset_key = "{}.pwm_scores_by_cluster".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=pwm_results[:,:,ordered_indices])
        hf[dataset_key].attrs["pwm_names"] = pwm_names

    # save out labels and logits
    dataset_key = "{}.labels".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=label_results)

    dataset_key = "{}.probs".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=prob_results)

    dataset_key = "{}.logits".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=logit_results)


    # save out delta logits
    # subtract from logits and then convert to probs?
    dataset_key = "{}.mut_probs".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=delta_prob_results[:,:,ordered_indices])

    # save out cluster pwms
    dataset_key = "{}.cluster_pwm_vectors".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key,
                          data=cluster_pwms_results[:,ordered_indices])
        
    
    return None


def aggreagate_dmim_results(
        h5_file,
        cluster_key,
        inference_task_indices,
        pwm_list,
        cutoff=0.05,
        other_task_indices=[],
        soft_clustering=True):
    """given the h5 file results, extract the grammar results

    What is happening here???
    
    per cluster, extract significant pwms and then save only those (using a pwm vector)
    for these results, just need the delta in logits/probs (across all tasks)
    and then also the adjacency results
    
    Returns:
    {cluster, M, tasks} with pwm vectors {cluster, M} - delta logits
    {cluster, M, M} with same pwm vectors as above - adjacency results

    # do this for various task index sets

    everything else does not need to be touched.
    

    
    """
    from scipy.stats import ttest_1samp
    from tronn.interpretation.motifs import reduce_pwms
    from tronn.interpretation.motifs import correlate_pwms
    
    dmim_prefix = "dmim-scores"
    pwm_prefix = "pwm-scores"

    # prep by keeping hclust for pwms (for reducing similar pwms)
    cor_filt_mat, distances = correlate_pwms(
        pwm_list,
        cor_thresh=0.3,
        ncor_thresh=0.2,
        num_threads=24)
    hclust = linkage(squareform(1 - distances), method="ward")
    
    # extract clusters and master pwm vector
    with h5py.File(results_h5_file, "r") as hf:

        # cluster ids
        if not soft_clustering:
            clusters = hf[cluster_key][:,cluster_col]
            cluster_ids = sorted(list(set(clusters.tolist())))
        else:
            clusters = hf[cluster_key][:]
            cluster_ids = range(hf[cluster_key].shape[1])

        # master pwm vector
        master_pwm_vector = hf["master_pwm_vector"][:]

    # set up output arrays {cluster, task, mut motif, response motif} and {cluster, motif}
    delta_logits_by_cluster = np.zeros((
        num_clusters,
        np.sum(master_pwm_vector > 0),
        len(task_indices))) # {cluster, tasks, M} collect across all
    dmim_results_by_cluster = np.zeros((
        num_clusters,
        len(inference_tasks),
        np.sum(master_pwm_vector > 0),
        len(pwm_list))) # {cluster, task, response motifs}
    cluster_pwm_vectors = np.zeros((
        num_clusters,
        np.sum(master_pwm_vector >0)))
    
    # for each cluster, for each task, extract subset
    for cluster_idx in xrange(len(cluster_ids)):
        cluster_id = cluster_ids[cluster_idx]

        # first extract the inference task set (that has adjacency matrix)
        
        
        # then extract the other task index sets (that don't have importance scores)
        
        with h5py.File(results_h5_file, "r") as hf:
            #in_cluster = hf["manifold_clusters.onehot"][:] == cluster
            in_cluster = hf["manifold_clusters"][:,cluster_i] == 1
            delta_logits = hf["delta_logits"][:][in_cluster] # {N, logit, mut}

        # save out the metadata as a bed file
        metadata_file = "{0}.cluster-{1}.metadata.txt".format(dmim_prefix, cluster)
        print metadata_file
        np.savetxt(metadata_file, metadata, fmt="%s", delimiter="\t")
        
        # make bed from the active region
        metadata_bed = "{0}.cluster-{1}.metadata.bed".format(dmim_prefix, cluster)
        make_bed = (
            "cat {0} | "
            "awk -F ';' '{{ print $2 }}' | "
            "awk -F '=' '{{ print $2 }}' | "
            "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
            "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
            "sort -k1,1 -k2,2n | "
            "bedtools merge -i stdin "
            "> {1}").format(
                metadata_file, metadata_bed)
        os.system(make_bed)
        
        #cluster_pwms = np.zeros((np.sum(in_cluster > 0), master_pwm_vector.shape[0]))
        cluster_pwms = np.zeros(master_pwm_vector.shape)
        
        for task_j in xrange(len(inference_tasks)):

            task_idx = inference_tasks[task_j]
            task_dmim_prefix = "{}.taskidx-{}".format(dmim_prefix, task_idx)
            task_pwm_prefix = "{}.taskidx-{}".format(pwm_prefix, task_idx)
            
            with h5py.File(results_h5_file, "r") as hf:
                dmim_scores = hf[task_dmim_prefix][:][in_cluster]
                pwm_scores = hf[task_pwm_prefix][:][in_cluster]

            print dmim_scores.shape
            print pwm_scores.shape

            # keep dmim results (sum) and pwm vector of things that are above importance thresh
            dmim_results[cluster_i,task_j,:,:] = np.sum(dmim_scores, axis=0) # {mut, motif}
            pwm_vector = reduce_pwms(pwm_scores, hclust, pwm_list, std_thresh=1)
            cluster_pwms = np.maximum(cluster_pwms, np.sum(pwm_scores, axis=0)) # {motif}
            #cluster_pwms += pwm_scores
            indices = np.where(pwm_vector > 0)[0].tolist()
            print [pwm_list[k].name for k in indices]

            # for each single mutant in set, check which ones responded
            for mut_k in xrange(dmim_scores.shape[1]):
                mut_data = dmim_scores[:,mut_k,:]
                ttest_results = ttest_1samp(mut_data, 0)
                #keep = pwm_vector * (ttest_results[1] < cutoff)
                if True:
                    #keep = (ttest_results[1] < cutoff) * pwm_vector
                    keep = ttest_results[1] < cutoff
                    dmim_results[cluster_i,task_j,mut_k,:] = np.multiply(
                        keep, dmim_results[cluster_i,task_j,mut_k,:])
                if False:
                    task_pwm_indices = np.where(pwm_vector > 0)[0]
                    task_mut_indices = np.where(master_pwm_vector > 0)[0]
                    if task_mut_indices[mut_k] not in task_pwm_indices:
                        dmim_results[cluster_i,task_j,mut_k,:] = np.multiply(
                            [0], dmim_results[cluster_i,task_j,mut_k,:])

                # save out delta logits
                mut_indices = np.where(master_pwm_vector > 0)[0]
                mut_motif_present = np.where(pwm_scores[:,mut_indices[mut_k]] > 0)
                #delta_prob_results[cluster_i,task_j,mut_k] = np.mean(
                #        delta_logits[mut_motif_present,task_j,mut_k])

                probs_orig = np.expand_dims(expit(logits), axis=2)
                probs_mut = expit(np.expand_dims(logits, axis=2) + delta_logits)
                delta_probs = probs_mut - probs_orig
                delta_prob_results[cluster_i,task_j,mut_k] = np.mean(
                    delta_probs[mut_motif_present,task_j,mut_k])
                        
            # also save out pwm results
            pwm_results[cluster_i, task_j,:] = np.sum(pwm_scores, axis=0)[master_pwm_vector > 0]

            # TODO - for each motif, get the best global hit position
            # and save out a bed file centered on the motif position.
            
        # get best pwms
        nonzero_means = np.hstack((cluster_pwms, -cluster_pwms))
        mean_val = np.mean(nonzero_means)
        std_val = np.std(nonzero_means)
        sig_pwms = (cluster_pwms > (mean_val + (std_val * 2))).astype(int)

        #sig_pwms = sd_cutoff(cluster_pwms, std_thresh=2)
        indices = np.where(sig_pwms > 0)[0].tolist()
        pwm_names = [pwm_list[k].name for k in indices]
        #import ipdb
        #ipdb.set_trace()
        
        cluster_pwms_results[cluster_i,:] = sig_pwms[master_pwm_vector > 0]
        #sd_cutoff(cluster_pwms, std_thresh=2)[master_pwm_vector > 0]

        for task_j in xrange(len(inference_tasks)):
            # edges
            adjacency_array = dmim_results[cluster_i,task_j,cluster_pwms_results[cluster_i,:]>0,:]
            adjacency_array = adjacency_array[:,master_pwm_vector>0]
            adjacency_array = adjacency_array[:,cluster_pwms_results[cluster_i,:]>0]
            print adjacency_array.shape
            
            # nodes
            node_sizes = pwm_results[cluster_i,task_j,cluster_pwms_results[cluster_i,:] > 0]
            print node_sizes

            # node names
            mut_indices = np.where(master_pwm_vector > 0)[0]
            mut_indices = mut_indices[cluster_pwms_results[cluster_i,:] > 0]
            pwm_names = [pwm_list[k].name.split("_")[1].split(".")[0] for k in mut_indices]
            print pwm_names

            gml_file = "{}.cluster_{}.task_{}.gml".format(results_h5_file.split(".h5")[0], cluster_i, task_j)
            make_network_from_adjacency(adjacency_array, pwm_names, node_sizes, gml_file)
        
    # save out dmim results
    dataset_key = "{}.merged".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=dmim_results)

    # get ordering
    flattened = np.reshape(
        np.transpose(dmim_results, axes=[2, 0, 1, 3]),
        [dmim_results.shape[2], -1])
    hclust_dmim = linkage(flattened, method="ward")
    ordered_indices = leaves_list(hclust_dmim)
    
    # get subset (mutated set) and reorder
    dmim_results_master = dmim_results[:,:,:,master_pwm_vector > 0]
    dmim_results_master = dmim_results_master[:,:,ordered_indices,:]
    dmim_results_master = dmim_results_master[:,:,:,ordered_indices]

    mut_indices = np.where(master_pwm_vector > 0)[0]
    mut_indices = mut_indices[ordered_indices]
    pwm_names = [pwm_list[i].name for i in mut_indices]
    
    dataset_key = "{}.merged.master".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=dmim_results_master)
        hf[dataset_key].attrs["pwm_names"] = pwm_names

    # save out pwm results
    dataset_key = "{}.pwm_scores_by_cluster".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=pwm_results[:,:,ordered_indices])
        hf[dataset_key].attrs["pwm_names"] = pwm_names

    # save out delta logits
    # subtract from logits and then convert to probs?
    dataset_key = "{}.mut_probs".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key, data=delta_prob_results[:,:,ordered_indices])

    # save out cluster pwms
    dataset_key = "{}.cluster_pwm_vectors".format(dmim_prefix)
    with h5py.File(results_h5_file, "a") as hf:
        if hf.get(dataset_key) is not None:
            del hf[dataset_key]
        hf.create_dataset(dataset_key,
                          data=cluster_pwms_results[:,ordered_indices])
        
    
    return None
