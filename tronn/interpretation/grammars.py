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

import phenograph

import networkx as nx
from networkx.drawing.nx_pydot import pydot_layout

from scipy.stats import pearsonr


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
            node_dict = {}
            edge_dict = {}

            # go through nodes TODO check for presence in pwm file?
            line = fp.readline().strip()
            assert line.startswith("#nodes")
            while True:
                line = fp.readline().strip()
                if line.startswith("#edges") or line == "": break

                fields = line.split()
                node_dict[fields[0]] = float(fields[1])

            # go through edges
            assert line.startswith("#edges")
            while True:
                line = fp.readline().strip()
                if line.startswith(">") or line == "": break

                fields = line.split()
                edge_dict[(fields[0], fields[1])] = float(fields[2])

            print header
            grammar = Grammar(pwm_file, node_dict, edge_dict, param_string, name=header)

            if as_dict:
                grammars[header] = grammar
            else:
                grammars.append(grammar)

    return grammars



class Grammar(object):

    def __init__(self, pwm_file, node_dict, edge_dict, param_string, name=None, threshold=None):
        self.name = name
        self.pwms = read_pwm_file(pwm_file)
        self.param_string = param_string
        self.nodes = node_dict
        self.edges = edge_dict

        # set up 1D pwm vector
        self.pwm_vector = np.zeros((len(self.pwms)))
        for pwm_idx in xrange(len(self.pwms)):
            pwm = self.pwms[pwm_idx]
            try:
                val = self.nodes[pwm.name]
                self.pwm_vector[pwm_idx] = val
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
        node_size = [node_size_dict[node] for node in G.nodes]
    print node_size

    # set up edge weights
    edge_to_weight = nx.get_edge_attributes(G, "value")
    edge_weights = [edge_to_weight[edge]*0.1 for edge in G.edges]
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
