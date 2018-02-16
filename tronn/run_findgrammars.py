# description: test function for a multitask interpretation pipeline


import matplotlib
matplotlib.use("Agg")

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from collections import Counter

from tronn.graphs import TronnGraph
from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.datalayer import load_step_scaled_data_from_filename_list
from tronn.nets.nets import net_fns

from tronn.interpretation.interpret import interpret

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import read_pwm_file
from tronn.interpretation.motifs import setup_pwms
from tronn.interpretation.motifs import setup_pwm_metadata

from tronn.interpretation.grammars import get_significant_correlations
from tronn.interpretation.grammars import reduce_corr_mat_by_motif_similarity
from tronn.interpretation.grammars import get_networkx_graph
from tronn.interpretation.grammars import plot_corr_as_network
from tronn.interpretation.grammars import plot_corr_on_fixed_graph
from tronn.interpretation.grammars import get_significant_motifs

from tronn.interpretation.grammars import read_grammar_file

from tronn.interpretation.networks import separate_and_save_components

import networkx as nx

import phenograph

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def h5_dataset_to_text_file(h5_file, key, text_file, col_keep_indices, colnames):
    """Grab a dataset out of h5 (2D max) and save out to a text file
    """
    with h5py.File(h5_file, "r") as hf:
        dataset = hf[key][:][:,np.array(col_keep_indices)]
        
        # set up dataframe and save out
        dataset_df = pd.DataFrame(dataset, index=hf["example_metadata"][:][:,0], columns=colnames)
        dataset_df.to_csv(text_file, sep='\t')

    return None


def quick_filter_old(pwm_hits_df):
    """basic filters
    """
    # remove rows that are zero
    pwm_hits_df_binary = (pwm_hits_df > 0).astype(int) #* pwm_hits_df
    pwm_hits_df = pwm_hits_df.loc[~(pwm_hits_df_binary==0).all(axis=1)] # remove elements with no hits
    # TODO - this is not exactly right - threshold? use real vals?
    pwm_hits_df_binary = pwm_hits_df_binary.loc[~(pwm_hits_df_binary==0).all(axis=1)] # remove elements with no hits
    
    # remove low scoring motifs
    #pwm_hits_df = pwm_hits_df.loc[:, np.sum(pwm_hits_df_binary, axis=0) > 100]
    #pwm_hits_df_binary = pwm_hits_df_binary.loc[:, np.sum(pwm_hits_df_binary, axis=0) > 100]
            
    return pwm_hits_df, pwm_hits_df_binary


def quick_filter(pwm_hits_df, threshold=0.002):
    """basic filters
    """
    # remove rows that are zero
    pwm_hits_df_mask = (pwm_hits_df > threshold).astype(int) #* pwm_hits_df
    pwm_hits_df = pwm_hits_df * pwm_hits_df_mask
    pwm_hits_df = pwm_hits_df.loc[~(pwm_hits_df_mask==0).all(axis=1)] # remove elements with no hits
    # TODO - this is not exactly right - threshold? use real vals?
    pwm_hits_df_mask = pwm_hits_df_mask.loc[~(pwm_hits_df_mask==0).all(axis=1)] # remove elements with no hits
    
    # remove low scoring motifs
    #pwm_hits_df = pwm_hits_df.loc[:, np.sum(pwm_hits_df_binary, axis=0) > 100]
    #pwm_hits_df_binary = pwm_hits_df_binary.loc[:, np.sum(pwm_hits_df_binary, axis=0) > 100]
            
    return pwm_hits_df, pwm_hits_df_mask


def build_grammar_tree(filter_pwms, mat_df, min_size=500):
    """Given a matrix of data, make a tree until leaves hit min_size.
     do this recursively.
    """
    # first filter with filter pwms

    # then with the resulting matrix, get num regions per motif


    # for each motif (until you hit min size):
    # add to filter_pwms and call build_grammar_tree to get subtree

    # attach subtree to tree (figure out appropriate data structure)
    # tree structure - nested lists? probably easiest just to make a simple Tree class (easy to find an example on stack overflow)

    # return the tree
    return


# TODO after building the tree, recurse through it to extract grammars.



def phenograph_cluster(mat_file, sorted_mat_file):
    """Use to quickly get a nicely clustered (sorted) output file to visualize examples
    """
    # read in file and adjust as needed
    mat_df = pd.read_table(mat_file, index_col=0)
    #mat_df = mat_df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
    #mat_df = mat_df.loc[~(mat_df==0).all(axis=1)] # remove elements with no hits
    
    # run Louvain here for basic clustering to visually look at patterns
    #mat_df, mat_df_binary = quick_filter(mat_df)
    print "total examples used:", mat_df.shape

    # quick test: filter for 1 motif and look
    pwm_key = "PWM_HCLUST_134.UNK.0.A" # CEBPB
    pwm_key = "PWM_HCLUST_121.UNK.0.A" # TFAP2B
    pwm_key = "PWM_HCLUST_131.UNK.0.A" # ETS
    
    #mat_df_binary = mat_df_binary[mat_df[pwm_key] > 0]
    #mat_df = mat_df[mat_df[pwm_key] > 0]

    # do it the other direction too
    #mat_df = mat_df.loc[:, np.sum(mat_df, axis=0) > 200]
    
    #pwm_communities, pwm_graph, pwm_Q = phenograph.cluster(mat_df.transpose())
    #mat_df.loc["community"] = pwm_communities
    #mat_df = mat_df.sort_values("community", axis=1)
    #mat_df = mat_df.drop("community", axis=0)
    
    # use top motifs?
    communities, graph, Q = phenograph.cluster(mat_df)

    # save out the sorted info into a new mat sorted by community
    mat_df["community"] = communities
    mat_df = mat_df.sort_values("community", axis=0)
    mat_df.to_csv(sorted_mat_file, sep='\t')
    mat_df = mat_df.drop(["community"], axis=1)

    return None


def generate_motif_modules_hclust(mat_file, motif_module_file, pwm_dict, corr_cutoff=0.1):
    """given a matrix, hclust the columns and then aggregate based on jaccard
    """
    from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
    from scipy.spatial.distance import squareform
    
    # get correlations
    corr_file = "{}.corr.txt".format(mat_file.split(".txt")[0])
    if not os.path.isfile(corr_file):
        get_correlation_file(
            mat_file,
            corr_file,
            corr_method="continuous_jaccard")

    # use the correlations to hclust
    cor_df = pd.read_table(corr_file, index_col=0)

    # hclust
    hclust = linkage(squareform(1 - cor_df.as_matrix()), method="ward")

    # now go through the hclust
    pwms = [[pwm] for pwm in cor_df.columns.tolist()]
    
    core_pwm_groups = []
    for i in xrange(hclust.shape[0]):
        idx1, idx2, dist, cluster_size = hclust[i,:]

        pwm_group1 = pwms[int(idx1)]
        pwm_group2 = pwms[int(idx2)]

        if (pwm_group1 is None) and (pwm_group2 is None):
            pwms.append(None)
            continue
        elif (pwm_group1 is None):
            core_pwm_groups.append(pwm_group2)
            pwms.append(None)
            continue
        elif (pwm_group2 is None):
            core_pwm_groups.append(pwm_group1)
            pwms.append(None)
            continue

        # otherwise, check for good correlation score
        min_cor = 1.0
        for i in xrange(len(pwm_group1)):
            for j in xrange(len(pwm_group2)):
                rowname = pwm_group1[i]
                colname = pwm_group2[j]
                corr = cor_df[rowname][colname]
                if corr < min_cor:
                    min_cor = corr

        if min_cor >= corr_cutoff:
            # merge groups and store out
            new_group = pwm_group1 + pwm_group2
            pwms.append(new_group)
        else:
            # save out and put a none
            core_pwm_groups.append(pwm_group1)
            core_pwm_groups.append(pwm_group2)
            pwms.append(None)

    # TODO save this information out as list of pwm sets.
    # for future piece of code: then for every set to every set (in sequential
    # timepoints) - compare, to link through time.
    # somehow want to track overlap info
    motif_expressed = False
    expressed_core_pwm_groups = []
    for pwms in core_pwm_groups:
        for pwm in pwms:
            if not "UNK" in pwm_dict[pwm]:
                motif_expressed = True
        if motif_expressed:
            expressed_core_pwm_groups.append(pwms)
        motif_expressed = False
    core_pwm_groups = expressed_core_pwm_groups

    # flatten pwms
    pwms_flattened = []
    for pwms in core_pwm_groups:
        pwms_flattened += pwms

    # reorder matrix
    motif_cluster_reordered_file = "{}.motif_cluster_ordered.txt".format(mat_file.split(".txt")[0])
    mat_df = pd.read_table(mat_file, index_col=0)
    mat_df = mat_df[pwms_flattened+["community"]]
    mat_df.to_csv(motif_cluster_reordered_file, sep="\t")
    
    with open(motif_module_file, "w") as out:
        for i in xrange(len(core_pwm_groups)):
            pwms = core_pwm_groups[i]
            
            # determine total regions
            group_regions = mat_df[pwms]
            num_group_regions = group_regions.loc[~(group_regions<=0).any(axis=1)].shape[0]
            
            # and save out
            out.write("#num_regions={}\n".format(num_group_regions))
            for pwm in pwms:
                if not "UNK" in pwm_dict[pwm]:
                    out.write("{}\t{}\n".format(pwm, pwm_dict[pwm]))

            # also write separate files to throw into grammar matrix plot
            module_file = "{}.core_pwms_{}.regions_{}.txt".format(motif_module_file.split(".txt")[0], i, num_group_regions)
            with open(module_file, "w") as fp:
                for pwm in pwms:
                    if not "UNK" in pwm_dict[pwm]:
                        fp.write("{}\t{}\n".format(pwm, pwm_dict[pwm]))

    # TODO change this here?
    # replace motif sets with mean (to be cluster)
    # then recalculate correlation, save both out
    motif_group_means = pd.DataFrame()
    motif_group_means["community"] = mat_df["community"]
    for pwms in core_pwm_groups:
        name = ";".join(pwms)
        motif_group_df = mat_df[pwms]
        motif_group_means[name] = motif_group_df.mean(axis=1)

    # save regions x clusters out
    region_x_motifset_file = "{}.region_x_motifset.txt".format(mat_file.split(".txt")[0])
    motif_group_means.to_csv(region_x_motifset_file, sep="\t")

    # calculate correlations by jaccard
    corr_file = "{}.corr.txt".format(region_x_motifset_file.split(".txt")[0])
    if not os.path.isfile(corr_file):
        get_correlation_file(
            region_x_motifset_file,
            corr_file,
            corr_method="continuous_jaccard")
    
    return


def adjust_phenograph_communities(community_file, out_file, min_fract=0.03):
    """adjust communities that are too small or non communities
    minimum fraction of 0.03 means (out of 30k dynamic regions) at least ~1k regions.
    """
    mat_df = pd.read_table(community_file, index_col=0)

    communities = list(set(mat_df["community"].tolist()))
    new_community = len(communities)
    for community in communities:

        # non community regions get put into new community
        if int(community) == -1:
            mat_df.loc[mat_df["community"] == community, "community"] = new_community

        # too small community regions get put into new community
        community_df = mat_df[mat_df["community"] == community]
        if community_df.shape[0] / float(mat_df.shape[0]) < min_fract:
            mat_df.loc[mat_df["community"] == community, "community"] = new_community
            
    # save
    mat_df.to_csv(out_file, sep="\t")
            
    return None

def generate_motif_modules_w_phenograph(
        mat_file,
        pwm_x_pwm_corr_file,
        pwm_dict,
        sig_community_fract=0.01):
    """procedure for semi-iteratively producing modules of regions and motifs
    """
    # read in file and adjust as needed
    mat_df = pd.read_table(mat_file, index_col=0)
    print "total examples used:", mat_df.shape

    # for each community, determine which motifs are high in value
    # currently just doing a signal cutoff
    # TODO - try to fit a mixture of gaussians?
    pwms_to_keep = []
    community_names = list(set(mat_df["community"]))
    for community_name in community_names:
        community_df = mat_df.loc[mat_df["community"] == community_name]
        community_df = community_df.drop(["community"], axis=1)

        # only look at significantly sized communities
        community_fract = float(community_df.shape[0]) / mat_df.shape[0]
        if community_fract < sig_community_fract:
            continue
        
        # get the column sums
        community_pwm_sums_df = community_df.sum(axis=0)
        stdev = community_pwm_sums_df.std()
        mean = community_pwm_sums_df.mean()

        # cutoff is currently 1 stdev above the mean
        community_pwm_sums_thresholded_df = community_pwm_sums_df[community_pwm_sums_df > mean + stdev]

        pwms_to_keep += community_pwm_sums_thresholded_df.index.tolist()
        pwms_to_keep = list(set(pwms_to_keep))

    # then reduce to the pwms to keep
    mat_df = mat_df[pwms_to_keep]
    
    # recalculate phenograph
    recalc_file = "{}.recalc.phenograph.txt".format(mat_file.split(".txt")[0])
    if not os.path.isfile(recalc_file):
        communities, graph, Q = phenograph.cluster(mat_df)
        mat_df["community"] = communities
        mat_df.to_csv(recalc_file, sep="\t")

    # Now take in the phenograph results and adjust
    # TODO - adjust the communities that are too small to be None for their community
    # remember to change from -1 to some other value - maybe the next largest number?
    recalc_adj_file = "{}.adjusted.txt".format(recalc_file.split(".txt")[0])
    if not os.path.isfile(recalc_adj_file):
        adjust_phenograph_communities(recalc_file, recalc_adj_file, min_fract=0.03)

    # separate out clusters
    if False:
        data = pd.read_table(recalc_file, index_col=0)
        communities = list(set(data["community"]))
        for community in communities:
            community = int(community)
            if community == -1:
                continue

            community_data = data[data["community"] == community]

            if community_data.shape[0] < 500: # TODO fix this
                continue
            community_file = "{}.community-{}.txt".format(recalc_file.split(".txt")[0], community)
            community_data.to_csv(community_file, sep="\t")
            
        # use an hclust to determine the motif modules
        module_file = "{}.motif_modules.txt".format(recalc_file.split(".txt")[0])
        generate_motif_modules_hclust(recalc_file, module_file, pwm_dict)
    
    return



def get_correlation_file(
        mat_file,
        corr_file,
        corr_method="intersection_size", # continuous_jaccard, pearson
        corr_min=0.4,
        pval_thresh=0.05):
    """Given a matrix file, calculate correlations across the columns
    """
    mat_df = pd.read_table(mat_file, index_col=0)
    mat_df = mat_df.drop(["community"], axis=1)
            
    corr_mat, pval_mat = get_significant_correlations(
        mat_df.as_matrix(),
        corr_method=corr_method,
        corr_min=corr_min,
        pval_thresh=pval_thresh)
    
    corr_mat_df = pd.DataFrame(corr_mat, index=mat_df.columns, columns=mat_df.columns)
    corr_mat_df.to_csv(corr_file, sep="\t")

    return None


def get_max_corr_mat(correlation_files, pwm_names_filt):
    """Get max correlations across correlation files 
    """
    num_pwms = len(pwm_names_filt)
    # get the max correlations/signal strengths to set up global graph here
    max_corr_df = pd.DataFrame(
        data=np.zeros((num_pwms, num_pwms)),
        index=pwm_names_filt,
        columns=pwm_names_filt)
    #max_signals = np.zeros((num_pwms))
    
    for correlation_file in correlation_files:
        # load in corr mat file
        corr_mat_df = pd.read_table(correlation_file, index_col=0)
        #corr_mat_df.columns = [";".join([name.split("_")[0] for name in pwm_name_to_hgnc[pwm_name].split(";")])
        #                       for pwm_name in corr_mat_df.columns]
        #corr_mat_df.index = corr_mat_df.columns
        
        # apply max
        max_array = np.maximum(max_corr_df.as_matrix(), corr_mat_df.as_matrix())
        max_corr_df = pd.DataFrame(max_array, index=corr_mat_df.index, columns=corr_mat_df.columns)

    return max_corr_df


def plot_motif_network(
        regions_x_pwm_mat_file,
        pwm_x_pwm_corr_file,
        id_to_name,
        prefix,
        pwm_dict,
        positions,
        edge_cor_thresh=0.05,
        reduce_pwms=True):
    """Plot full motif network with signal strengths
    """
    # Extract signal strengths
    pwm_hits_df = pd.read_table(regions_x_pwm_mat_file, index_col=0)
    pwm_hits_df = pwm_hits_df.drop(["community"], axis=1)
    #pwm_hits_df = pwm_hits_df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
    #pwm_hits_df = pwm_hits_df.loc[~(pwm_hits_df==0).all(axis=1)] # remove elements with no hits

    # TODO: figure out how to get signal to noise score here - sum gets diluted by lots of noise
    # would there be any rationale to squaring?
    signal = np.sum(pwm_hits_df, axis=0)
    signal = 300 * signal / signal.max()

    # pwm name to signal
    id_to_signal_dict = dict(zip(signal.index, signal.as_matrix()))

    # hgnc to signal
    signal.index = [";".join([name.split("_")[0] for name in id_to_name[pwm_name].split(";")])
                           for pwm_name in signal.index]
    print signal.nlargest(n=10)
    node_size_dict = dict(zip(signal.index, signal.as_matrix()))

    # Get correlations
    corr_mat_df = pd.read_table(pwm_x_pwm_corr_file, index_col=0)

    # here, reduce pwms
    if reduce_pwms:
        reduced_corr_mat_file = "{}.reduced.txt".format(pwm_x_pwm_corr_file.split(".tmp")[0])
        corr_mat_df = reduce_corr_mat_by_motif_similarity(
            corr_mat_df,
            pwm_dict,
            id_to_signal_dict,
            edge_cor_thresh=edge_cor_thresh)
        

    # and adjust names after, need to match pwms
    corr_mat_df.columns = [";".join([name.split("_")[0] for name in id_to_name[pwm_name].split(";")])
                           for pwm_name in corr_mat_df.columns]
    corr_mat_df.index = corr_mat_df.columns
    
    # save this out to quickly plot later
    if reduce_pwms:
        corr_mat_df.to_csv(reduced_corr_mat_file, sep="\t")

    # plotting
    task_G = plot_corr_on_fixed_graph(
        corr_mat_df,
        positions,
        prefix,
        corr_thresh=edge_cor_thresh,
        node_size_dict=node_size_dict)

    return task_G


def run_apriori(regions_x_pwm_mat_file, pwm_x_pwm_corr_file, pwm_name_to_hgnc, pwm_dict):
    """Generate association rules
    """
    pwm_hits_df = pd.read_table(regions_x_pwm_mat_file, index_col=0)
    pwm_hits_df = pwm_hits_df.drop(["community"], axis=1)
    pwm_hits_df = (pwm_hits_df > 0).astype(int)

    pwm_hits_df.columns = [";".join([name.split("_")[0] for name in pwm_name_to_hgnc[pwm_name].split(";")])
                           for pwm_name in pwm_hits_df.columns]
    
    import ipdb
    ipdb.set_trace()

    # TO THINK ABOUT - how to reduce time here?
    # reduce motif space (motif reduction...)
    # maybe help it along by selecting top level motifs and then
    # subsetting to run from there?
    if False:
        signal = np.sum(pwm_hits_df, axis=0)
        signal = 300 * signal / signal.max()

        # pwm name to signal
        id_to_signal_dict = dict(zip(signal.index, signal.as_matrix()))
        
        corr_mat_df = pd.read_table(pwm_x_pwm_corr_file, index_col=0)
        corr_mat_df = reduce_corr_mat_by_motif_similarity(
            corr_mat_df,
            pwm_dict,
            id_to_signal_dict,
            edge_cor_thresh=0.25)

        pwm_hits_df = pwm_hits_df[corr_mat_df.columns]

    pwm_hits_df = pwm_hits_df.sample(10000)
        
    
    print "running apriori"
    itemsets = apriori(pwm_hits_df, min_support=0.5, use_colnames=True)
    rules = association_rules(itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values("lift", ascending=False)

    rules.to_csv("association_rules.testing.txt", sep="\t", header=True, index=False)


    
    return None


def enumerate_trajectory_communities(community_files, indices, sig_threshold=0.005):
    """given communities for each timepoint, merge into one file and enumerate along start to finish
    """
    data = pd.DataFrame()
    for i in xrange(len(community_files)):
        community_file = community_files[i]
        index = indices[i]

        data_tmp = pd.read_table(community_file, sep="\t", index_col=0)
        if data.shape[0] == 0:
            data["id"] = data_tmp.index
            data.index = data_tmp.index
            data["task-{}".format(index)] = data_tmp["community"]
        else:
            data_tmp["id"] = data_tmp.index
            data_tmp = data_tmp[["id", "community"]]
            data_tmp.columns = ["id", "task-{}".format(index)]
            data = data.merge(data_tmp, how="inner", on="id")

    data.index = data["id"]
    data = data.drop(["id"], axis=1)
    
    # enumerate
    data["enumerated"] = ["" for i in xrange(data.shape[0])]

    for i in xrange(data.shape[1]):
        print i
        data["enumerated"] = data["enumerated"] + data.iloc[:, data.shape[1]-i-2].astype(str).str.zfill(2)
        #data["enumerated"] = data["enumerated"] + (100**i) * (data.iloc[:, data.shape[1]-i-1])
    #data["enumerated"] = data["enumerated"].astype(int)

    # figure out which ones are significant and only keep those
    community_patterns = pd.DataFrame()
    from collections import Counter
    counts = Counter(data["enumerated"].tolist())
    enumerated_clusters = list(set(data["enumerated"].tolist()))

    for enumerated_cluster in enumerated_clusters:
        count = counts[enumerated_cluster]
        if float(count) / data.shape[0] >= sig_threshold:
            # keep
            pattern = data[data["enumerated"] == enumerated_cluster].iloc[0,:]
            community_patterns = community_patterns.append(
                pattern, ignore_index=True)[pattern.index.tolist()]

    #community_patterns = community_patterns.drop("enumerated", axis=1)
    community_patterns.to_csv("testing.communities.timeseries.txt", sep="\t")


    # TODO - here, write a kmeans method to better cluster from the enumeration derived means
    # TODO - check number of communities that come out.
    
    # from here, for each set of patterns, want to go through files and extract profiles
    # across time.
    for pattern_idx in xrange(community_patterns.shape[0]):
        print pattern_idx
        timeseries_motif_scores = pd.DataFrame()
        timeseries_motif_file = "testing.communities.timeseries.pattern_{}.txt".format(pattern_idx)
        timeseries_region_ids_file = "testing.communities.timeseries.pattern_{}.region_ids.txt".format(pattern_idx)
        timeseries_bed_file = "testing.communities.timeseries.pattern_{}.region_ids.bed".format(pattern_idx)
        
        # get the related regions
        regions = data[data["enumerated"] == community_patterns["enumerated"].iloc[pattern_idx]]
        regions.to_csv(timeseries_region_ids_file, columns=[], header=False)
        
        # make a bed
        to_bed = (
            "cat {0} | "
            "awk -F ';' '{{ print $3 }}' | "
            "awk -F '=' '{{ print $2 }}' | "
            "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
            "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
            "sort -k1,1 -k2,2n | "
            "bedtools merge -i stdin > "
            "{1}").format(
                timeseries_region_ids_file,
                timeseries_bed_file)
        print to_bed
        os.system(to_bed)

        # TODO - change this, should use the exact same set of regions as above
        for i in xrange(len(community_files)):
            community_file = community_files[i]
            index = indices[i]

            # extract the community
            data_tmp = pd.read_table(community_file, sep="\t", index_col=0)

            data_tmp = data_tmp.loc[regions.index,:]
            #data_tmp = data_tmp[data_tmp["community"] == community_patterns.iloc[pattern_idx, i]]

            # get the mean across columns
            data_mean = data_tmp.mean(axis=0)

            # append
            timeseries_motif_scores = timeseries_motif_scores.append(data_mean, ignore_index=True)

            
        timeseries_motif_scores = timeseries_motif_scores.fillna(0)
        timeseries_motif_scores.to_csv(timeseries_motif_file, sep="\t")


    # TODO - can take these patterns, overlap with other task information to get
    # the patterns across time.
            
    import ipdb
    ipdb.set_trace()

            
    return None



def run(args):
    """Find grammars utilizing the timeseries tasks
    """
    os.system('mkdir -p {}'.format(args.tmp_dir))
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logging.info("Found {} chrom files".format(len(data_files)))
            
    # set up pwms to use here, also pwm names
    pwm_list_file = "global.pwm_names.txt"
    
    # motif annotation
    # TODO add an arg here to just use all if no list
    pwm_name_to_hgnc, hgnc_to_pwm_name = setup_pwm_metadata(args.pwm_metadata_file)
    pwm_list, pwm_list_filt, pwm_list_filt_indices, pwm_names_filt = setup_pwms(args.pwm_file, pwm_list_file)
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    pwm_names_clean = [pwm_name.split("_")[0] for pwm_name in pwm_names_filt]
        
    print args.model["name"]
    print net_fns[args.model["name"]]


    # set up file loader, dependent on importance fn
    if args.backprop == "integrated_gradients":
        data_loader_fn = load_step_scaled_data_from_filename_list
    else:
        data_loader_fn = load_data_from_filename_list
    
    # set up graph
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        args.tasks,
        data_loader_fn,
        args.batch_size,
        net_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        inference_fn=net_fns[args.inference_fn],
        importances_tasks=args.importances_tasks,
        shuffle_data=True,
        filter_tasks=args.interpretation_tasks) # this keeps only from dynamic regions

    # checkpoint file (unless empty net)
    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    elif args.model["name"] == "empty_net":
        checkpoint_path = None
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    logging.info("Checkpoint: {}".format(checkpoint_path))

    # get pwm scores
    pwm_scores_h5 = '{0}/{1}.pwm-scores.h5'.format(
        args.tmp_dir, args.prefix)
    if not os.path.isfile(pwm_scores_h5):
        interpret(
            tronn_graph,
            checkpoint_path,
            args.batch_size,
            pwm_scores_h5,
            args.sample_size,
            {"pwms": pwm_list,
             "importances_fn": args.backprop},
            keep_negatives=False,
            validate_grammars=False,
            filter_by_prediction=True)
            #method=args.backprop if args.backprop is not None else "input_x_grad")

    import ipdb
    ipdb.set_trace()
        
    # now for each timepoint task, go through and calculate communities
    for i in xrange(len(args.importances_tasks)):

        interpretation_task_idx = args.importances_tasks[i]
            
        # extract global motif mat (region, motif) and save out to text file (to handle in R or python)
        region_x_pwm_mat_file = "{0}/{1}.task-{2}.region_x_pwm.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        if not os.path.isfile(region_x_pwm_mat_file):
            h5_dataset_to_text_file(
                pwm_scores_h5,
                "pwm-scores.taskidx-{}".format(i), # use i because the ordering in the file is just 0-10
                region_x_pwm_mat_file,
                pwm_list_filt_indices,
                pwm_names_filt)

        # get a sorted (ie clustered) version of the motif mat using phenograph (Louvain)
        region_x_pwm_sorted_mat_file = "{0}.phenograph_sorted.txt".format(
            region_x_pwm_mat_file.split(".txt")[0])
        if not os.path.isfile(region_x_pwm_sorted_mat_file):
            phenograph_cluster(region_x_pwm_mat_file, region_x_pwm_sorted_mat_file)
            
        # get the correlation matrix
        pwm_x_pwm_corr_file = "{0}/{1}.task-{2}.pwm_x_pwm.corr.mat.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        if not os.path.isfile(pwm_x_pwm_corr_file):
            get_correlation_file(
                region_x_pwm_sorted_mat_file,
                pwm_x_pwm_corr_file,
                corr_method="continuous_jaccard")

        # refine the region modules with another phenograph run
        if False:
            generate_motif_modules_w_phenograph(
                region_x_pwm_sorted_mat_file,
                pwm_x_pwm_corr_file,
                pwm_name_to_hgnc)


    # and then enumerate
    community_files = [
        #"grammars/{}.task-{}.region_x_pwm.phenograph_sorted.recalc.phenograph.adjusted.txt".format(
        "grammars/{}.task-{}.region_x_pwm.phenograph_sorted.txt".format(
            args.prefix, i)
        for i in args.importances_tasks]

    enumerate_trajectory_communities(community_files, args.importances_tasks)
    quit()






    # OLD

    
    
    # go through each interpretation task to get grammars
    #for i in xrange(len(args.interpretation_tasks)):
    #for i in xrange(1): # FOR TESTING
    #args.importances_tasks = [0, 1, 2, 3]
    for i in xrange(len(args.importances_tasks)):
    
        #interpretation_task_idx = args.interpretation_tasks[i]
        interpretation_task_idx = args.importances_tasks[i]
        filter_tasks = [interpretation_task_idx] + args.interpretation_tasks # this way, only get the dynamic peaks in that timepoint
        print "interpreting task", interpretation_task_idx

        #interpretation_task_idx = 12
        
        # set up pwms to use here, also pwm names
        pwm_list_file = "global.pwm_names.txt"
        
        # motif annotation
        pwm_name_to_hgnc, hgnc_to_pwm_name = setup_pwm_metadata(args.pwm_metadata_file)
        # TODO add an arg here to just use all if no list
        pwm_list, pwm_list_filt, pwm_list_filt_indices, pwm_names_filt = setup_pwms(args.pwm_file, pwm_list_file)
        pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
        
        print args.model["name"]
        print net_fns[args.model["name"]]

        # set up graph
        tronn_graph = TronnNeuralNetGraph(
            {'data': data_files},
            args.tasks,
            load_data_from_filename_list,
            args.batch_size,
            net_fns[args.model['name']],
            args.model,
            tf.nn.sigmoid,
            inference_fn=net_fns[args.inference_fn],
            importances_tasks=args.importances_tasks,
            shuffle_data=True,
            filter_tasks=filter_tasks)
            #filter_tasks=args.interpretation_tasks) # FOR TESTING
            #filter_tasks=[interpretation_task_idx]) 

        # checkpoint file (unless empty net)
        if args.model_checkpoint is not None:
            checkpoint_path = args.model_checkpoint
        elif args.model["name"] == "empty_net":
            checkpoint_path = None
        else:
            checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
        logging.info("Checkpoint: {}".format(checkpoint_path))

        # get pwm scores
        pwm_hits_mat_h5 = '{0}/{1}.task-{2}.pwm-hits.h5'.format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        if not os.path.isfile(pwm_hits_mat_h5):
            interpret(
                tronn_graph,
                checkpoint_path,
                args.batch_size,
                pwm_hits_mat_h5,
                args.sample_size,
                {"pwms": pwm_list},
                #pwm_list_filt,
                keep_negatives=False,
                filter_by_prediction=True,
                method=args.backprop if args.backprop is not None else "input_x_grad")

        # set up task idx for the global scores
        if args.model["name"] == "empty_net":
            global_taskidx = 1
        else:
            global_taskidx = 10

        pwm_names_clean = [pwm_name.split("_")[0] for pwm_name in pwm_names_filt]
            
        # extract global motif mat (region, motif) and save out to text file (to handle in R or python)
        region_x_pwm_mat_file = "{0}/{1}.task-{2}.region_x_pwm.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        if not os.path.isfile(region_x_pwm_mat_file):
            h5_dataset_to_text_file(
                pwm_hits_mat_h5,
                "pwm-scores.taskidx-{}".format(i), # TODO check this
                #"pwm-scores.taskidx-{}".format(global_taskidx),
                region_x_pwm_mat_file,
                pwm_list_filt_indices,
                pwm_names_filt)

        # get a sorted (ie clustered) version of the motif mat using phenograph (Louvain)
        region_x_pwm_sorted_mat_file = "{0}.phenograph_sorted.txt".format(
            region_x_pwm_mat_file.split(".txt")[0])
        if not os.path.isfile(region_x_pwm_sorted_mat_file):
            phenograph_cluster(region_x_pwm_mat_file, region_x_pwm_sorted_mat_file)
            
        # get the correlation matrix
        pwm_x_pwm_corr_file = "{0}/{1}.task-{2}.pwm_x_pwm.corr.mat.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        if not os.path.isfile(pwm_x_pwm_corr_file):
            get_correlation_file(
                region_x_pwm_sorted_mat_file,
                pwm_x_pwm_corr_file,
                corr_method="continuous_jaccard")

        # generate motif modules
        if False:
            generate_motif_modules_w_phenograph(
                region_x_pwm_sorted_mat_file,
                pwm_x_pwm_corr_file,
                pwm_name_to_hgnc)

        # calculate the grammars here
        #run_apriori(region_x_pwm_sorted_mat_file, pwm_x_pwm_corr_file, pwm_name_to_hgnc, pwm_dict)
        
            
    # at this stage, gather the phenograph files and collect communities, and enumerate
    community_files = ["grammars/{}.task-{}.region_x_pwm.phenograph_sorted.recalc.phenograph.txt".format(args.prefix, i) for i in args.importances_tasks]
    enumerate_trajectory_communities(community_files, args.importances_tasks)

    
    quit()
    args.interpretation_tasks = [16]
    
    # get the max correlations/signal strengths to set up global graph here
    # adjust names here, this is for plotting
    correlation_files = [
        "{0}/{1}.task-{2}.pwm_x_pwm.corr.mat.txt".format(
            args.tmp_dir, args.prefix, args.interpretation_tasks[i])
        for i in xrange(len(args.interpretation_tasks))]
    max_corr_df = get_max_corr_mat(correlation_files, pwm_names_filt)
    max_corr_df.columns = [";".join([name.split("_")[0] for name in pwm_name_to_hgnc[pwm_name].split(";")])
                           for pwm_name in max_corr_df.columns]
    max_corr_df.index = max_corr_df.columns
    
    # then with that, get back a G (graph) and the positioning according to this graph
    max_G = get_networkx_graph(max_corr_df, corr_thresh=0.25)
    max_G_positions = nx.spring_layout(max_G, weight="value") # k is normally 1/sqrt(n), n=node_num k=0.15

    # then replot the network using this graph
    for i in xrange(len(args.interpretation_tasks)):

        interpretation_task_idx = args.interpretation_tasks[i]
        print interpretation_task_idx
        prefix = "{0}/{1}.task-{2}.graph".format(
            args.out_dir, args.prefix, interpretation_task_idx)

        # files
        mat_file = "{0}/{1}.task-{2}.region_x_pwm.phenograph_sorted.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        corr_file = "{0}/{1}.task-{2}.pwm_x_pwm.corr.mat.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        
        # plot
        task_G = plot_motif_network(
            mat_file,
            corr_file,
            pwm_name_to_hgnc,
            prefix,
            pwm_dict,
            max_G_positions,
            reduce_pwms=False,
            edge_cor_thresh=0.25) # 0.25 is good

        cliques = list(nx.find_cliques(task_G))

        # for each clique, remove if below certain size and reduce motif redundancy
        

        import ipdb
        ipdb.set_trace()

        # and save out components
        grammar_file = "{0}/{1}.task-{2}.grammars.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        separate_and_save_components(
            task_G,
            mat_file,
            prefix,
            hgnc_to_pwm_name,
            grammar_file)

    return None
