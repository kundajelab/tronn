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
from tronn.nets.nets import net_fns

from tronn.interpretation.interpret import interpret

from tronn.interpretation.importances import split_importances_by_task_positives
from tronn.interpretation.seqlets import extract_seqlets
from tronn.interpretation.seqlets import reduce_seqlets
from tronn.interpretation.seqlets import cluster_seqlets
from tronn.interpretation.seqlets import make_motif_sets

from tronn.datalayer import get_total_num_examples

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import get_encode_pwms

import phenograph

import networkx as nx
from networkx.drawing.nx_pydot import pydot_layout

from scipy.stats import pearsonr


def get_significant_correlations(motif_mat, pval_thresh=0.05, corr_min=0.4):
    """Given a matrix, calculate pearson correlation for each pair of 
    columns and look at p-val. If p-val is above threshold, keep 
    otherwise leave as 0.
    """
    num_columns = motif_mat.shape[1]
    correlation_vals = np.zeros((num_columns, num_columns))
    correlation_pvals = np.zeros((num_columns, num_columns))
    for i in xrange(num_columns):
        for j in xrange(num_columns):
            cor_val, pval = pearsonr(motif_mat[:,i], motif_mat[:,j])

            if pval < pval_thresh and cor_val > corr_min:
                correlation_vals[i,j] = cor_val
                correlation_pvals[i,j] = pval

    return correlation_vals, correlation_pvals


def get_networkx_graph(corr_mat, corr_thresh=0.0):
    """preprocessing to get a clean graph object
    """
    links = corr_mat.stack().reset_index()
    links.columns = ["var1", "var2", "value"]

    # remove self correlation and zeros
    links_filtered = links.loc[ (links["value"] > corr_thresh) & (links["var1"] !=  links["var2"]) ]
    links_filtered = links_filtered.loc[links["var1"] != "UNK"]
    links_filtered = links_filtered.loc[links["var2"] != "UNK"]

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
    
    return None


def plot_corr_on_fixed_graph(corr_mat, positions, prefix, node_size_dict=None):
    """Probably need some assertions to make sure that the
    same nodes are represented?
    """
    G = get_networkx_graph(corr_mat, corr_thresh=0.5)

    # set up node sizes
    if node_size_dict is None:
        node_size = 50
    else:
        node_size = [node_size_dict[node] for node in G.nodes]
    print node_size

    # set up edge weights
    edge_to_weight = nx.get_edge_attributes(G, "value")
    edge_weights = [edge_to_weight[edge] for edge in G.edges]


    
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
    
    return




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


def run(args):
    """Find grammars utilizing the timeseries tasks
    """
    os.system('mkdir -p {}'.format(args.tmp_dir))
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logging.info("Found {} chrom files".format(len(data_files)))

    # debug
    #args.interpretation_tasks = [16, 17, 18, 19, 20, 21, 22, 23]
    #args.interpretation_tasks = [16, 18, 23]
    #args.interpretation_tasks = [32]
    
    # go through each interpretation task
    for i in xrange(len(args.interpretation_tasks)):

        interpretation_task_idx = args.interpretation_tasks[i]
        print interpretation_task_idx
        
        # set up pwms to use here, also pwm names
        #pwm_list_file = "{}/task-{}.permutation_test.cutoff.txt".format(
        #    args.motif_dir, interpretation_task_idx) # for now, use --pwm_list for the directory with pwm lists
            #args.motif_dir, 16) # for now, use --pwm_list for the directory with pwm lists
        #pwm_list_file = "{}/global.permutation_test.cutoff.txt".format(args.motif_dir)
        pwm_list_file = "global.pwm_names.txt"
        
        # motif annotation
        metadata_file = "/srv/scratch/shared/indra/dskim89/ggr/integrative/v0.2.5/annotations/HOCOMOCOv11_core_annotation_HUMAN_mono.nonredundant.expressed.txt"
        pwm_name_to_hgnc = {}
        with open(metadata_file, "r") as fp:
            for line in fp:
                fields = line.strip().split("\t")
                try:
                    pwm_name_to_hgnc[fields[0]] = fields[4]
                except:
                    pwm_name_to_hgnc[fields[0]] = fields[0].split(".")[0].split("_")[2]
                    pwm_name_to_hgnc[fields[0]] = "UNK"
        
        pwms_to_use = []
        with open(pwm_list_file, "r") as fp:
            for line in fp:
                pwms_to_use.append(line.strip().split('\t')[0])        

        pwm_list = get_encode_pwms(args.pwm_file)
        pwm_list_filt = []
        pwm_list_filt_indices = []
        for i in xrange(len(pwm_list)):
            pwm = pwm_list[i]
            for pwm_name in pwms_to_use:
                if pwm_name in pwm.name:
                    pwm_list_filt.append(pwm)
                    pwm_list_filt_indices.append(i)
        print "Using PWMS:", [pwm.name for pwm in pwm_list_filt]
        print len(pwm_list_filt)
        pwm_names_filt = [pwm.name for pwm in pwm_list_filt]

        # set up graph
        tronn_graph = TronnNeuralNetGraph(
            {'data': data_files},
            args.tasks,
            load_data_from_filename_list,
            args.batch_size,
            net_fns[args.model['name']],
            args.model,
            tf.nn.sigmoid,
            importances_fn=net_fns["importances_to_motif_assignments"],
            importances_tasks=args.importances_tasks,
            shuffle_data=True,
            filter_tasks=[interpretation_task_idx])

        # checkpoint file
        if args.model_checkpoint is not None:
            checkpoint_path = args.model_checkpoint
        else:
            checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
        logging.info("Checkpoint: {}".format(checkpoint_path))

        # get motif hits
        pwm_hits_mat_h5 = '{0}/{1}.task-{2}.pwm-hits.h5'.format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        if not os.path.isfile(pwm_hits_mat_h5):
            interpret(
                tronn_graph,
                checkpoint_path,
                args.batch_size,
                pwm_hits_mat_h5,
                args.sample_size,
                pwm_list,
                #pwm_list_filt,
                keep_negatives=False,
                filter_by_prediction=True,
                method=args.backprop if args.backprop is not None else "input_x_grad")

        # put those into a text file to load into R
        reduced_mat_file = "{0}/{1}.task-{2}.motif_mat.reduced.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        pwm_names_clean = [pwm_name.split("_")[0] for pwm_name in pwm_names_filt]
        if not os.path.isfile(reduced_mat_file):
            with h5py.File(pwm_hits_mat_h5, "r") as hf:
                # only keep those in filtered set
                pwm_hits = hf["pwm-counts.taskidx-10"][:][:,np.array(pwm_list_filt_indices)]

                # further reduce out those that are low scoring?
                #index = hf["example_metadata"][:][~np.all(pwm_hits == 0, axis=1),:]
                #pwm_hits = pwm_hits[~np.all(pwm_hits == 0, axis=1),:]
                #pwm_hits = pwm_hits[:, pwm_indices]
                #pwm_hits = pwm_hits[:, ~np.all(pwm_hits == 0, axis=0)]
                
                # set up dataframe and save out
                pwm_hits_df = pd.DataFrame(pwm_hits, index=hf["example_metadata"][:][:,0], columns=pwm_names_filt)
                pwm_hits_df.to_csv(reduced_mat_file, sep='\t')


        # always reload in case you are using a smaller file
        pwm_hits_df = pd.read_table(reduced_mat_file, index_col=0)
        pwm_hits_df = pwm_hits_df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
        pwm_hits_df = pwm_hits_df.loc[~(pwm_hits_df==0).all(axis=1)] # remove elements with no hits
        print "total examples used:", pwm_hits_df.shape

        # for this task, get correlation matrices
        corr_mat_file = "{0}/{1}.task-{2}.corr_mat.tmp".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        if not os.path.isfile(corr_mat_file):
            corr_mat, pval_mat = get_significant_correlations(pwm_hits_df.as_matrix(), pval_thresh=0.05)
            corr_mat_df = pd.DataFrame(corr_mat, index=pwm_names_filt, columns=pwm_names_filt)
            corr_mat_df.to_csv(corr_mat_file, sep="\t")

            #corr_mat_df.columns = [pwm_name_to_hgnc[pwm_name] for pwm_name in corr_mat_df.columns]
            corr_mat_df.columns = [";".join([name.split("_")[0] for name in pwm_name_to_hgnc[pwm_name].split(";")])
                                   for pwm_name in corr_mat_df.columns]
            corr_mat_df.index = corr_mat_df.columns
        
            # generate a network plot
            plot_corr_as_network(corr_mat_df, "task-{}.testing.network".format(interpretation_task_idx))

        continue
        
        # also do Louvain clustering to visualize here
        
        

        # for now, just want the reduced mats
        # but maybe need to do the community clustering first?

        # TODO throw in Louvain communities here
        # this is unsupervised learning - clusters. within these, need to get significant motifs (relative to background)

        # filtering:
        if True:
            # remove rows that are zero
            pwm_hits_df_binary = (pwm_hits_df > 0).astype(int) #* pwm_hits_df

            pwm_hits_df = pwm_hits_df.loc[~(pwm_hits_df_binary==0).all(axis=1)] # remove elements with no hits
            pwm_hits_df_binary = pwm_hits_df_binary.loc[~(pwm_hits_df_binary==0).all(axis=1)] # remove elements with no hits
            
            # remove low scoring motifs
            pwm_hits_df = pwm_hits_df.loc[:, np.sum(pwm_hits_df_binary, axis=0) > 100]
            pwm_hits_df_binary = pwm_hits_df_binary.loc[:, np.sum(pwm_hits_df_binary, axis=0) > 100]
            
        communities, graph, Q = phenograph.cluster(pwm_hits_df_binary)

        # save out a new mat sorted by community
        sorted_mat_file = "{0}/{1}.task-{2}.motif_mat.reduced.community_sorted.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        pwm_hits_df["community"] = communities
        pwm_hits_df = pwm_hits_df.sort_values("community")
        pwm_hits_df.to_csv(sorted_mat_file, sep='\t')
        pwm_hits_df = pwm_hits_df.drop(["community"], axis=1)

        # continue here?

        # and then only keep reasonably sized communities
        for community_idx in np.unique(communities).tolist():

            # determine community size
            region_indices = np.where(communities == community_idx)[0]

            # only keep large communities
            if region_indices.shape[0] < 100:
                continue
            
            # pull out matrix
            community_mat = pwm_hits_df.iloc[region_indices,]


            # TODO: bootstrap FDR to figure out which motifs are truly enriched relative to others in the set
            # DO THIS
            get_significant_motifs(community_mat.as_matrix(), community_mat.columns, "testing", num_shuffles=99, pval_thresh=0.05)

            quit()
            
            # determine community name
            ranked_pwms = np.sum(community_mat, axis=0).sort_values()
            stop_idx = -1
            top_score = ranked_pwms[-1]
            thresh = 0.10
            while True:
                print stop_idx
                if -stop_idx == len(ranked_pwms):
                    break
                elif ranked_pwms[stop_idx-1] < top_score * thresh:
                    break
                else:
                    stop_idx -= 1
            
            top_pwms = ranked_pwms[stop_idx:].index.tolist()
            print top_pwms
            print ranked_pwms[stop_idx:]
            print region_indices.shape[0]
            
            # and save out community matrix and BED file
            out_prefix = "{0}/{1}.task-{2}.community-{3}.{4}".format(
                args.tmp_dir, args.prefix, interpretation_task_idx, community_idx, "_".join(list(reversed(top_pwms))[:3]))
            out_mat_file = "{}.mat.txt".format(out_prefix)
            community_mat.to_csv(out_mat_file, sep='\t')

            out_bed_file = "{}.bed".format(out_prefix)
            make_bed = (
                "cat {0} | "
                "awk -F '\t' '{{ print $1 }}' | "
                "grep -v index | "
                "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
                "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
                "sort -k1,1 -k2,2n "
                "> {1}").format(out_mat_file, out_bed_file)
            os.system(make_bed)

    # get the max correlations/signal strengths to set up global graph here
    max_corr_df = pd.DataFrame(
        data=np.zeros((len(pwm_names_filt), len(pwm_names_filt))),
        index=pwm_names_filt,
        columns=pwm_names_filt)
    max_signals = np.zeros((len(pwm_names_filt)))
    for i in xrange(len(args.interpretation_tasks)):

        interpretation_task_idx = args.interpretation_tasks[i]
        print interpretation_task_idx

        # load in corr mat file
        corr_mat_file = "{0}/{1}.task-{2}.corr_mat.tmp".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        corr_mat_df = pd.read_table(corr_mat_file, index_col=0)
        corr_mat_df.columns = [";".join([name.split("_")[0] for name in pwm_name_to_hgnc[pwm_name].split(";")])
                               for pwm_name in corr_mat_df.columns]
        corr_mat_df.index = corr_mat_df.columns

        # apply max
        max_array = np.maximum(max_corr_df.as_matrix(), corr_mat_df.as_matrix())
        max_corr_df = pd.DataFrame(max_array, index=corr_mat_df.index, columns=corr_mat_df.columns)

        
    # then with that, get back a G (graph)
    max_G = get_networkx_graph(max_corr_df)
    max_G_positions = nx.spring_layout(max_G, weight="value")

    import ipdb
    ipdb.set_trace()

    # then replot the network using this graph
    for i in xrange(len(args.interpretation_tasks)):

        interpretation_task_idx = args.interpretation_tasks[i]
        print interpretation_task_idx

        # load in corr mat file
        corr_mat_file = "{0}/{1}.task-{2}.corr_mat.tmp".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        corr_mat_df = pd.read_table(corr_mat_file, index_col=0)
        corr_mat_df.columns = [";".join([name.split("_")[0] for name in pwm_name_to_hgnc[pwm_name].split(";")])
                               for pwm_name in corr_mat_df.columns]
        corr_mat_df.index = corr_mat_df.columns

        # TODO also extract signal strength here, look at the raw values and
        # calculate colSums
        reduced_mat_file = "{0}/{1}.task-{2}.motif_mat.reduced.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        pwm_hits_df = pd.read_table(reduced_mat_file, index_col=0)
        pwm_hits_df = pwm_hits_df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
        pwm_hits_df = pwm_hits_df.loc[~(pwm_hits_df==0).all(axis=1)] # remove elements with no hits

        # TODO: figure out how to get signal to noise score here - sum gets diluted by lots of noise
        # would there be any rationale to squaring?
        signal = np.sum(pwm_hits_df, axis=0)
        signal = 300 * signal / signal.max()
        signal.index = corr_mat_df.index
        print signal.nlargest(n=10)
        node_size_dict = dict(zip(signal.index, signal.as_matrix()))
        
        prefix = "task-{}.testing.max_pos".format(interpretation_task_idx)
        plot_corr_on_fixed_graph(corr_mat_df, max_G_positions, prefix, node_size_dict=node_size_dict)

        # key point - if we can see it on the network, it needs to be clear enough to threshold out
        # that set and check to make sure it's real
        
    return
