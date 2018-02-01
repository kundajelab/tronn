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

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import read_pwm_file

from tronn.interpretation.grammars import get_significant_correlations
from tronn.interpretation.grammars import reduce_corr_mat_by_motif_similarity
from tronn.interpretation.grammars import get_networkx_graph
from tronn.interpretation.grammars import plot_corr_as_network
from tronn.interpretation.grammars import plot_corr_on_fixed_graph
from tronn.interpretation.grammars import get_significant_motifs

from tronn.interpreation.networks import separate_and_save_components

import networkx as nx

import phenograph


def setup_pwms(master_pwm_file, pwm_subset_list_file):
    """setup which pwms are being used
    """
    # open the pwm subset file to get names of the pwms to use
    pwms_to_use = []
    with open(pwm_subset_list_file, "r") as fp:
        for line in fp:
            pwms_to_use.append(line.strip().split('\t')[0])        
            
    # then open the master file and filter out unused ones
    pwm_list = read_pwm_file(master_pwm_file)
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

    return pwm_list_filt, pwm_list_filt_indices, pwm_names_filt


def setup_pwm_metadata(metadata_file):
    """read in metadata to dicts for easy use
    """
    pwm_name_to_hgnc = {}
    hgnc_to_pwm_name = {}
    with open(metadata_file, "r") as fp:
        for line in fp:
            fields = line.strip().split("\t")
            try:
                pwm_name_to_hgnc[fields[0]] = fields[4]
                hgnc_to_pwm_name[fields[4]] = fields[0]
            except:
                pwm_name_to_hgnc[fields[0]] = fields[0].split(".")[0].split("_")[2]
                pwm_name_to_hgnc[fields[0]] = "UNK"

    return pwm_name_to_hgnc, hgnc_to_pwm_name


def quick_filter(pwm_hits_df):
    """basic filters
    """
    # remove rows that are zero
    pwm_hits_df_binary = (pwm_hits_df > 0).astype(int) #* pwm_hits_df
    pwm_hits_df = pwm_hits_df.loc[~(pwm_hits_df_binary==0).all(axis=1)] # remove elements with no hits
    pwm_hits_df_binary = pwm_hits_df_binary.loc[~(pwm_hits_df_binary==0).all(axis=1)] # remove elements with no hits
    
    # remove low scoring motifs
    pwm_hits_df = pwm_hits_df.loc[:, np.sum(pwm_hits_df_binary, axis=0) > 100]
    pwm_hits_df_binary = pwm_hits_df_binary.loc[:, np.sum(pwm_hits_df_binary, axis=0) > 100]
            
    return pwm_hits_df, pwm_hits_df_binary


def phenograph_cluster(pwm_hits_df, sorted_mat_file):
    """Use to quickly get a nicely clustered (sorted) output file to visualize examples
    """
    # run Louvain here for basic clustering to visually look at patterns
    pwm_hits_df, pwm_hits_df_binary = quick_filter(pwm_hits_df)
    communities, graph, Q = phenograph.cluster(pwm_hits_df_binary)

    # save out the sorted info into a new mat sorted by community
    sorted_mat_file = "{0}/{1}.task-{2}.motif_mat.reduced.community_sorted.txt".format(
        args.tmp_dir, args.prefix, interpretation_task_idx)
    pwm_hits_df["community"] = communities
    pwm_hits_df = pwm_hits_df.sort_values("community")
    pwm_hits_df.to_csv(sorted_mat_file, sep='\t')
    pwm_hits_df = pwm_hits_df.drop(["community"], axis=1)

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
    max_signals = np.zeros((num_pwms))

    for correlation_file in correlation_files:
        
        # load in corr mat file
        corr_mat_df = pd.read_table(correlation_file, index_col=0)
        #corr_mat_df.columns = [";".join([name.split("_")[0] for name in pwm_name_to_hgnc[pwm_name].split(";")])
        #                       for pwm_name in corr_mat_df.columns]
        #corr_mat_df.index = corr_mat_df.columns

        # apply max
        max_array = np.maximum(max_corr_df.as_matrix(), corr_mat_df.as_matrix())
        #max_corr_df = pd.DataFrame(max_array, index=corr_mat_df.index, columns=corr_mat_df.columns)

    return max_corr_df



def plot_motif_network(regions_x_pwm_mat_file, pwm_x_pwm_corr_file, id_to_name, reduce_pwms=True):
    """Plot full motif network with signal strengths
    """
    # Extract signal strengths
    pwm_hits_df = pd.read_table(regions_x_pwm_mat_file, index_col=0)
    pwm_hits_df = pwm_hits_df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
    pwm_hits_df = pwm_hits_df.loc[~(pwm_hits_df==0).all(axis=1)] # remove elements with no hits

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
        reduced_corr_mat_file = "{}.reduced.txt".format(corr_mat_file.split(".tmp")[0])
        pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
        corr_mat_df = reduce_corr_mat_by_motif_similarity(
            corr_mat_df,
            pwm_dict,
            id_to_signal_dict,
            edge_cor_thresh=0.25)

    # and adjust names after, need to match pwms
    corr_mat_df.columns = [";".join([name.split("_")[0] for name in id_to_name[pwm_name].split(";")])
                           for pwm_name in corr_mat_df.columns]
    corr_mat_df.index = corr_mat_df.columns

    # save this out to quickly plot later
    corr_mat_df.to_csv(reduced_corr_mat_file, sep="\t")

    # plotting
    prefix = "task-{}.testing.max_pos".format(interpretation_task_idx)
    task_G = plot_corr_on_fixed_graph(corr_mat_df, max_G_positions, prefix, corr_thresh=0.25, node_size_dict=node_size_dict)

    return task_G


def run(args):
    """Find grammars utilizing the timeseries tasks
    """
    os.system('mkdir -p {}'.format(args.tmp_dir))
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logging.info("Found {} chrom files".format(len(data_files)))
    
    # go through each interpretation task to get grammars
    for i in xrange(len(args.interpretation_tasks)):

        interpretation_task_idx = args.interpretation_tasks[i]
        print "interpreting task", interpretation_task_idx
        
        # set up pwms to use here, also pwm names
        pwm_list_file = "global.pwm_names.txt"
        
        # motif annotation
        #metadata_file = "/srv/scratch/shared/indra/dskim89/ggr/integrative/v0.2.5/annotations/HOCOMOCOv11_core_annotation_HUMAN_mono.nonredundant.expressed.txt"
        pwm_name_to_hgnc, hgnc_to_pwm_name = setup_pwm_metadata(args.metadata_file)
        # TODO add an arg here to just use all if no list
        pwm_list_filt, pwm_list_filt_indices, pwm_names_filt = setup_pwms(args.pwm_file, pwm_list_file)

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
            importances_fn=net_fns[args.importances_fn],
            importances_tasks=args.importances_tasks,
            shuffle_data=True,
            filter_tasks=[interpretation_task_idx])

        # checkpoint file (unless empty net)
        if args.model_checkpoint is not None:
            checkpoint_path = args.model_checkpoint
        elif args.model["name"] == "empty_net":
            checkpoint_path = None
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

        # set up task idx for the global scores
        if args.model["name"] == "empty_net":
            global_taskidx = 1
        else:
            global_taskidx = 10
            
        # put those into a text file for easy downstream handling
        reduced_mat_file = "{0}/{1}.task-{2}.motif_mat.reduced.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        pwm_names_clean = [pwm_name.split("_")[0] for pwm_name in pwm_names_filt]
        if not os.path.isfile(reduced_mat_file):
            with h5py.File(pwm_hits_mat_h5, "r") as hf:
                # only keep those in filtered set
                pwm_hits = hf["pwm-scores.taskidx-{}".format(global_taskidx)][:][:,np.array(pwm_list_filt_indices)]
                
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
            corr_mat, pval_mat = get_significant_correlations(
                pwm_hits_df.as_matrix(),
                corr_method="continuous_jaccard",
                #corr_method="pearson",
                corr_min=0.4,
                pval_thresh=0.05)
            
            corr_mat_df = pd.DataFrame(corr_mat, index=pwm_names_filt, columns=pwm_names_filt)
            corr_mat_df.to_csv(corr_mat_file, sep="\t")

            #corr_mat_df.columns = [pwm_name_to_hgnc[pwm_name] for pwm_name in corr_mat_df.columns]
            corr_mat_df.columns = [";".join([name.split("_")[0] for name in pwm_name_to_hgnc[pwm_name].split(";")])
                                   for pwm_name in corr_mat_df.columns]
            corr_mat_df.index = corr_mat_df.columns

        # phenograph to have nice output to look at
        sorted_mat_file = "{0}/{1}.task-{2}.motif_mat.reduced.community_sorted.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        phenograph_cluster(pwm_hits_df, sorted_mat_file)
        
    # get the max correlations/signal strengths to set up global graph here
    correlation_files = [
        "{}/{}.task-{}.corr_mat.tmp".format(args.tmp_dir, args.prefix, args.interpretation_tasks[i])
        for i in xrange(len(args.interpretation_tasks))]
    max_corr_df = get_max_corr_mat(correlation_files, pwm_names_filt)
        
    # then with that, get back a G (graph)
    max_G = get_networkx_graph(max_corr_df, corr_thresh=0.2)
    max_G_positions = nx.spring_layout(max_G, weight="value", k=0.15) # k is normally 1/sqrt(n), n=node_num

    # then replot the network using this graph
    for i in xrange(len(args.interpretation_tasks)):

        interpretation_task_idx = args.interpretation_tasks[i]
        print interpretation_task_idx

        # files
        reduced_mat_file = "{0}/{1}.task-{2}.motif_mat.reduced.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)
        corr_mat_file = "{0}/{1}.task-{2}.corr_mat.tmp".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)

        # plot
        task_G = plot_motif_network(reduced_mat_file, corr_mat_file, pwm_name_to_hgnc, reduce_pwms=True)

        # and save out components
        separate_and_save_components(task_G, pwm_hits_df, prefix, hgnc_to_pwm_name)

        
    return None
