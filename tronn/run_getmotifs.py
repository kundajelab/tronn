# description: test function for a multitask interpretation pipeline

import os
import h5py
import glob
import logging

import numpy as np
import tensorflow as tf

from collections import Counter

from tronn.graphs import TronnGraph
from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.nets.nets import model_fns
from tronn.interpretation.importances import extract_importances_and_motif_hits
from tronn.interpretation.importances import extract_motif_assignments
from tronn.interpretation.importances import get_pwm_hits_from_raw_sequence
from tronn.interpretation.importances import layerwise_relevance_propagation
from tronn.interpretation.importances import visualize_sample_sequences

from tronn.interpretation.importances import split_importances_by_task_positives
from tronn.interpretation.seqlets import extract_seqlets
from tronn.interpretation.seqlets import reduce_seqlets
from tronn.interpretation.seqlets import cluster_seqlets
from tronn.interpretation.seqlets import make_motif_sets

from tronn.datalayer import get_total_num_examples

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import get_encode_pwms
from tronn.interpretation.motifs import bootstrap_fdr_v2
from tronn.interpretation.motifs import make_motif_x_timepoint_mat


def setup_pwms(pwm_file, task_idx):
    """Janky helper function - to be cleaned up!
    """
    # with inference this produces the importances and motif hits.
    pwm_list = get_encode_pwms(pwm_file)

    pwms_per_task = { # remember 0-indexed
        0: [
            "ETS",      
            "FOSL",
            "NFKB1",
            "RUNX",
            "SOX3"],
        1: [
            "FOSL",
            "NFKB1",
            "RUNX",
            "SOX3"],
        2: [
            "FOSL",
            "NFKB1",
            "RUNX",
            "TEAD",
            "SOX3"],
        3: [
            "NFKB1",
            "FOSL",
            "TEAD",
            "RUNX"],
        4: [
            "FOSL1",
            "NFY",
            "TEAD",
            "RUNX",
            "TP63"],
        5: [
            "CEBPA",
            "KLF4",
            "NFY",
            "TEAD",
            "TP63",        
            "ZNF750"],
        6: [
            "CEBPA",
            "GRHL",
            "KLF4",
            "TP63",        
            "ZNF750"],
        7: [
            "CEBPA",
            "GRHL",
            "KLF4",
            "ZNF750"],
        8: [
            "CEBPA",
            "GRHL",
            "KLF4",
            "ZNF750"],
        9: [
            "CEBPA",
            "GRHL",
            "KLF4",
            "ZNF750"],
        10: [
            "FOSL1",
            "TP63"],
        11: [ # extra task for stable openness, dynamic H3K27ac
            "CEBPA",
            "GRHL",
            "KLF4",
            "ZNF750",
            "ETS"]
    }

    pwm_list_filt = []
    for pwm in pwm_list:
        for pwm_name in pwms_per_task[task_idx]:
            if pwm_name in pwm.name:
                pwm_list_filt.append(pwm)

    print "Using PWMS:", [pwm.name for pwm in pwm_list_filt]

    return pwm_list_filt


def run(args):
    """Find motifs (global and interpretation task specific)
    """
    os.system('mkdir -p {}'.format(args.tmp_dir))
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logging.info("Found {} chrom files".format(len(data_files)))

    # pwms
    pwm_list = get_encode_pwms(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]

    # get global as well as interpretation task motif matrices
    
    # set up graph
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size / 2,
        model_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        importances_fn=layerwise_relevance_propagation,
        importances_tasks=args.importances_tasks,
        shuffle_data=True, # NOTE: CHANGE LATER
        filter_tasks=[])

    # checkpoint file
    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    logging.info("Checkpoint: {}".format(checkpoint_path))

    # get motif hits
    pwm_counts_mat_h5 = '{0}/{1}.pwm-counts.h5'.format(
        args.tmp_dir, args.prefix)
    if not os.path.isfile(pwm_counts_mat_h5):
        extract_motif_assignments(
            tronn_graph,
            checkpoint_path,
            pwm_counts_mat_h5,
            args.sample_size,
            pwm_list,
            method="guided_backprop") # simple_gradients

    # now run a bootstrap FDR for the various tasks

    # first global
    if True:
        i = 10
        bootstrap_fdr_v2(
            pwm_counts_mat_h5, 
            pwm_names, 
            "global", 
            10, # not used
            "pwm-counts.taskidx-{}".format(i), 
            global_importances=True)
        
    if False:
        print "bootstrap fdr test"
        i = 7
        interpretation_task_idx = args.interpretation_tasks[i]
        print interpretation_task_idx
        bootstrap_fdr_v2(
            pwm_counts_mat_h5, 
            pwm_names, 
            "testing.task-{}".format(interpretation_task_idx), 
            interpretation_task_idx, 
            "pwm-counts.taskidx-{}".format(10))
    
    # and now can take this new master list and then go through timepoints and extract counts per timepoint
    master_pwm_names = []
    with open("global.bootstrap_fdr.cutoff.txt", "r") as fp:
        for line in fp:
            master_pwm_names.append(line.strip().split('\t')[0])
            
    master_pwm_indices = [i for i in xrange(len(pwm_names)) if pwm_names[i] in master_pwm_names]
    master_pwm_names_sorted = [pwm_name.split("_")[0] for pwm_name in pwm_names if pwm_name in master_pwm_names]
    key_list = ["pwm-counts.taskidx-{}".format(i) for i in xrange(10)]
    if True:
        make_motif_x_timepoint_mat(
            pwm_counts_mat_h5, 
            key_list, 
            args.importances_tasks, 
            master_pwm_indices, 
            master_pwm_names_sorted)

    # and now do this for trajectories too
    key_list = ["pwm-counts.taskidx-{}".format(10) for i in xrange(10)]
    if False:
        make_motif_x_timepoint_mat(
            pwm_counts_mat_h5, 
            key_list, 
            args.interpretation_tasks, 
            master_pwm_indices, 
            master_pwm_names_sorted, 
            prefix="trajectories.")

    return
