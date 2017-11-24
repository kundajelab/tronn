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
from tronn.interpretation.importances import extract_importances_and_viz
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

from tronn.preprocess import generate_nn_dataset


def run(args):
    """Find motifs (global and interpretation task specific)
    """
    os.system('mkdir -p {}'.format(args.tmp_dir))

    if True:
        # first generate dataset
        generate_nn_dataset(
            args.labels[0],
            args.annotations["univ_dhs"],
            args.annotations["ref_fasta"],
            args.labels,
            args.out_dir,
            args.prefix,
            parallel=1,
            neg_region_num=0,
            use_dhs=False,
            use_random=False,
            chrom_sizes=args.annotations["chrom_sizes"],
            bin_method="naive",
            reverse_complemented=False)

    data_files = glob.glob("{}/h5/*h5".format(args.out_dir))
    print data_files

    # now run dataset and viz
    # set up graph
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        [],
        load_data_from_filename_list,
        1,
        model_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        importances_fn=layerwise_relevance_propagation,
        importances_tasks=args.importances_tasks,
        fake_task_num=118,
        shuffle_data=False, # NOTE: CHANGE LATER
        filter_tasks=[],
        ordered_num_epochs=100)

    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    logging.info("Checkpoint: {}".format(checkpoint_path))

    # extract importances and plot
    prefix = "viz.test"
    extract_importances_and_viz(
        tronn_graph,
        checkpoint_path,
        prefix,
        method="simple_gradients")
        #method="guided_backprop")
        
    return
