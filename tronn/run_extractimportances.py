"""Contains the run function for extracting importances
"""

import os
import glob
import logging

import tensorflow as tf

from tronn.graphs import TronnGraph
from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.architectures import models
from tronn.architectures import stdev_cutoff
from tronn.interpretation.importances import extract_importances
from tronn.interpretation.importances import layerwise_relevance_propagation
from tronn.interpretation.importances import call_importance_peaks_v2
from tronn.interpretation.importances import visualize_sample_sequences


def run(args):
    """Run pipeline to extract importance scores
    """
    logging.info('Running extractimportances...')
    
    # set up scratch_dir
    os.system('mkdir -p {}'.format(args.tmp_dir))
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logging.info("Found {} chrom files".format(len(data_files)))

    # set up graph
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size / 2,
        models[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        importances_fn=layerwise_relevance_propagation)

    # checkpoint file
    checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    logging.info("Checkpoint: {}".format(checkpoint_path))

    # get importances
    importances_mat_h5 = '{0}/{1}.importances.h5'.format(args.tmp_dir, args.prefix)
    if not os.path.isfile(importances_mat_h5):
        extract_importances(
            tronn_graph,
            args.model_dir,
            importances_mat_h5,
            args.sample_size,
            method="guided_backprop")

    # plot as needed
    if args.plot_samples:
        sample_viz_dir = "{0}/{1}.seq_viz_samples".format(args.tmp_dir, args.prefix)
        os.system("mkdir -p {}".format(sample_viz_dir))
        visualize_sample_sequences(importances_mat_h5, 0, sample_viz_dir)

    # threshold importances
    # this is per task (and separate) because you don't
    # want thresholded importances on negatives so number of examples in each is different
    #task_nums = [0, 14, 15]
    task_nums = [0]
    for task_num_idx in range(len(task_nums)):

        task_num = task_nums[task_num_idx]

        # set up graph
        callpeak_graph = TronnGraph(
            {"data": [importances_mat_h5]},
            [task_num],
            load_data_from_filename_list,
            stdev_cutoff,
            {"pval": 0.05},
            args.batch_size * 4,
            feature_key="importances_task{}".format(task_num))

        # Get thresholded importances
        thresholded_importances_mat_h5 = '{0}/{1}.task_{2}/{1}.task_{2}.importances.thresholded.h5'.format(
            args.out_dir, args.prefix, task_num)
        if not os.path.isfile(thresholded_importances_mat_h5):
            call_importance_peaks_v2(importances_mat_h5, callpeak_graph, thresholded_importances_mat_h5)

        # plot as needed
        if args.plot_samples:
            sample_viz_dir = "{0}/{1}.task_{2}/{1}.task_{2}.seq_viz_samples".format(
                args.out_dir, args.prefix, task_num)
            os.system("mkdir -p {}".format(sample_viz_dir))
            visualize_sample_sequences(thresholded_importances_mat_h5, task_num, sample_viz_dir)
        

    

    return None
