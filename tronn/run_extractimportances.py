"""Contains the run function for extracting importances
"""

import os
import glob
import logging

import tensorflow as tf

from tronn.datalayer import load_data_from_filename_list
from tronn.models import models
from tronn.interpretation.importances import extract_importances
from tronn.interpretation.importances import call_importance_peaks


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
        {'data', data_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size / 2,
        models[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        importances_method="guided_backprop")

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

    # TODO(dk) plot as needed
    

    quit()

    # threshold importances
    # this is per task (and separate) because you don't
    # want thresholded importances on negatives (I think?)
    task_nums = [0, 14, 15]
    for task_num_idx in range(len(task_nums)):

        task_num = task_nums[task_num_idx]

        # TODO(dk) make sure this goes in correct folder
        thresholded_importances_mat_h5 = 'task_{}.importances.thresholded.h5'.format(task_num)
        if not os.path.isfile(thresholded_importances_mat_h5):
            call_importance_peaks(data_loader,
                                  importances_mat_h5,
                                  thresholded_importances_mat_h5,
                                  args.batch_size * 4,
                                  task_num,
                                  pval=pval)

    # TODO(dk) plot as needed
    

    

    return None
