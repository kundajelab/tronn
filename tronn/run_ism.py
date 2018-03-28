# description: run ISM on the motifs to build linear (+pairwise interaction) models

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from tronn.util.h5_utils import h5_dataset_to_text_file

from tronn.graphs import TronnGraph
from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.nets.nets import net_fns

from tronn.interpretation.interpret import interpret

from tronn.interpretation.motifs import read_pwm_file
from tronn.interpretation.motifs import setup_pwms
from tronn.interpretation.motifs import setup_pwm_metadata

from tronn.interpretation.grammars import read_grammar_file



def run(args):
    """Scan and score grammars
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running motif in silico mutagenesis")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logging.info("Found {} chrom files".format(len(data_files)))

    # given a grammar file (always with pwm file) scan for grammars.
    grammar_sets = []
    for grammar_file in args.grammar_files:
        grammar_sets.append(read_grammar_file(grammar_file, args.pwm_file))
    assert len(grammar_sets) == 1 # don't do more than one at a time? maybe adjust later

    # pull in motif annotation
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))
    
    # set up file loader, dependent on importance fn
    if args.backprop == "integrated_gradients":
        data_loader_fn = load_step_scaled_data_from_filename_list
    elif args.backprop == "deeplift":
        data_loader_fn = load_data_with_shuffles_from_filename_list
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
        importances_tasks=args.inference_tasks,
        shuffle_data=True,
        filter_tasks=args.filter_tasks)

    # checkpoint file (unless empty net)
    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    elif args.model["name"] == "empty_net":
        checkpoint_path = None
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    logging.info("Checkpoint: {}".format(checkpoint_path))

    # check if validation flag is set
    if args.validate:
        visualize = True
        validate_grammars = True
    else:
        visualize = args.plot_importance_sample
        validate_grammars = False

    if visualize == True:
        viz_dir = "{}/viz".format(args.out_dir)
        os.system("mkdir -p {}".format(viz_dir))
        
    # run interpret on the graph
    # this should give you back everything with scores, then set the cutoff after
    score_mat_h5 = '{0}/{1}.grammar-scores.h5'.format(
        args.tmp_dir, args.prefix)
    if not os.path.isfile(score_mat_h5):
        interpret(
            tronn_graph,
            checkpoint_path,
            args.batch_size,
            score_mat_h5,
            args.sample_size,
            {"importances_fn": args.backprop,
             "pwms": pwm_list,
             "grammars": grammar_sets},
            keep_negatives=False,
            visualize=visualize,
            scan_grammars=True,
            validate_grammars=validate_grammars,
            filter_by_prediction=False)

    # TODO - here want to save out grammars as linear models, based on scores that came out

    


    
    return None
