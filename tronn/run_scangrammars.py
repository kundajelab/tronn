# description: scan for grammar scores

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
    logger.info("Running motif scan")
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

    # pull in motif annotation
    pwm_name_to_hgnc, hgnc_to_pwm_name = setup_pwm_metadata(args.pwm_metadata_file)
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_names_clean = [pwm_name.split("_")[0] for pwm_name in pwm_names]
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
            filter_by_prediction=True)

    # save out text version of score file
    score_mat_file = "{}/{}.grammar-scores.txt".format(args.out_dir, args.prefix)
    if not os.path.isfile(score_mat_file):
        h5_dataset_to_text_file(
            score_mat_h5,
            "grammar-scores.taskidx-10",
            score_mat_file,
            xrange(len(grammar_sets)),
            [os.path.basename(grammar_file) for grammar_file in args.grammar_files])

    if visualize:
        # TODO plot the matrix of scores
        pass
        
    # validation - give a confusion matrix after re-scanning, if metacommunity bed files available
    
    

    # give an option here to optimize thresholds and save into new grammar files
    

    if args.validate:
        # here always give back position plot, so can look at where the motifs are
        # relative to each other

        # NOTE: this is a larger general function - use labels in h5 file in conjunction with
        # example information
        # TODO - confusion matrix - what timepoints and what tasks are most enriched? should be able to
        # recover expected timepoints and tasks.
        # make a region x timepoint (for 1 grammar) matrix - pull from the hdf5 file
        # make a grammar x timepoint (collapse the regions grammar)
        # make a grammar x task matrix (tasks ordered by waves of accessibility)
        pass
    
    return None
