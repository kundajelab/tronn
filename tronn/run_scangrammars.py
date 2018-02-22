# description: test function for a multitask interpretation pipeline

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

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

    # ==============
    #for idx in xrange(len(grammars)):
    #    grammars_tmp = [grammars[idx]]
    #    print idx, [pwm_name_to_hgnc[name] for grammar in grammars_tmp for name in grammar.nodes]

    #grammars = [grammars[0]]
    #print "DEBUG: using", [pwm_name_to_hgnc[name] for grammar in grammars for name in grammar.nodes]
    
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
            visualize=args.plot_importance_sample,
            scan_grammars=True,
            validate_grammars=False,
            filter_by_prediction=True)
        
    # here always give back position plot, so can look at where the motifs are
    # relative to each other

    # NOTE: this is a larger general function - use labels in h5 file in conjunction with
    # example information
    # TODO - confusion matrix - what timepoints and what tasks are most enriched? should be able to
    # recover expected timepoints and tasks.
    # make a region x timepoint (for 1 grammar) matrix - pull from the hdf5 file
    # make a grammar x timepoint (collapse the regions grammar)
    # make a grammar x task matrix (tasks ordered by waves of accessibility)
    
    return None
