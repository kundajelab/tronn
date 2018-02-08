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
    """Find grammars utilizing the timeseries tasks
    """
    os.system('mkdir -p {}'.format(args.tmp_dir))
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logging.info("Found {} chrom files".format(len(data_files)))

    # given a grammar file (always with pwm file) scan for grammars.
    grammars = read_grammar_file(args.grammar_file, args.pwm_file)

    # pull in motif annotation
    pwm_list_file = "global.pwm_names.txt" # TODO - fix this
    pwm_name_to_hgnc, hgnc_to_pwm_name = setup_pwm_metadata(args.pwm_metadata_file)
    # TODO add an arg here to just use all if no list
    pwm_list, pwm_list_filt, pwm_list_filt_indices, pwm_names_filt = setup_pwms(args.pwm_file, pwm_list_file)
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)

    # ==============
    # for debug
    # ==============
    for idx in xrange(len(grammars)):
        grammars_tmp = [grammars[idx]]
        print idx, [pwm_name_to_hgnc[name] for grammar in grammars_tmp for name in grammar.nodes]

    grammars = [grammars[0]]
    print "DEBUG: using", [pwm_name_to_hgnc[name] for grammar in grammars for name in grammar.nodes]
    
    # set up graph
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
        inference_fn=net_fns[args.inference_fn], # write an assert? to check it's the right fn being passed?
        importances_tasks=args.importances_tasks,
        shuffle_data=True,
        filter_tasks=args.interpretation_tasks) # TODO adjust this as needed. in best case, want to run through the whole dataset

    # checkpoint file (unless empty net)
    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    elif args.model["name"] == "empty_net":
        checkpoint_path = None
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    logging.info("Checkpoint: {}".format(checkpoint_path))

    if False:
        # validate
        visualize_only = True
        validate_grammars = True
    else:
        # just scan
        visualize_only = False
        validate_grammars = False
        
    # run interpret on the graph
    # TODO perform visualization of importance scores in here
    score_mat_h5 = '{0}/{1}.grammar_scores.h5'.format(
        args.tmp_dir, args.prefix)
    if not os.path.isfile(score_mat_h5):
        interpret(
            tronn_graph,
            checkpoint_path,
            args.batch_size,
            score_mat_h5,
            args.sample_size,
            {"pwms": pwm_list, "grammars": grammars},
            keep_negatives=False,
            filter_by_prediction=True,
            visualize_only=visualize_only, # toggle this with validate grammars?
            scan_grammars=True,
            validate_grammars=validate_grammars, # add a flag here to toggle
            method=args.backprop if args.backprop is not None else "input_x_grad")


    # TODO - write a function for this, can append as an option to any filtered set of importance scores
    # TODO - take the importances (global) and generate modisco. make it so that can run on any dataset
    # to look at timepoint specific ones later.


    # TODO - visualize important motifs by positioning. make a profile map, output BED file too
    # then can use deeptools + bigwigs to quickly view


    # TODO - confusion matrix - what timepoints and what tasks are most enriched? should be able to
    # recover expected timepoints and tasks.
    # make a region x timepoint (for 1 grammar) matrix - pull from the hdf5 file
    # make a grammar x timepoint (collapse the regions grammar)
    # make a grammar x task matrix (tasks ordered by waves of accessibility)
    
    
    
    
    
    return None
