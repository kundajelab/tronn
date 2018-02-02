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
        filter_tasks=[]) # TODO adjust this as needed. in best case, want to run through the whole dataset

    # checkpoint file (unless empty net)
    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    elif args.model["name"] == "empty_net":
        checkpoint_path = None
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    logging.info("Checkpoint: {}".format(checkpoint_path))

    # run interpret on the graph
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
            method=args.backprop if args.backprop is not None else "input_x_grad")

    # from here, take a look.
    
    
    
    return None
