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
from tronn.datalayer import load_data_with_shuffles_from_filename_list
from tronn.datalayer import H5DataLoader

from tronn.nets.nets import net_fns

from tronn.interpretation.interpret import interpret

from tronn.interpretation.motifs import read_pwm_file
from tronn.interpretation.motifs import setup_pwms
from tronn.interpretation.motifs import setup_pwm_metadata

from tronn.interpretation.grammars import read_grammar_file


def visualize_scores(
        h5_file,
        dataset_key):
    """Visualize clustering. Note that the R script is downsampling
    to make things visible.
    """
    # do this in R
    plot_example_x_pwm = (
        "plot.example_x_pwm_mut.from_h5.R {0} {1}").format(
            h5_file, dataset_key)
    print plot_example_x_pwm
    os.system(plot_example_x_pwm)
    
    return None


def run(args):
    """Scan and score grammars
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running grammar scan")
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

    assert len(grammar_sets) == 1

    # pull in motif annotation
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))
    
    # set up file loader, dependent on importance fn
    #if args.backprop == "integrated_gradients":
    #    data_loader_fn = load_step_scaled_data_from_filename_list
    #elif args.backprop == "deeplift":
    #    data_loader_fn = load_data_with_shuffles_from_filename_list
    #else:
    #    data_loader_fn = load_data_from_filename_list
        #data_loader_fn = load_data_with_shuffles_from_filename_list
        #print "set for shuffles!"
    dataloader = H5DataLoader(
        {"data": data_files},
        filter_tasks=args.filter_tasks)
        
    # set up graph
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        args.tasks,
        dataloader,
        args.batch_size,
        net_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        inference_fn=net_fns[args.inference_fn],
        importances_tasks=args.inference_tasks,
        shuffle_data=True,
        filter_tasks=args.filter_tasks,
        checkpoints=args.model_checkpoints)

    # checkpoint file (unless empty net)
    #if args.model_checkpoint is not None:
    #    checkpoint_path = args.model_checkpoint
    #elif args.model["name"] == "empty_net":
    #    checkpoint_path = None
    #else:
    #    checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    #logging.info("Checkpoint: {}".format(checkpoint_path))

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
        # run interpret
        interpret(
            tronn_graph,
            None,
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

        # save PWM names with the mutation dataset in hdf5
        with h5py.File(score_mat_h5, "a") as hf:

            # get motifs from grammar
            motifs = []
            for grammar in grammar_sets[0]:
                motifs += np.where(grammar.pwm_thresholds > 0)[0].tolist()
            pwm_indices = sorted(list(set(motifs)))

            # get names
            pwm_names = [pwm_list[i].name.split(".")[0].split("_")[1] for i in pwm_indices]

            # attach to delta logits and mutated scores
            hf["delta_logits"].attrs["pwm_mut_names"] = pwm_names
            for task_idx in args.inference_tasks:
                hf["dmim-scores.taskidx-{}".format(task_idx)].attrs["pwm_mut_names"] = pwm_names

    # and plot stuff out
    plot_summary = (
        "plot.pwm_x_pwm.mut2.from_h5.R {0} "
        "logits pwm-scores delta_logits dmim-scores pwm_mut_names {1}/{2} {3}").format(
            score_mat_h5,
            args.out_dir,
            grammar_sets[0][0].name.split(".")[0],
            " ".join([str(i) for i in args.inference_tasks]))
    print plot_summary
    os.system(plot_summary)
        
    if False:
        # get back the dataset keys and plot out
        dataset_keys = ["dmim-scores.taskidx-{}".format(i)
                        for i in args.inference_tasks]
        for i in xrange(len(dataset_keys)):
            visualize_scores(
                score_mat_h5,
                dataset_keys[i])

    # give an option here to optimize thresholds and save into new grammar files?
    

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
