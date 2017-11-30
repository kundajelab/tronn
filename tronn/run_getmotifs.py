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
from tronn.nets.nets import net_fns

from tronn.interpretation.interpret import interpret

from tronn.datalayer import get_total_num_examples

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import get_encode_pwms
from tronn.interpretation.motifs import bootstrap_fdr_v2
from tronn.interpretation.motifs import make_motif_x_timepoint_mat


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

    # set up graph
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size / 2,
        net_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        importances_fn=net_fns["importances_to_motif_assignments"], # TODO change this to allow input of inference stack
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
        interpret(
            tronn_graph,
            checkpoint_path,
            pwm_counts_mat_h5,
            args.sample_size,
            pwm_list,
            method=args.backprop if args.backprop is not None else "input_x_grad") # simple_gradients or guided_backprop

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

        # and now can take this new master list and then go through timepoints and extract counts per timepoint
        master_pwm_names = []
        with open("global.bootstrap_fdr.cutoff.txt", "r") as fp:
            for line in fp:
                master_pwm_names.append(line.strip().split('\t')[0])
            
        master_pwm_indices = [i for i in xrange(len(pwm_names)) if pwm_names[i] in master_pwm_names]
        master_pwm_names_sorted = [pwm_name.split("_")[0] for pwm_name in pwm_names if pwm_name in master_pwm_names]
        key_list = ["pwm-counts.taskidx-{}".format(i) for i in xrange(10)]
        make_motif_x_timepoint_mat(
            pwm_counts_mat_h5, 
            key_list, 
            args.importances_tasks, 
            master_pwm_indices, 
            master_pwm_names_sorted,
            prefix="global.")

        # and plot with R
        os.system("Rscript plot_tfs.R global.pwm_x_timepoint.mat.txt global.pwm_x_timepoint.pdf")
        
        
    if True:
        # get motif sets for interpretation tasks (ie clusters)
        # load only specific PWMs, from the global set?
        print "running pwm sets for clusters"
        for i in xrange(len(args.interpretation_tasks)):
            interpretation_task_idx = args.interpretation_tasks[i]
            print interpretation_task_idx
            bootstrap_fdr_v2(
                pwm_counts_mat_h5, 
                pwm_names, 
                "interpretation.task-{}".format(interpretation_task_idx), 
                interpretation_task_idx, 
                "pwm-counts.taskidx-{}".format(10)) # pull from global importances, not timepoints
    
            # and now can take this new master list and then go through timepoints and extract counts per timepoint
            master_pwm_names = []
            motif_results_list = "interpretation.task-{}.bootstrap_fdr.cutoff.txt".format(interpretation_task_idx)
            with open(motif_results_list, "r") as fp:
                for line in fp:
                    master_pwm_names.append(line.strip().split('\t')[0])
            
            master_pwm_indices = [i for i in xrange(len(pwm_names)) if pwm_names[i] in master_pwm_names]
            master_pwm_names_sorted = [pwm_name.split("_")[0] for pwm_name in pwm_names if pwm_name in master_pwm_names]
            key_list = ["pwm-counts.taskidx-{}".format(i) for i in xrange(10)]
            make_motif_x_timepoint_mat(
                pwm_counts_mat_h5, 
                key_list, 
                args.importances_tasks, 
                master_pwm_indices, 
                master_pwm_names_sorted,
                prefix="interpretation.task-{}.".format(interpretation_task_idx))

            # and plot with R
            os.system(
                ("Rscript plot_tfs.R "
                 "interpretation.task-{0}.pwm_x_timepoint.mat.txt "
                 "interpretation.task-{0}.pwm_x_timepoint.pdf").format(interpretation_task_idx))


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
