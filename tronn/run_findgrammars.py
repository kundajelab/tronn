# description: test function for a multitask interpretation pipeline

import os
import glob
import logging

import tensorflow as tf

from tronn.graphs import TronnGraph
from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.nets.nets import model_fns
from tronn.interpretation.importances import extract_importances_and_motif_hits
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



def run(args):
    """Find grammars utilizing the timeseries tasks
    """
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
        model_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        importances_fn=layerwise_relevance_propagation,
        importances_tasks=args.importances_tasks,
        shuffle_data=False,
        filter_tasks=args.interpretation_tasks)

    # checkpoint file
    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    logging.info("Checkpoint: {}".format(checkpoint_path))

    # with inference this produces the importances and motif hits.
    # save all out
    pwm_list = get_encode_pwms(args.pwm_file)
    
    # get importances
    importances_mat_h5 = '{0}/{1}.importances.h5'.format(args.tmp_dir, args.prefix)
    if not os.path.isfile(importances_mat_h5):
        extract_importances_and_motif_hits(
            tronn_graph,
            checkpoint_path,
            importances_mat_h5,
            args.sample_size,
            pwm_list,
            method="guided_backprop")

    quit()

        
    # from there divide up into tasks you care about
    # per task:
    prefix = "{0}/{1}.importances".format(args.tmp_dir, args.prefix)

    # split into task files
    task_importance_files = glob.glob("{}.task*".format(prefix))
    if len(task_importance_files) == 0:
        split_importances_by_task_positives(
            importances_mat_h5, args.interpretation_tasks, prefix)

    # per task file (use parallel processing):
    # extract the seqlets into other files with timepoints (seqlet x task)
    # AND keep track of seqlet size
    for task in args.interpretation_tasks:
        
        task_importance_file = "{}.task_{}.h5".format(prefix, task)
        task_seqlets_file = "{}.task_{}.seqlets.h5".format(prefix, task)

        if not os.path.isfile(task_seqlets_file):
            extract_seqlets(task_importance_file, args.importances_tasks, task_seqlets_file)

        # TODO filter seqlets
        task_seqlets_filt_file = "{}.task_{}.seqlets.filt.h5".format(prefix, task)

        if not os.path.isfile(task_seqlets_filt_file):
            reduce_seqlets(task_seqlets_file, task_seqlets_filt_file)
        
        # then cluster seqlets - phenograph
        # output: (seqlet, task) but clustered
        #cluster_seqlets("{}.task_{}.seqlets.h5".format(prefix, task))
        seqlets_w_communities_file = "{}.task_{}.seqlets.filt.communities.h5".format(prefix, task)

        if not os.path.isfile(seqlets_w_communities_file):
            cluster_seqlets(task_seqlets_filt_file, seqlets_w_communities_file)

        # then hAgglom the seqlets
        # output: motifs
        motif_dir = "{}/{}.motifs.task_{}".format(args.tmp_dir, args.prefix, task)
        motif_prefix = "{}/{}.task_{}".format(motif_dir, args.prefix, task)
        if not os.path.isdir(motif_dir):
            os.system("mkdir -p {}".format(motif_dir))
            make_motif_sets(seqlets_w_communities_file, motif_prefix)

        # and generate quick plots
        trajectories_txt = "{}.master.community_trajectories.txt".format(motif_prefix)
        communities_txt = "{}.master.community_to_motif.txt".format(motif_prefix)
        trajectories_png = "{}.master.community_trajectories.hclust.png".format(motif_prefix)
        communities_png = "{}.master.community_to_motif.hclust.png".format(motif_prefix)
        communities_corr_png = "{}.master.community_to_motif.corr.png".format(motif_prefix)
        os.system("Rscript make_motif_maps.R {} {} {} {} {}".format(
            communities_txt,
            trajectories_txt,
            trajectories_png,
            communities_png,
            communities_corr_png))
        
    return
