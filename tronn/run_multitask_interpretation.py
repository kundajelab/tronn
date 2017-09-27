# description: test function for a multitask interpretation pipeline

import os
import glob
import logging

import tensorflow as tf

from tronn.graphs import TronnGraph
from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.nets.nets import model_fns
from tronn.interpretation.importances import extract_importances
from tronn.interpretation.importances import layerwise_relevance_propagation
from tronn.interpretation.importances import call_importance_peaks_v2
from tronn.interpretation.importances import visualize_sample_sequences

from tronn.datalayer import get_total_num_examples

def run(args):

    # build an inference graph and run it
    # output: (example, task, 4, 1000)
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
        model_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        importances_fn=layerwise_relevance_propagation,
        importances_tasks=args.importances_tasks,
        shuffle_data=False)

    # checkpoint file
    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    logging.info("Checkpoint: {}".format(checkpoint_path))
    
    # get importances
    importances_mat_h5 = '{0}/{1}.importances.h5'.format(args.tmp_dir, args.prefix)
    if not os.path.isfile(importances_mat_h5):
        extract_importances(
            tronn_graph,
            checkpoint_path,
            importances_mat_h5,
            args.sample_size,
            method="guided_backprop")

    print "DONE FOR NOW"
    quit()
    

    # from there divide up into tasks you care about
    # per task:


    # get all examples belonging to that task (go through file once and distribute out)
    # here, can cull for correctly predicted (write separate function for this)
    # output: (example, task, 4, 1000)
    


    
    # here, figure out how to extract important seqlets
    # likely can just threshold and keep big ones
    # and normalize here
    # output: (seqlet, task)


    # then cluster seqlets - phenograph
    # output: (seqlet, task) but clustered


    # then hAgglom the seqlets
    # output: motifs
    
    

    return
