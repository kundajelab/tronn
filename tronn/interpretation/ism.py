"""Contains code for in silico mutagenesis screens to 
look at distance dependencies
"""

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_utils import setup_tensorflow_session, close_tensorflow_session



def run_ism_for_motif_pairwise_dependency(ism_graph, model_dir, batch_size, num_evals=100):
    """ISM here
    """
    total_example_num = batch_size * num_evals
    
    with tf.Graph().as_default() as g:

        # setup graph
        synergy_score_tensor = ism_graph.build_graph()

        # setup session
        sess, coord, threads = setup_tensorflow_session()

        # restore the basset part of the model

        # TODO(dk) get checkpoint for model
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
        variables_to_restore = slim.get_model_variables()
        variables_to_restore_tmp = [ var for var in variables_to_restore if ('mutate' not in var.name) ]
        variables_to_restore = variables_to_restore_tmp

        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            checkpoint_path,
            variables_to_restore)
        sess.run(init_assign_op, init_feed_dict)

        # run the model
        synergy_scores = np.zeros((total_example_num))
        motif1_scores = np.zeros((total_example_num))
        motif2_scores = np.zeros((total_example_num))
        for i in range(num_evals):
            # TODO(dk) only keep if the prediction was correct
            
            [synergy_score, motif1_score, motif2_score] = sess.run(synergy_score_tensor)
            synergy_scores[i] = synergy_score
            motif1_scores[i] = motif1_score
            motif2_scores[i] = motif2_score

        synergy_scores[np.isnan(synergy_scores)] = 1.
        synergy_scores[np.isinf(synergy_scores)] = 1.
        motif1_scores[np.isnan(motif1_scores)] = 1.
        motif1_scores[np.isinf(motif1_scores)] = 1.
        motif2_scores[np.isnan(motif2_scores)] = 1.
        motif2_scores[np.isinf(motif2_scores)] = 1.
        
        print np.mean(synergy_scores)
        print np.mean(motif1_scores)
        print np.mean(motif2_scores)
        
        # close
        close_tensorflow_session(coord, threads)

        
    return np.mean(synergy_scores), np.mean(motif1_scores), np.mean(motif2_scores)








