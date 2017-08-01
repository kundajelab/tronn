"""Contains code for in silico mutagenesis screens to 
look at distance dependencies
"""

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_utils import setup_tensorflow_session, close_tensorflow_session



def run_ism_for_motif_pairwise_dependency(ism_graph, num_evals=100):
    """ISM here
    """
    
    with tf.Graph().as_default() as g:

        # setup graph
        synergy_score = ism_graph.build_graph()

        # setup session
        setup_tensorflow_session()

        # restore the basset part of the model
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
            # TODO(dk) also track individual motif scores
            
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



def scan_motif_group(mutate=False):
    """Scans to get sequences that have all motifs in group
    and saves them out to new file for further analysis
    Save out: sequence, motif locations (can calculate distances)

    """

    
    return None




def ism_for_synergistic_distance():
    """Do in silico mutagenesis to get synergistic distances between
    pairs of motifs
    """

    # for every pair of motifs (that are not the same):


    # for each sequence:
    # for each spot in motif 1:
    # mutate position (just drop out, or try other three base pairs)
    # 

    

    return None








