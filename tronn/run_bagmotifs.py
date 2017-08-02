"""Contains pipeline for making grammars of various types
"""

import os
import h5py

from tronn.graphs import TronnGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.datalayer import get_total_num_examples
from tronn.nets.nets import model_fns
from tronn.learn.learning import predict
from tronn.interpretation.wkm import get_sequence_communities
from tronn.interpretation.motifs import get_encode_pwms
from tronn.interpretation.motifs import extract_positives_from_motif_mat



def bag_motifs(
        motifs_graph,
        batch_size,
        task_num,
        pwm_list,
        prefix,
        motif_scores_h5,
        pos_motif_mat,
        seq_communities_file,
        grammar_file,
        num_evals=1000):
    """Pipeline for clustering and bagging motifs
    """
    labels, motif_scores, _, metadata = predict(
        motifs_graph, None, batch_size, num_evals=num_evals) # TODO put in num examples instead

    # and save out motif scores to file
    num_examples = motif_scores.shape[0]
    with h5py.File(motif_scores_h5, "w") as out_hf:

        # create datasets
        motif_mat = out_hf.create_dataset(
            'motif_scores', [num_examples, motif_scores.shape[1]])
        labels_mat = out_hf.create_dataset(
            'labels', [num_examples, labels.shape[1]])
        regions_mat = out_hf.create_dataset(
            'regions', [num_examples, 1], dtype='S100')
        motif_names_mat = out_hf.create_dataset(
            'motif_names', [len(pwm_list), 1], dtype='S100')

        # save out motif names
        for i in range(len(pwm_list)):
            motif_names_mat[i] = pwm_list[i].name

        # save in batches
        for batch_idx in range(num_examples / batch_size):
            
            #print batch_idx * batch_size
            batch_start = batch_idx * batch_size
            batch_stop = batch_start + batch_size
            
            # save out to hdf5 file
            if batch_stop < num_examples:
                motif_mat[batch_start:batch_stop,:] = motif_scores[batch_start:batch_stop,:]
                labels_mat[batch_start:batch_stop,:] = labels[batch_start:batch_stop, :]
                regions_mat[batch_start:batch_stop,0] = metadata[batch_start:batch_stop]
            else:
                motif_mat[batch_start:num_examples,:] = motif_scores[batch_start:num_examples,:]
                labels_mat[batch_start:num_examples,:] = labels[batch_start:num_examples]
                regions_mat[batch_start:num_examples,0] = metadata[batch_start:num_examples]
                    
    # extract the positives to cluster
    if not os.path.isfile(pos_motif_mat):
        extract_positives_from_motif_mat(motif_scores_h5, pos_motif_mat, task_num)
                
    # use phenograph, find sequence communities
    if not os.path.isfile(seq_communities_file):
        grammar_file, seq_communities_file, communities = get_sequence_communities(
            pos_motif_mat, prefix)

    return None


def run(args):
    """Run pipeline for bagging motifs
    """
    # get data files, set up graph, etc
    pwm_list = get_encode_pwms(args.motif_file)
    feature_key = "importances_task{}".format(args.task_num)

    
    # set up pwm convolve graph
    motifs_graph = TronnGraph(
        {"data": [args.importance_file]}, # TODO right now, run on importances file
        [],
        load_data_from_filename_list,
        model_fns["pwm_convolve"],
        {"pwms": pwm_list},
        args.batch_size,
        feature_key=feature_key)

    # set up various files
    full_prefix = "{0}/{1}".format(args.out_dir, args.prefix)
    
    motif_scores_h5 = "{0}.motif_scores.h5".format(full_prefix)
    pos_motif_mat = '{0}.wkm_mat.positives.txt.gz'.format(full_prefix)
    seq_communities_file = '{0}.seq_communities.txt'.format(full_prefix)
    grammar_file = '{0}.motif_bags.txt'.format(full_prefix)

    # bag the motifs
    num_evals = get_total_num_examples([args.importance_file], feature_key=feature_key) / args.batch_size
    print num_evals
    bag_motifs(
        motifs_graph,
        args.batch_size,
        args.task_num,
        pwm_list,
        full_prefix,
        motif_scores_h5,
        pos_motif_mat,
        seq_communities_file,
        grammar_file,
        num_evals=num_evals)

    # TODO(dk) make a json output that links motif bags to BED files

    return None
