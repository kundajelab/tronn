"""Description: get motifs from importance scores
"""

import os
import h5py
import glob
import logging

import numpy as np
import tensorflow as tf

from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.nets.nets import net_fns

from tronn.interpretation.interpret import interpret
from tronn.interpretation.motifs import get_encode_pwms
from tronn.interpretation.motifs import bootstrap_fdr_v2
from tronn.interpretation.motifs import make_motif_x_timepoint_mat

from sklearn.linear_model import LogisticRegression


def run_permutation_test_and_plot(
        h5_file,
        counts_key,
        task_idx,
        pwm_names,
        prefix,
        importances_tasks,
        global_scoring=False):
    """Given the pwm scores, use shuffled nulls to determine which motifs
    are significant beyond a pval
    """
    # bootstrap FDR
    sig_file = bootstrap_fdr_v2(
        h5_file,
        counts_key,
        pwm_names, 
        "{}.permutation_test".format(prefix), 
        task_idx,
        global_importances=global_scoring) # TODO add significance options for bootstrap FDR

    # get the motif x timepoint matrix to look at
    sig_pwm_names = []
    with open(sig_file, "r") as fp:
        for line in fp:
            sig_pwm_names.append(line.strip().split('\t')[0])
    sig_pwm_indices = [i for i in xrange(len(pwm_names)) if pwm_names[i] in sig_pwm_names]
    sig_pwm_names_sorted = [pwm_name.split("_")[0] for pwm_name in pwm_names if pwm_name in sig_pwm_names]
    
    mat_file = make_motif_x_timepoint_mat(
        h5_file, 
        ["pwm-counts.taskidx-{}".format(i) for i in xrange(len(importances_tasks))], 
        importances_tasks, 
        sig_pwm_indices, 
        sig_pwm_names_sorted,
        prefix=prefix)

    # plot
    out_plot = "{}.pdf".format(mat_file.split(".txt")[0])
    os.system("plot_tfs.R {} {}".format(mat_file, out_plot))

    return


def run_logistic_regression(
        h5_file,
        feature_key,
        label_key,
        task_idx,
        C=1):
    """Run logistic regression on the motif mat
    """
    print "running logistic regression"
    with h5py.File(h5_file, "r") as hf:
        # get X and y
        X = hf[feature_key]
        y = hf[label_key][:,task_idx]

        # get a balanced set
        # NOTE this naturally lends itself to a bootstrap-ish situation?
        X_pos = X[y > 0,:]
        y_pos = y[y > 0]
        pos_size = X_pos.shape[0]

        X_neg = X[y == 0,:]
        y_neg = y[y == 0]
        neg_size = X_neg.shape[0]
        idx = np.random.randint(neg_size, size=pos_size)
        X_neg = X_neg[idx,:]
        y_neg = y_neg[idx]
    
        X_bal = np.vstack((X_pos, X_neg))
        y_bal = np.hstack((y_pos, y_neg))
    
    # fit
    model = LogisticRegression(C=C, penalty="l1", tol=0.0001)
    model.fit(X_bal, y_bal)
    print model.score(X_bal, y_bal)

    coefs = model.coef_.ravel()
    
    return np.where(coefs != 0)


def run(args):
    """Find motifs (global and interpretation task specific groups)
    """
    os.system('mkdir -p {}'.format(args.out_dir))
    
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
        args.batch_size,
        net_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        importances_fn=net_fns["importances_to_motif_assignments"], # TODO change this to allow input of inference stack
        #importances_fn=net_fns["get_importances"], # TODO change this to allow input of inference stack
        importances_tasks=args.importances_tasks,
        shuffle_data=True,
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
            args.batch_size,
            pwm_counts_mat_h5,
            args.sample_size,
            pwm_list,
            keep_negatives=True,
            filter_by_prediction=True,
            method=args.backprop if args.backprop is not None else "input_x_grad") # simple_gradients or guided_backprop


    global_task_idx = len(args.importances_tasks)
    
    # TODO: for each task of interest, build an ML model
    # (simplest: L1 regularized logistic regression) to
    # extract top motif features per task
    # take that master set and save out list as well as ordering
    # by correlation
    # don't viz correlations at this stage?
    all_pwm_indices = []
    for i in xrange(len(args.interpretation_tasks)):
        interpretation_task_idx = args.interpretation_tasks[i]
        # TODO eventually convert this to bootstrapped version
        pwm_indices = run_logistic_regression(
            pwm_counts_mat_h5,
            "pwm-counts.taskidx-{}".format(global_task_idx),
            "labels",
            interpretation_task_idx)
        print "task:", pwm_indices[0].tolist()
        all_pwm_indices = sorted(list(set(all_pwm_indices + pwm_indices[0].tolist())))
        print "all:", all_pwm_indices
        print len(all_pwm_indices)

        # save out PWMs per task
        # TODO
        sig_file = "task-{}.permutation_test.cutoff.txt".format(interpretation_task_idx)
        with open(sig_file, "w") as out:
            for pwm_idx in pwm_indices[0].tolist():
                out.write("{}\n".format(pwm_list[pwm_idx].name))

    # save out selected PWMs
    sig_file = "global.permutation_test.cutoff.txt"
    with open(sig_file, "w") as out:
        for pwm_idx in all_pwm_indices:
            out.write("{}\tNA\n".format(pwm_list[pwm_idx].name))
        
    import ipdb
    ipdb.set_trace()

    quit()

    # first global permutation test

    run_permutation_test_and_plot(
        pwm_counts_mat_h5,
        "pwm-counts.taskidx-{}".format(global_task_idx),
        global_task_idx,
        pwm_names,
        "{}/global".format(args.out_dir),
        args.importances_tasks,
        global_scoring=True)
        
    # get motif sets for interpretation tasks (ie clusters)
    print "running pwm sets for clusters"
    for i in xrange(len(args.interpretation_tasks)):
        interpretation_task_idx = args.interpretation_tasks[i]

        print interpretation_task_idx
        run_permutation_test_and_plot(
            pwm_counts_mat_h5,
            "pwm-counts.taskidx-{}".format(global_task_idx),
            interpretation_task_idx,
            pwm_names,
            "{}/task-{}".format(args.out_dir, interpretation_task_idx),
            args.importances_tasks,
            global_scoring=False)

    # at the end of this, return a master list that will be used to scan each time
    # for grammars?

    return
