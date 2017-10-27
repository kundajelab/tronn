# description: some code to extract params to share with other frameworks

import os
import glob
import logging

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim

from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.nets.nets import model_fns
from tronn.learn.cross_validation import setup_cv

from tronn.run_predict import setup_model

from tronn.util.tf_ops import restore_variables_op
from tronn.util.tf_utils import setup_tensorflow_session
from tronn.util.tf_utils import close_tensorflow_session


def extract_params(
        tronn_graph,
        model_checkpoint,
        out_dir,
        prefix,
        skip=["logit", "out"]):
    """Instantiate a model and extract the params and save as numpy arrays
    """

    with tf.Graph().as_default():

        # build graph
        out = tronn_graph.build_graph(data_key="data")

        # extract params
        trainable_params = []
        for v in tf.trainable_variables():
            skip_var = False
            for skip_expr in skip:
                if skip_expr in v.name:
                    skip_var = True
            if not skip_var:
                trainable_params.append(v)

        # run a session to get the params
        sess, coord, threads = setup_tensorflow_session()

        # load model back into graph
        init_assign_op, init_feed_dict = restore_variables_op(model_checkpoint, skip=skip)
        sess.run(init_assign_op, init_feed_dict)
        
        # and extract params
        trainable_params_numpy = sess.run(trainable_params)

        # set up as dictionary
        params_dict = {}
        params_order = []
        for param_idx in xrange(len(trainable_params)):

            # set up name
            param_name = "{}.{}".format(param_idx, trainable_params[param_idx].name)

            # put in dict
            params_dict[param_name] = trainable_params_numpy[param_idx]

            # save in order
            params_order.append(param_name)

        params_dict["param_order"] = np.array(params_order)

        # and save out to numpy array
        np.savez("{}/{}.params.npz".format(out_dir, prefix), **params_dict)
        
        close_tensorflow_session(coord, threads)

        quit()
        
    return


def run(args):
    """Setup things and extract params
    """
    
    # find data files
    # NOTE right now this is technically validation set
    data_files = sorted(glob.glob("{}/*.h5".format(args.data_dir)))
    train_files, valid_files, test_files = setup_cv(data_files, cvfold=0)
    
    # set up model params
    model_fn, model_params = setup_model(args)

    # set up graph, but mostly don't care about things    
    tronn_graph = TronnNeuralNetGraph(
        {"data": train_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size,
        model_fn,
        model_params,
        tf.nn.sigmoid,
        shuffle_data=True)

    # and now call extract params 
    extract_params(tronn_graph, args.model_checkpoint, args.out_dir, args.prefix)
    
    
    return
