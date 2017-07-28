"""Contains code to run predictions
"""

import os
import glob
import logging

import numpy as np
import tensorflow as tf

from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.architectures import models
from tronn.learn.learning import predict


def run(args):
    """Pipeline to output predictions for regions.
    
    Sets up a graph and then runs the graph for some number of batches
    to get predictions. Then, per task, these are saved out into 
    BED style files with label and probabilities.
    """
    logging.info("Running predict...")
    os.system("mkdir -p {}".format(args.out_dir))
    
    # find data_files
    data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))
    
    # set up neural network graph
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size,
        models[args.model['name']],
        args.model,
        tf.nn.sigmoid)
  
    # and predict
    labels, predictions, probs = predict(
        tronn_graph,
        args.model_dir,
        args.batch_size,
        num_evals=args.num_evals)
    
    import ipdb
    ipdb.set_trace()


    
    

    return None
