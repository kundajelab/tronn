"""Contains code to run predictions
"""

import os
import glob
import logging

import numpy as np
import pandas as pd
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

    TODO(dk) extend this to motifs scans, grammar scans

    """
    logging.info("Running predict...")
    os.system("mkdir -p {}".format(args.out_dir))
    os.system("mkdir -p {0}/{1}".format(args.out_dir, args.prefix))
    
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
  
    # predict
    labels, predictions, probs, metadata = predict(
        tronn_graph,
        args.model_dir,
        args.batch_size,
        num_evals=args.num_evals)

    # per task
    for task_idx in range(labels.shape[1]):
        task_key = "task_{}".format(task_idx)
        logging.info("Predicting on {}...".format(task_key))
        
        task_labels = labels[:, task_idx]
        task_probs = probs[:, task_idx]
        task_predictions = predictions[:, task_idx]
        
        # With outputs, save out to file
        out_table_file = "{0}/{1}/{2}.predictions.txt".format(
            args.out_dir, args.prefix, task_key)
        task_df = pd.DataFrame(
            data=np.stack([
                task_labels,
                task_probs,
                task_predictions], axis=1),
            columns=["labels", "probabilities", "logits"],
            index=metadata)
        task_df.to_csv(out_table_file, sep='\t')
        
        # and convert to BED format too
        task_df['region'] = task_df.index
        task_df['chr'], task_df['start-stop'] = task_df['region'].str.split(':', 1).str
        task_df['start'], task_df['stop'] = task_df['start-stop'].str.split('-', 1).str

        task_df['joint'] = task_df['labels'].map(str) + ";" + task_df['probabilities'].map(str) + ";" + task_df['logits'].map(str)

        out_bed_file = "{0}/{1}/{2}.predictions.bed".format(
            args.out_dir, args.prefix, task_key)
        task_df.to_csv(out_bed_file, columns=['chr', 'start', 'stop', 'joint'], sep='\t', header=False, index=False)

    return None
