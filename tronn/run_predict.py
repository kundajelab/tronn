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
from tronn.nets.nets import model_fns
from tronn.learn.learning import predict



def setup_model(args):
    """setup model params for various types of nets
    """
    if args.model_type == "nn":
        assert(args.model is not None) and (args.model_dir is not None)
        model_params = args.model
        model_fn = model_fns[args.model['name']]
        
    elif args.model_type == "motif":
        assert args.pwm_files is not None
        model_params = {"pwms": args.pwm_files}
        model_fn = model_fns["pwm_convolve"]
        
    elif args.model_type == "grammar":
        assert (args.pwm_files is not None) and (args.grammar_files is not None)
        model_params = {
            "pwms": args.pwm_files,
            "grammars": args.grammar_files}
        model_fn = model_fns["grammars"]
        
    return model_fn, model_params


def split_by_label(metadata, labels, predictions):
    """Splits output by label set to have mat and BED (just positives) outputs
    """
    for task_idx in range(labels.shape[1]):
        task_key = "task_{}".format(task_idx)
        task_labels = labels[:, task_idx]
        task_probs = probs[:, task_idx]
        task_predictions = predictions[:, task_idx]
        
        # Take task specific info and save out to file
        task_df = pd.DataFrame(
            data=np.stack([task_labels, task_probs, task_predictions], axis=1),
            columns=["labels", "probabilities", "logits"],
            index=metadata)
        
        out_table_file = "{0}/{1}/{2}.predictions.txt".format(
            args.out_dir, args.prefix, task_key)
        task_df.to_csv(out_table_file, sep='\t')
        
        # and convert to BED format too
        # TODO consider only outputting positives?
        task_df['region'] = task_df.index
        task_df['chr'], task_df['start-stop'] = task_df['region'].str.split(':', 1).str
        task_df['start'], task_df['stop'] = task_df['start-stop'].str.split('-', 1).str
        task_df['joint'] = task_df['labels'].map(str) + ";" + task_df['probabilities'].map(str) + ";" + task_df['logits'].map(str)

        out_bed_file = "{0}/{1}/{2}.predictions.bed".format(
            args.out_dir, args.prefix, task_key)
        task_df.to_csv(
            out_bed_file,
            columns=['chr', 'start', 'stop', 'joint'],
            sep='\t',
            header=False,
            index=False)

    return None


def split_by_prediction(metadata, labels, predictions, prefix, fdr=0.05):
    """For each prediction set, threshold and output mat and BED (just positives) after thresholds
    """

    # TODO: also allow splitting by predictions (pheno cluster)
    # to split by predictions, set cutoff to be 0.05 FDR (ie, 5% false pos allowed), relative to a label set.
    # ie, given a grammar set, the cutoff for that grammar should be such that in the appropriate task, 5% above the cutoff are false positives.
    # Then use this to generate BED files for each grammar.

    for prediction_set_idx in range(predictions.shape[1]):
        set_labels = labels[:, prediction_set_idx]
        set_predictions = predictions[:, prediction_set_idx]

        # set up initial data frame to hold data
        prediction_set_df = pd.DataFrame(
            data=np.stack([set_labels, set_predictions], axis=1),
            columns=["labels", "predictions"],
            index=metadata)
        
        # calculate precision and recall
        try:
            precision, recall, thresholds = precision_recall_curve(set_labels, set_predictions)
        except:
            # TODO fix this
            return
            
        # using FDR on precision, get a threshold
        threshold = thresholds[np.searchsorted(precision - fdr, 0)]
        
        # filter on that threshold val
        prediction_set_filt_df = prediction_set_df.loc[prediction_set_df["predictions"] >= threshold]

        # and save that out to text
        prediction_set_filt_df.to_csv("{}.prediction_set{}.txt".format(prefix, prediction_set_idx), sep='\t')
        
        # and convert to BED and save out
        task_df = prediction_set_filt_df
        task_df['region'] = task_df.index
        task_df['chr'], task_df['start-stop'] = task_df['region'].str.split(':', 1).str
        task_df['start'], task_df['stop'] = task_df['start-stop'].str.split('-', 1).str
        task_df['joint'] = task_df['labels'].map(str) + ";" + task_df['probabilities'].map(str) + ";" + task_df['logits'].map(str)

        out_bed_file = "{0}.prediction_set{1}.bed".format(
            prefix, prediction_set_idx)
        task_df.to_csv(
            out_bed_file,
            columns=['chr', 'start', 'stop', 'joint'],
            sep='\t',
            header=False,
            index=False)
    
    return None



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

    assert ((args.model_type == "nn") or
            (args.model_type == "motif") or
            (args.model_type == "grammar"))

    # find data files
    data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))

    # set up model params
    model_fn, model_params = setup_model(args)

    # set up network graph and outputs
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size,
        model_fn,
        model_params,
        tf.nn.sigmoid)
        
    # predict
    labels, predictions, probs, metadata = predict(
        tronn_graph,
        args.model_dir,
        args.batch_size,
        num_evals=args.num_evals)

    # TODO: save out overall large matrix file
    split_by_label(metadata, labels, probs)

    if False:
        split_by_prediction(metadata, labels, probs)

    return None
