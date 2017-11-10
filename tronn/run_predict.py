"""Contains code to run predictions
"""

import os
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.stats import zscore
from scipy.special import expit

from sklearn.metrics import precision_recall_curve

from tronn.preprocess import generate_ordered_single_file_nn_dataset

from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.nets.nets import model_fns
from tronn.learn.learning import predict
from tronn.learn.learning import predict_variant_scores
from tronn.interpretation.motifs import get_encode_pwms

from tronn.interpretation.importances import layerwise_relevance_propagation

def load_pwms(pwm_file_list):
    """Load pwm files from list of filenames into one dict
    """
    pwms = {}
    for pwm_file in pwm_file_list:
        pwms_tmp = get_encode_pwms(pwm_file, as_dict=True)
        pwms.update(pwms_tmp)
        
    return pwms


def setup_model(args):
    """setup model params for various types of nets
    """
    if args.model_type == "nn":
        assert(args.model is not None) and ((args.model_dir is not None) or (args.model_checkpoint is not None))
        model_params = args.model
        model_fn = model_fns[args.model['name']]
        
    elif args.model_type == "motif":
        assert args.pwm_files is not None
        model_params = {"pwms": load_pwms(args.pwm_files)}
        model_fn = model_fns["pwm_convolve"]
        
    elif args.model_type == "grammar":
        assert (args.pwm_files is not None) and (args.grammar_files is not None)
        model_params = {
            "pwms": load_pwms(args.pwm_files),
            "grammars": args.grammar_files}
        model_fn = model_fns["grammars"]
        
    return model_fn, model_params


def scores_to_probs(np_array):
    """In a case where prediction scores are in a numpy matrix,
    zscore and push through sigmoid
    """
    zscore_array = zscore(np_array, axis=0)
    probs = expit(zscore_array)

    return probs


def split_by_label(metadata, labels, predictions, probs, out_dir, prefix):
    """Splits output by label set to have mat and BED (just positives) outputs
    """
    os.system("mkdir -p {}/by_label".format(out_dir))
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
        
        out_table_file = "{0}/by_label/{1}.{2}.predictions.txt".format(
            out_dir, prefix, task_key)
        task_df.to_csv(out_table_file, sep='\t')
        
        # and convert to BED format too
        # TODO consider only outputting positives?
        task_df['region'] = task_df.index
        task_df['chr'], task_df['start-stop'] = task_df['region'].str.split(':', 1).str
        task_df['start'], task_df['stop'] = task_df['start-stop'].str.split('-', 1).str
        task_df['joint'] = task_df['labels'].map(str) + ";" + task_df['probabilities'].map(str) + ";" + task_df['logits'].map(str)

        out_bed_file = "{0}/by_label/{1}.{2}.predictions.bed".format(
            out_dir, prefix, task_key)
        task_df.to_csv(
            out_bed_file,
            columns=['chr', 'start', 'stop', 'joint'],
            sep='\t',
            header=False,
            index=False)

    return None


def split_single_label_set_by_prediction(
        metadata,
        labels,
        predictions,
        probs,
        out_dir,
        prefix,
        fdr=0.05):
    """For each prediction set, threshold and output mat and BED (just positives) after thresholds

    Args:
      Note that labels is 1D here

    """
    # to split by predictions, set cutoff to be 0.05 FDR (ie, 5% false pos allowed), relative to a label set.
    # ie, given a grammar set, the cutoff for that grammar should be such that in the appropriate task, 5% above the cutoff are false positives.
    # Then use this to generate BED files for each grammar.
    os.system("mkdir -p {}/by_prediction".format(out_dir))
    
    for prediction_set_idx in range(predictions.shape[1]):
        set_predictions = predictions[:, prediction_set_idx]
        set_probs = probs[:, prediction_set_idx]

        # set up initial data frame to hold data
        prediction_set_df = pd.DataFrame(
            data=np.stack([labels, set_probs, set_predictions], axis=1),
            columns=["labels", "probabilities", "logits"],
            index=metadata)
        
        # calculate precision and recall
        precision, recall, thresholds = precision_recall_curve(labels, set_probs)
        try:
            precision, recall, thresholds = precision_recall_curve(labels, set_probs)
        except:
            # TODO fix this
            continue

        # using FDR on precision, get a threshold
        threshold = thresholds[np.searchsorted(precision - (1-fdr), 0)-1] # hack here, make sure this is right
        
        # filter on that threshold val
        prediction_set_filt_df = pd.DataFrame(prediction_set_df[prediction_set_df["probabilities"] >= threshold])

        # and save that out to text
        prediction_set_filt_df.to_csv("{0}/by_prediction/{1}.prediction_set{2}.txt".format(out_dir, prefix, prediction_set_idx), sep='\t')
        
        # and convert to BED and save out
        task_df = prediction_set_filt_df
        task_df['region'] = task_df.index
        task_df['chr'], task_df['start-stop'] = task_df['region'].str.split(':', 1).str
        task_df['start'], task_df['stop'] = task_df['start-stop'].str.split('-', 1).str
        task_df['joint'] = task_df['labels'].map(str) + ";" + task_df['probabilities'].map(str) + ";" + task_df['logits'].map(str)

        out_bed_file = "{0}/by_prediction/{1}.prediction_set{2}.bed".format(
            out_dir, prefix, prediction_set_idx)
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

    assert ((args.model_type == "nn") or
            (args.model_type == "motif") or
            (args.model_type == "grammar"))

    # find data files
    # TODO(dk) set up data files here
    if args.input_bed is not None:
        assert (args.input_bed is not None) and (len(args.input_labels) > 0)
        data_processing_dir = "{}/data".format(args.out_dir)
        os.system("mkdir -p {}".format(data_processing_dir))
        generate_ordered_single_file_nn_dataset(
            args.input_bed,
            args.annotations["ref_fasta"],
            args.input_labels,
            data_processing_dir,
            args.prefix)
        data_files = glob.glob("{}/h5/*h5".format(data_processing_dir))
    else:
        data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))
        
    # set up model params
    model_fn, model_params = setup_model(args)

    # set up network graph and outputs
    if args.reconstruct_regions:
        shuffle_data = False
    elif args.variants:
        shuffle_data = False
    else:
        shuffle_data = True
        
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        args.tasks,
        load_data_from_filename_list,
        args.batch_size,
        model_fn,
        model_params,
        tf.nn.sigmoid,
        importances_fn=layerwise_relevance_propagation,
        shuffle_data=shuffle_data)
        
    # predict
    if not args.variants:
        labels, logits, probs, metadata, all_metadata = predict(
            tronn_graph,
            args.model_dir,
            args.batch_size,
            num_evals=args.num_evals,
            reconstruct_regions=args.reconstruct_regions)
    else:
        labels, logits, probs, metadata, all_metadata, all_importance_scores = predict_variant_scores(
            tronn_graph,
            args.model_dir,
            args.batch_size,
            num_evals=args.num_evals,
            reconstruct_regions=args.reconstruct_regions)

    # save out labels and logits
    # TODO(dk) formalize this to save
    labels_df = pd.DataFrame(data=labels, index=metadata)
    labels_df.to_csv("{}/labels.test.txt".format(args.out_dir), sep='\t', header=False)

    if args.variants:
        probs_df = pd.DataFrame(data=logits, index=all_metadata) # change from logits to probs?
        probs_df.to_csv("{}/probs.test.txt".format(args.out_dir), sep='\t', header=False)
        importances_df = pd.DataFrame(data=all_importance_scores, index=all_metadata) # change from logits to probs?
        importances_df.to_csv("{}/importances.test.txt".format(args.out_dir), sep='\t', header=False)
    else:
        probs_df = pd.DataFrame(data=probs, index=metadata)
        probs_df.to_csv("{}/probs.test.txt".format(args.out_dir), sep='\t', header=False)
    
    # push predictions through activation to get probs
    if args.model_type != "nn":
        probs = scores_to_probs(logits)

    if args.single_task is None:
        assert labels.shape[1] == logits.shape[1]
        
        # prediction column matches label column
        split_by_label(
            metadata,
            labels,
            logits,
            probs,
            args.out_dir,
            args.prefix)
    else:
        # take single task and split into probs
        split_single_label_set_by_prediction(
            metadata,
            labels[:,args.single_task],
            logits,
            probs,
            args.out_dir,
            args.prefix)

    return None
