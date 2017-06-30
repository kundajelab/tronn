"""Contains code for analyzing grammars
"""

import os
import json
import glob
import gzip

import numpy as np
import pandas as pd
import tensorflow as tf

from tronn.preprocess import generate_nn_dataset
from tronn.datalayer import load_data_from_filename_list, get_total_num_examples
from tronn.models import grammar_scanner
from tronn.interpretation.motifs import PWM

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from scipy.stats import zscore


class Grammar(object):
    def __init__(self, pwm_list, name):
        self.pwms = pwm_list
        self.name = name

        # consider adding: threshold, distance constraints

        
def setup_tensorflow_session():
    """Start up session in a graph
    """

    # set up session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # start queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    
    return sess, coord, threads


def close_tensorflow_session(coord, threads):
    """Cleanly close a running graph
    """

    coord.request_stop()
    coord.join(threads)

    
    return None


def scan_grammars(data_files, motif_file, grammars, prefix, out_dir, batch_size, total_region_num):
    """Scans given regions for grammars and returns a matrix of hits
    """

    total_example_num = get_total_num_examples(data_files)
    num_grammars = len(grammars.keys())
    print 'scanning {} examples'.format(total_example_num)
    
    # create grammar graph
    with tf.Graph().as_default() as g:

        # data loader
        features, labels_tensor, metadata = load_data_from_filename_list(data_files,
                                                                         batch_size,
                                                                         [],
                                                                         shuffle=False)

        # model
        prediction_tensor = grammar_scanner(features, grammars)
        
        # setup
        sess, coord, threads = setup_tensorflow_session()

        current_example_num = 0
        current_chr, current_start, current_stop = '', 0, 0
        current_label = np.zeros(labels_tensor.get_shape()[1])
        current_predictions = np.zeros((num_grammars,))
        current_num_joined_examples = 0
        regions_list = []
        motif_results = np.zeros((total_region_num, num_grammars))
        all_labels = np.zeros((total_region_num,labels_tensor.get_shape()[1]))
        current_region_idx = 0
        #while current_example_num < total_example_num:
        while current_region_idx < total_region_num: # TODO fix this
        
            # run batches, as running batches, output average over a region
            predictions, labels, regions = sess.run([prediction_tensor, labels_tensor, metadata])
            
            for example_idx in range(regions.shape[0]):
                region = regions[example_idx,0]
                prediction = predictions[example_idx,:]
                label = labels[example_idx,:]

                region_chr = region.split(':')[0]
                region_start = int(region.split(':')[1].split('-')[0])
                region_stop = int(region.split('-')[1].split('(')[0])

                # only join if same chromosome and start is less than current end
                # otherwise, start a new region
                if region_chr == current_chr:
                    if region_start < current_stop:
                        # join to region
                        current_stop = region_stop
                        # TODO consider different ways to add to prediction
                        current_predictions += prediction
                        #current_predictions = np.maximum(current_predictions, prediction)
                        
                        current_num_joined_examples += 1
                        current_label += label
                        continue

                if current_chr != '':
                    regions_list.append('{0}:{1}-{2}'.format(current_chr, current_start, current_stop))
                    motif_results[current_region_idx,:] = current_predictions / current_num_joined_examples
                    #motif_results[current_region_idx,:] = current_predictions 
                    all_labels[current_region_idx,:] = (current_label > 0).astype(int)
                    current_region_idx += 1

                    if current_region_idx % 1000 == 0:
                        print current_region_idx
                    
                    if current_region_idx == total_region_num:
                        break
                    
                # start a new region
                current_chr, current_start, current_stop = region_chr, region_start, region_stop
                current_predictions = prediction
                current_label = label
                current_num_joined_examples = 1

        # close session
        close_tensorflow_session(coord, threads)

    # Make a pandas dataframe and save this info out
    # TODO also save out labels
    grammar_hits = pd.DataFrame(motif_results[0:total_region_num,:],
                                index=regions_list[0:total_region_num],
                                columns=sorted(grammars.keys()))

    grammar_hits.to_csv('task_0.testing_grammar_hits.txt', sep='\t')


    # To think about: right now taking the max prob from independent grammars
    # so the key is how motifs in each group are reordering ranking
    # so then no changes should happen here (outside of normalizing) since
    # at this stage there is no reordering. all changes need to happen at the
    # grammar module level. ie, relative weights of motifs in groups
    
    if False:
        #grammar_mean_vals = grammar_hits.mean(axis=1)
        #grammar_std_vals = grammar_hits.std(axis=1)
        #grammar_zscores = (grammar_hits.as_matrix() - grammar_mean_vals) / grammar_std_vals
        grammar_zscores = zscore(grammar_hits.as_matrix())
        grammar_activation = 1 / (1 + np.exp(-grammar_zscores))
        grammar_prediction_probs = np.max(grammar_activation, axis=1)
    else:
        # here set up predictions vector and compare to labels
        grammar_max_vals = grammar_hits.max(axis=1)
        grammar_min_vals = grammar_hits.min(axis=1)
        grammar_predictions = grammar_hits.max(axis=1) # need to convert this to probabilities...
        grammar_prediction_probs = (grammar_predictions - grammar_predictions.min()) / grammar_predictions.max()
        
    for label_idx in range(all_labels.shape[1]):
        # get PR val
        precision, recall = precision_recall_curve(all_labels[:,label_idx], grammar_prediction_probs)[:2]
        print auc(recall, precision)
        # get AUROC val
        print roc_auc_score(all_labels[:,label_idx], grammar_prediction_probs)


    quit()

    
    return grammar_hits



def run(args):

    print "not implemented yet"

    with open(args.annotations_json, 'r') as fp:
        annotation_files = json.load(fp)

    num_regions = 0
    with gzip.open(args.regions) as fp:
        for line in fp:
            num_regions += 1

    print num_regions
        
        
    # first set up grammar dataset
    if not os.path.isdir('{}/data'.format(args.scratch_dir)):
        data_dir = generate_nn_dataset(args.regions,
                                       annotation_files['univ_dhs'],
                                       annotation_files['ref_fasta'],
                                       [args.labels],
                                       '{}/data'.format(args.scratch_dir),
                                       args.prefix,
                                       bin_method='naive',
                                       parallel=args.parallel,
                                       neg_region_num=args.univ_neg_num)

    data_files = glob.glob('{}/data/h5/*.h5'.format(args.scratch_dir))

    # load up grammar objects
    pwm_list = PWM.get_encode_pwms(args.motifs)
    pwm_dict = {}
    for pwm in pwm_list:
        pwm_dict[pwm.name] = pwm

    grammar_dict = {}
    with open(args.grammars, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')
            grammar_name = 'grammar_{}'.format(fields[0])
            pwm_names = fields[1:]

            grammar_pwms = [pwm_dict[pwm_name] for pwm_name in pwm_names]
            grammar_dict[grammar_name] = Grammar(grammar_pwms, grammar_name)
    
    # scan grammars
    predictions = scan_grammars(data_files,
                                args.motifs,
                                grammar_dict,
                                args.prefix,
                                args.out_dir,
                                args.batch_size,
                                num_regions)

    # evaluate grammars using scikit learn tools
    
    # and then visualize



    
    

    return
    
