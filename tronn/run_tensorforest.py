"""Contains code to run baseline models as comparisons
"""

import glob
import numpy as np

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
#from tensorflow.contrib.tensor_forest.python import tensor_forest
#from tensorflow.contrib.tensor_forest.client import random_forest

from tronn.datalayer import tflearn_input_fn
from tronn.learn.tensor_forest import tensor_forest
from tronn.learn.tensor_forest import random_forest
from tronn.learn.cross_validation import setup_cv

from tronn.nets.kmer_nets import featurize_kmers
from tronn.nets.motif_nets import featurize_motifs

from tronn.interpretation.motifs import get_encode_pwms


def build_tensorforest_estimator(
        model_dir,
        num_features,
        num_classes,
        num_trees=100,
        max_nodes=10000,
        early_stopping_rounds=300):
    """Build an estimator for tensorforest
    
    The num_trees and max_nodes are from Google recommendations
    Note that I increased early stopping rounds because biology
    data is super noisy. These presets seem pretty good - even
    if I run the model longer, the numbers stabilize out pretty
    quickly (open question of the fact that the RF is not seeing
    or really able to use all the data at the moment...)
    """
    params = tensor_forest.ForestHParams(
        num_classes=num_classes,
        num_features=num_features,
        num_trees=num_trees,
        max_nodes=max_nodes,
        regression=True if num_classes > 2 else False)

    # set up head if multi output
    if num_classes > 2:
        print "using multi label head"
        head = head_lib.multi_label_head(num_classes) # reactivate this for multi label learning
    else:
        head = None

    return random_forest.TensorForestEstimator(
        params,
        graph_builder_class=tensor_forest.RandomForestGraphs,
        early_stopping_rounds=early_stopping_rounds,
        head=head,
        model_dir=model_dir)


def train_and_evaluate_tensorforest(
        train_input_fn,
        test_input_fn,
        num_features,
        batch_size,
        out_dir,
        num_classes,
        eval_prefix,
        num_evals=1000):
    """Runs random forest baseline model
    Note that this is TFLearn's model
    """
    # build estimator
    estimator = build_tensorforest_estimator(out_dir, num_features, num_classes)

    # fit
    estimator.fit(input_fn=train_input_fn) # steps=5000

    # evaluation
    results = estimator.evaluate(
        input_fn=test_input_fn,
        steps=num_evals)
    
    eval_file = "{}.tflearn.eval.txt".format(eval_prefix)
    with open(eval_file, 'w') as out:
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))
            out.write("{}: {}\n".format(key, results[key]))

    # TODO(dk) are predictions needed?
    if False:
        predict_total = 50
        prediction_generator = estimator.predict(
            input_fn=data_loader_test)
    
        for i in xrange(predict_total):
            blah = prediction_generator.next()
            print blah
    
    return None


def run(args):
    """Run command
    """
    data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))
    train_files, valid_files, test_files = setup_cv(data_files, cvfold=args.cvfold)
    tf.logging.set_verbosity(tf.logging.INFO)

    # set up featurization layers
    if args.kmers:
        featurize_fn = featurize_kmers
        featurize_params = {"kmer_len": args.kmer_len}
        num_features = 5**args.kmer_len
    elif args.motifs:
        featurize_fn = featurize_motifs
        pwm_list = get_encode_pwms(args.pwm_file)
        featurize_params = {"pwm_list": pwm_list}
        num_features = len(pwm_list)

    # set up input functions
    train_input_fn = tflearn_input_fn(
        train_files,
        args.batch_size,
        tasks=args.tasks,
        featurize_fn=featurize_fn,
        featurize_params=featurize_params)
    test_input_fn = tflearn_input_fn(
        test_files,
        args.batch_size,
        tasks=args.tasks,
        featurize_fn=featurize_fn,
        featurize_params=featurize_params)
            
    # train and evaluate
    train_and_evaluate_tensorforest(
        train_input_fn,
        test_input_fn,
        num_features,
        args.batch_size,
        args.out_dir,
        args.num_classes,
        "{}/{}".format(args.out_dir, args.prefix))
    
    return None




