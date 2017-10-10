"""Contains code to run baseline models as comparisons
"""

import glob
import numpy as np

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.tensor_forest.client import eval_metrics
#from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.platform import app

from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.ops import array_ops

from tronn.datalayer import tflearn_input_fn

from tensorflow.contrib.learn.python.learn.estimators import head as head_lib

from tronn.learn import tensor_forest
from tronn.learn import random_forest
#from tensorflow.contrib.tensor_forest.python import tensor_forest
#from tensorflow.contrib.tensor_forest.client import random_forest

def build_estimator(model_dir, num_classes=3):
    """Build an estimator."""
    params = tensor_forest.ForestHParams(
        num_classes=num_classes, num_features=15625, # num classes = 2
        num_trees=100, max_nodes=10000, regression=True) # make this bigger later 500, max nodes 3000
    graph_builder_class = tensor_forest.RandomForestGraphs
    
    if num_classes > 2:
        print "using multi label head"
        head = head_lib.multi_label_head(num_classes) # reactivate this for multi label learning
    else:
        head = None

    return random_forest.TensorForestEstimator(
        params,
        graph_builder_class=graph_builder_class,
        early_stopping_rounds=100000000,
        head=head,
        model_dir=model_dir)


def auprc_old(probs, targets, weights=None):
    return metric_ops.streaming_auc(array_ops.slice(probs, [0, 1], [-1, 1]),
                                    targets, weights, curve='PR')
    #return metric_ops.streaming_auc(probs, targets, weights, curve='PR')

def auprc(probs, targets, weights=None):
    return tf.metrics.auc(targets, probs, curve='PR')

    
def auroc_old(probs, targets, weights=None):
    return metric_ops.streaming_auc(array_ops.slice(probs, [0, 1], [-1, 1]),
                                    targets, weights, curve='ROC')
    #return tf.metrics.auc(targets, probs, curve='ROC')

def auroc(probs, targets, weights=None):
    return tf.metrics.auc(targets, probs, curve='ROC')


def accuracy(probs, targets):
    print probs.get_shape()
    print targets.get_shape()
    
    predictions = tf.cast(tf.greater(tf.cast(probs, 'float32'), tf.cast(0.5, 'float32')), 'float32')
    targets_expanded = tf.expand_dims(targets, 2)
    predictions_expanded = tf.expand_dims(predictions, 2)
    return tf.metrics.accuracy(targets_expanded, predictions_expanded)


def train_and_eval_tensorforest(
        data_loader_train,
        data_loader_test,
        batch_size,
        #tasks,
        out_dir,
        num_classes=3):
    """Runs random forest baseline model
    Note that this is TFLearn's model
    """

    est = build_estimator(out_dir, num_classes=num_classes)

    # TODO change metrics here, here for debugging, move back down later
    metric = {}
    metric['accuracy'] = metric_spec.MetricSpec(
        accuracy,
        prediction_key='logits') 
    metric['auroc_tf'] = metric_spec.MetricSpec(
        auroc,
        prediction_key='logits')
    metric['auprc_tf'] = metric_spec.MetricSpec(
        auprc,
        prediction_key='logits')

    # code to check the key names
    from tensorflow.contrib.learn.python.learn.estimators import prediction_key
    print prediction_key.PredictionKey.PROBABILITIES
    print prediction_key.PredictionKey.CLASSES # TODO need to fix this?
    #variable_names = est.get_variable_names()

    est.fit(input_fn=data_loader_train, max_steps=5000) # steps=5000

    if True:
        results = est.evaluate(input_fn=data_loader_test,
                               metrics=metric,
                               steps=10)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


    import ipdb
    ipdb.set_trace()

    predict_total = 50
    prediction_generator = est.predict(input_fn=data_loader_test, outputs=["probabilities", "logits", "classes", "labels"])
    #np.zeros((predict_total, num_classes))
    
    
    for i in xrange(predict_total):
        blah = prediction_generator.next()
        print blah

    
    return None



def run(args):
    """Run command
    """

    data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))
    #train_files = data_files
    #test_files = data_files
    train_files = data_files[0:20]
    test_files = data_files[20:22]
    
    # TODO run through at least 1 epoch and at least until loss drops more
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval_tensorforest(tflearn_input_fn(train_files, args.batch_size, tasks=args.tasks),
                                tflearn_input_fn(test_files, args.batch_size, tasks=args.tasks),
                                args.batch_size,
                                args.out_dir,
                                num_classes=3 if len(args.tasks) == 0 else 2)
    
    return None




