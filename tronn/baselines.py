"""Contains code to run baseline models as comparisons
"""

import glob

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.platform import app

from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.ops import array_ops

from tronn.datalayer import tflearn_input_fn



def compatible_log_loss(predictions, labels):
    """Wrapper for new log loss to switch order of predictions and labels
    per warning in tensorflow (this needs to change if RF is updated)
    """
    return tf.losses.log_loss(labels, predictions)


def build_estimator(model_dir):
    """Build an estimator."""
    params = tensor_forest.ForestHParams(
        num_classes=2, num_features=1000,
        num_trees=500) # make this bigger later
        #loss_fn=compatible_log_loss) # fix these # ADD LOSS FUNCTION??
    graph_builder_class = tensor_forest.RandomForestGraphs
    #if FLAGS.use_training_loss:
    if False:
        graph_builder_class = tensor_forest.TrainingLossForest
    return random_forest.TensorForestEstimator(
        params, graph_builder_class=graph_builder_class,
        model_dir=model_dir)

        
    # Use the SKCompat wrapper, which gives us a convenient way to split
    # in-memory data like MNIST into batches.
    #return estimator.SKCompat(random_forest.TensorForestEstimator(
    #    params, graph_builder_class=graph_builder_class,
    #    model_dir=model_dir))


def auprc(probs, targets, weights=None):
    return metric_ops.streaming_auc(array_ops.slice(probs, [0, 1], [-1, 1]),
                                    targets, weights, curve='PR')
    #return metric_ops.streaming_auc(probs, targets, weights, curve='PR')

def auroc(probs, targets, weights=None):
    return metric_ops.streaming_auc(array_ops.slice(probs, [0, 1], [-1, 1]),
                                    targets, weights, curve='ROC')

    #return tf.metrics.auc(targets, probs, curve='ROC')

    
def train_and_eval_tensorforest(
        data_loader_train,
        data_loader_test,
        batch_size,
        #tasks,
        out_dir):
    """Runs random forest baseline model
    Note that this is TFLearn's model
    """

    # TODO change metrics here, here for debugging, move back down later
    metric_name = 'accuracy'
    metric = {metric_name:
              metric_spec.MetricSpec(
                  eval_metrics.get_metric(metric_name),
                  prediction_key=eval_metrics.get_prediction_key(metric_name))}
    metric['auroc_tf'] = metric_spec.MetricSpec(
        auroc,
        prediction_key=eval_metrics.get_prediction_key('sigmoid_entropy'))
    metric['auprc_tf'] = metric_spec.MetricSpec(
        auprc,
        prediction_key=eval_metrics.get_prediction_key('sigmoid_entropy'))

    
    #metric['auroc'] = metric_spec.MetricSpec(
    #    auroc, prediction_key=eval_metrics.get_prediction_key('accuracy'))
    #metric['auprc'] = metric_spec.MetricSpec(
    #    auprc, prediction_key=eval_metrics.get_prediction_key('accuracy'))


    est = build_estimator(out_dir)
    
    #est.fit(x=features, y=labels,
    #        batch_size=batch_size)

    est.fit(input_fn=data_loader_train, steps=5000)
    #for i in range(10):
    #    print i
    #    est.partial_fit(input_fn=data_loader_train, steps=1000)
    


    #metric['predictions'] = metric_spec.MetricSpec(
    #    eval_metrics.get_metric('predictions'),
    #    prediction_key=eval_metrics.get_prediction_key('predictions'))

    # get the results
    #results = est.score(input_fn=data_loader_test, 
    #                    batch_size=batch_size,
    #                    metrics=metric)
    
    results = est.evaluate(input_fn=data_loader_test,
                           metrics=metric,
                           #steps=10000)
                           steps=1)

    for key in sorted(results):
        print('%s: %s' % (key, results[key]))

    return None



def run(args):
    """Run command
    """

    data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))
    train_files = data_files[0:20]
    test_files = data_files[20:22]

    # TODO run through at least 1 epoch and at least until loss drops more
    
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval_tensorforest(tflearn_input_fn(train_files, args.batch_size),
                                tflearn_input_fn(test_files, args.batch_size),
                                args.batch_size,
                                args.out_dir)
    
    return None
