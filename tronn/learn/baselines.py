"""Contains code to run baseline models as comparisons
"""

import glob

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.tensor_forest.client import eval_metrics
from tronn.util import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.platform import app

from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.ops import array_ops

from tronn.datalayer import tflearn_input_fn


#from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tronn.util import head as head_lib


def compatible_log_loss(predictions, labels):
    """Wrapper for new log loss to switch order of predictions and labels
    per warning in tensorflow (this needs to change if RF is updated)
    """
    return tf.losses.log_loss(labels, predictions)


class ForestHParamsMulticlass(tensor_forest.ForestHParams):
    """My wrapper around the params to adjust as I need them
    """
    
    def fill(self):
        self = tensor_forest.ForestHParams.fill(self)
        #self.num_outputs = 2
        #self.num_output_columns = 33
        return self


class RandomForestGraphsExtended(tensor_forest.RandomForestGraphs):

    def training_graph(self, input_data, input_labels, num_trainers=1, trainer_id=0, **tree_kwargs):
        test = tensor_forest.RandomForestGraphs.training_graph(self,
                                                               input_data,
                                                               input_labels,
                                                               num_trainers,
                                                               trainer_id,
                                                               **tree_kwargs)
        print test.get_shape()

        return test
        

def get_multi_label_head():
    """
    """
    return head_lib.multi_label_head(2) # for multitask
    
def build_estimator(model_dir):
    """Build an estimator."""
    params = ForestHParamsMulticlass(
        num_classes=3, num_features=1000, # num classes = 2
        num_trees=5000, max_nodes=500) # make this bigger later 500, max nodes 3000
        #loss_fn=compatible_log_loss) # fix these # ADD LOSS FUNCTION??
    graph_builder_class = tensor_forest.RandomForestGraphs
    #graph_builder_class = RandomForestGraphsExtended
    #if FLAGS.use_training_loss:
    if True:
        graph_builder_class = tensor_forest.TrainingLossForest
    return random_forest.TensorForestEstimator(
        params,
        graph_builder_class=graph_builder_class,
        early_stopping_rounds=100000000,
        model_dir=model_dir)#,
        #head=head_lib.multi_label_head(32))

        
    # Use the SKCompat wrapper, which gives us a convenient way to split
    # in-memory data like MNIST into batches.
    #return estimator.SKCompat(random_forest.TensorForestEstimator(
    #    params, graph_builder_class=graph_builder_class,
    #    model_dir=model_dir))


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
        out_dir):
    """Runs random forest baseline model
    Note that this is TFLearn's model
    """

    est = build_estimator(out_dir)
    
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

    est.fit(input_fn=data_loader_train, max_steps=2000) # steps=5000

    results = est.evaluate(input_fn=data_loader_test,
                           metrics=metric,
                           #steps=10000)
                           steps=10)

    for key in sorted(results):
        print('%s: %s' % (key, results[key]))

    return None



def run(args):
    """Run command
    """

    data_files = sorted(glob.glob('{}/*.h5'.format(args.data_dir)))
    train_files = data_files[0:20]
    test_files = data_files[20:22]

    #train_files = [data_files[22]]
    #test_files = [data_files[23]]
    
    # TODO run through at least 1 epoch and at least until loss drops more
    
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval_tensorforest(tflearn_input_fn(train_files, args.batch_size, tasks=[19, 20, 21]),
                                tflearn_input_fn(test_files, args.batch_size, tasks=[19, 20, 21]),
                                args.batch_size,
                                args.out_dir)
    
    return None
