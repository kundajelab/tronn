"""Contains functions to make running tensorflow graphs easier
"""

import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_ops import task_weighted_loss_fn


class TronnGraph(object):
    """Builds out a general purpose TRONN model graph"""

    def __init__(self,
                 data_files,
                 tasks,
                 data_loader,
                 model_fn,
                 model_params,
                 batch_size):
        logging.info("Initialized TronnGraph")
        self.data_files = data_files # data files is a dict of lists
        self.tasks = tasks
        self.data_loader = data_loader
        self.model_fn = model_fn
        self.model_params = model_params
        self.batch_size = batch_size
        
    def build_graph(self, data_key="data", is_training=False):
        """Main function of graph: puts together the pieces
        so that the graph is ready.
        """
        logging.info("Built TronnGraph")
        
        # Set up data loader
        self.features, self.labels, self.metadata = self.data_loader(
            self.data_files[data_key],
            self.batch_size,
            self.tasks)
        
        # model
        out = self.model_fn(self.features, self.labels, self.model_params,
                            is_training=is_training)
        
        return out

    
class TronnNeuralNetGraph(TronnGraph):
    """Builds a training graph"""

    def __init__(self,
                 data_files,
                 tasks,
                 data_loader,
                 model_fn,
                 model_params,
                 final_activation_fn,
                 loss_fn,
                 optimizer_fn,
                 optimizer_params,
                 batch_size,
                 metrics_fn=None,
                 weighted_cross_entropy=False):
        super(TronnNeuralNetGraph, self).__init__(
            data_files, tasks, data_loader, model_fn, model_params, batch_size)
        self.final_activation_fn = final_activation_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.metrics_fn = metrics_fn
        self.weighted_cross_entropy = weighted_cross_entropy

        
    def build_training_graph(self, data_key="train"):
        """Main function: use base class and then add to it
        """
        self.build_graph(data_key, is_training=True)

        # add optimizer and get train op
        self.optimizer = self.optimizer_fn(**self.optimizer_params)
        train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, summarize_gradients=True)

        return train_op

    
    def build_graph(self, data_key="valid", is_training=False):
        """Build a graph for evaluation/prediction/etc
        """
        # base function builds datalayer and model
        self.logits = super(TronnNeuralNetGraph, self).build_graph(
            data_key, is_training=is_training)

        # add a final activation function
        self.probs = self.final_activation_fn(self.logits)
        
        # add a loss
        if not self.weighted_cross_entropy:
            self.loss = self.loss_fn(self.labels, self.logits)
        else:
            self.loss = task_weighted_loss_fn(data_files, loss_fn, labels, logits)
        self.total_loss = tf.losses.get_total_loss()

        if self.metrics_fn is not None:
            self._add_metrics()
        
        return None

    
    def _add_metrics(self):
        """set up metrics function with summaries etc
        """
        # set up metrics and values
        self.metric_values, self.metric_updates = self.metrics_fn(self.labels, self.probs, self.tasks)
        for update in self.metric_updates: tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)

        # Add losses to metrics
        mean_loss, _ = tf.metrics.mean(self.loss, updates_collections=tf.GraphKeys.UPDATE_OPS)
        self.metric_values.update({
            "loss": self.loss,
            "total_loss": self.total_loss,
            "mean_loss": mean_loss
        })

        return

        

    

    
    
