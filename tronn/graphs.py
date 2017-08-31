"""Contains functions to make running tensorflow graphs easier
"""

import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_ops import class_weighted_loss_fn
from tronn.util.tf_ops import positives_focused_loss_fn

from tronn.datalayer import get_task_and_class_weights
from tronn.datalayer import get_positive_weights_per_task


class TronnGraph(object):
    """Builds out a general purpose TRONN model graph"""

    def __init__(self,
                 data_files,
                 tasks,
                 data_loader,
                 model_fn,
                 model_params,
                 batch_size,
                 feature_key="features",
                 shuffle_data=True):
        logging.info("Initialized TronnGraph")
        self.data_files = data_files # data files is a dict of lists
        self.tasks = tasks
        self.data_loader = data_loader
        self.model_fn = model_fn
        self.model_params = model_params
        self.batch_size = batch_size
        self.feature_key = feature_key
        self.shuffle_data = shuffle_data
        
    def build_graph(self, data_key="data", is_training=False):
        """Main function of graph: puts together the pieces
        so that the graph is ready.
        """
        logging.info("Built TronnGraph")
        
        # Set up data loader
        self.features, self.labels, self.metadata = self.data_loader(
            self.data_files[data_key],
            self.batch_size,
            self.tasks,
            features_key=self.feature_key,
            shuffle=self.shuffle_data)

        # adjust tasks
        if self.tasks == []:
            self.tasks = range(self.labels.get_shape()[1])
        
        # model
        out = self.model_fn(self.features, self.labels, self.model_params,
                            is_training=is_training)
        
        return out

    
class TronnNeuralNetGraph(TronnGraph):
    """Builds a trainable graph"""

    def __init__(self,
                 data_files,
                 tasks,
                 data_loader,
                 batch_size,
                 model_fn,
                 model_params,
                 final_activation_fn,
                 loss_fn=None,
                 optimizer_fn=None,
                 optimizer_params=None,
                 metrics_fn=None,
                 importances_fn=None,
                 feature_key="features",
                 shuffle_data=True,
                 class_weighted_loss=False,
                 positives_focused_loss=False,
                 finetune=False,
                 finetune_tasks=[]):
        super(TronnNeuralNetGraph, self).__init__(
            data_files, tasks, data_loader,
            model_fn, model_params, batch_size,
            feature_key=feature_key, shuffle_data=shuffle_data)
        self.final_activation_fn = final_activation_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.metrics_fn = metrics_fn
        self.importances_fn = importances_fn
        self.class_weighted_loss = class_weighted_loss
        self.positives_focused_loss = positives_focused_loss

        
    def build_graph(self, data_key="data", is_training=False):
        """Build a graph for evaluation/prediction/etc
        """
        # base function builds datalayer and model
        self.logits = super(TronnNeuralNetGraph, self).build_graph(
            data_key, is_training=is_training)
        
        # add a final activation function on the logits
        self.probs = self.final_activation_fn(self.logits)
        
        return self.labels, self.logits, self.probs


    def build_evaluation_graph(self, data_key="valid"):
        """Build a graph with metrics functions for evaluation
        """
        assert self.metrics_fn is not None, "Need metrics to evaluate"
        
        self.build_graph(data_key, is_training=False)
        
        # add a loss
        if self.loss_fn is not None:
            self._add_loss(data_key)

        # add metrics
        if self.metrics_fn is not None:
            self._add_metrics()

        return None
        
        
    def build_training_graph(self, data_key="train"):
        """Main function: use base class and then add to it
        """
        assert self.loss_fn is not None, "Need loss to train on"
        assert self.optimizer_fn is not None, "Need an optimizer for training"
        
        self.build_graph(data_key, is_training=True)

        # add a loss
        self._add_loss(data_key)

        # add metrics
        if self.metrics_fn is not None:
            self._add_metrics()
        
        # add optimizer and get train op
        if self.finetune:
            variables_to_train = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "logits")
        else:
            variables_to_train = None # slim will use all trainable in graphkeys
                        
        self.optimizer = self.optimizer_fn(**self.optimizer_params)
        train_op = slim.learning.create_train_op(
            self.total_loss,
            self.optimizer,
            variables_to_train=variables_to_train,
            summarize_gradients=True)

        return train_op

    
    def build_inference_graph(self, data_key="data"):
        """Build a graph with back prop ties to be able to get 
        importance scores
        """
        assert self.importances_fn is not None
        
        self.build_graph(data_key, is_training=False)

        # split logits into task level
        task_logits = tf.unstack(self.logits, axis=1)
        
        # add in importance score calculations
        self.importances = {}
        for task_idx in range(len(self.tasks)):
            importance_key = "importances_task{}".format(self.tasks[task_idx])
            self.importances[importance_key] = self.importances_fn(
                task_logits[task_idx], self.features)

        return self.importances


    def _add_loss(self, data_key):
        """set up loss function
        """
        assert not (self.class_weighted_loss and self.positives_focused_loss)

        if self.finetune:
            # adjust which labels and logits go into loss if finetuning
            labels_unstacked = tf.unstack(self.labels, axis=1)
            labels = tf.stack([labels_unstacked[i] for i in self.finetune_tasks], axis=1)
            logits_unstacked = tf.unstack(self.logits, axis=1)
            logits = tf.stack([logits_unstacked[i] for i in self.finetune_tasks], axis=1)
            print labels.get_shape()
            print logits.get_shape()
        else:
            labels = self.labels
            logits = self.logits

        # split out getting the positive weights so that only the right ones go into the loss function
            
        if self.class_weighted_loss:
            pos_weights = get_positive_weights_per_task(self.data_files[data_key])
            if self.finetune:
                pos_weights = [pos_weights[i] for i in self.finetune_tasks]
            self.loss = class_weighted_loss_fn(
                self.loss_fn, labels, logits, pos_weights)
        elif self.positives_focused_loss:
            task_weights, class_weights = get_task_and_class_weights(self.data_files[data_key])
            if self.finetune:
                task_weights = [task_weights[i] for i in self.finetune_tasks]
            if self.finetune:
                class_weights = [class_weights[i] for i in self.finetune_tasks]
            self.loss = positives_focused_loss_fn(
                self.loss_fn, labels, logits, task_weights, class_weights)
        else:
            self.loss = self.loss_fn(labels, logits)

        self.total_loss = tf.losses.get_total_loss()

        return
    
    
    def _add_metrics(self):
        """set up metrics function with summaries etc
        """
        # set up metrics and values
        self.metric_values, self.metric_updates = self.metrics_fn(
            self.labels, self.probs, self.tasks)
        for update in self.metric_updates: tf.add_to_collection(
                tf.GraphKeys.UPDATE_OPS, update)

        # Add losses to metrics
        mean_loss, _ = tf.metrics.mean(
            self.loss, updates_collections=tf.GraphKeys.UPDATE_OPS)
        self.metric_values.update({
            "loss": self.loss,
            "total_loss": self.total_loss,
            "mean_loss": mean_loss
        })

        return

        

    

    
    
