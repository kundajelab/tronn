"""Contains functions to make running tensorflow graphs easier
"""

import logging

import tensorflow as tf

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
        self.data_files = data_files
        self.tasks = tasks
        self.data_loader = data_loader
        self.model_fn = model_fn
        self.model_params = model_params
        self.batch_size = batch_size
        
    def build_graph(self, is_training=False):
        """Main function of graph: puts together the pieces
        so that the graph is ready.
        """
        logging.info("Built TronnGraph")
        
        # Set up data loader
        self.features, self.labels, self.metadata = self.data_loader(self.data_files, self.batch_size, self.tasks)
        
        # model
        out = self.model_fn(features, labels, self.model_params, is_training=is_training)
        
        return out

    
class TronnTrainingGraph(TronnGraph):
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
                 optimizer_params
                 batch_size):
        super(TronnTrainingGraph, self).__init__(data_files, tasks, data_loader, model_fn, model_params, batch_size)
        self.final_activation_fn = final_activation_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params

    def build_graph(self):
        """Main function: use base class and then add to it
        """

        # base function builds datalayer and model
        self.logits = super(TronnTrainingGraph, self).build_graph(is_training=True)

        # add a final activation function
        self.probs = self.final_activation_fn(logits)
        
        # add a loss
        if not weighted_cross_entropy:
            self.loss = loss_fn(labels, logits)
        else:
            self.loss = task_weighted_loss_fn(data_files, loss_fn, labels, logits)
        self.total_loss = tf.losses.get_total_loss()

        # add optimizer and get train op
        self.optimizer = optimizer_fn(**optimizer_params)
        train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

        return train_op
        

    

    
    
