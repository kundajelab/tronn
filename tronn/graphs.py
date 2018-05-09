"""Contains functions to make running tensorflow graphs easier
"""

import os
import logging
import h5py

import six

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables

from tensorflow.python.training import monitored_session
from tensorflow.python.training import training

from tronn.util.tf_ops import restore_variables_op
from tronn.util.tf_ops import class_weighted_loss_fn
from tronn.util.tf_ops import positives_focused_loss_fn

#from tronn.datalayer import get_task_and_class_weights
#from tronn.datalayer import get_positive_weights_per_task

from tronn.outlayer import H5Handler
from tronn.learn.evaluation import get_global_avg_metrics

from tronn.learn.learning_2 import RestoreHook


class TronnGraph(object):
    """Builds out a general purpose TRONN model graph"""

    def __init__(self,
                 data_files, # TODO add an arg for data params
                 tasks,
                 data_loader,
                 model_fn,
                 model_params,
                 batch_size,
                 feature_key="features",
                 shuffle_data=True,
                 fake_task_num=0,
                 filter_tasks=[],
                 ordered_num_epochs=1,
                 checkpoints=[]): # changed for prediction...
        logging.info("Initialized TronnGraph")
        self.data_files = data_files # data files is a dict of lists
        self.tasks = tasks
        self.data_loader = data_loader
        self.model_fn = model_fn
        self.model_params = model_params
        self.batch_size = batch_size
        self.feature_key = feature_key
        self.shuffle_data = shuffle_data
        self.fake_task_num = fake_task_num
        self.filter_tasks = filter_tasks
        self.ordered_num_epochs = ordered_num_epochs
        self.checkpoints = checkpoints

    # eventually:
    # def build_graph(self, inputs, params)
    def build_graph(self, data_key="data", is_training=False):
        """Main function of graph: puts together the pieces
        so that the graph is ready.
        """
        logging.info("Built TronnGraph")
        # TODO this bit should be: add dataloader, add model, add inference stack (if needed)
        
        # Set up data loader
        #if False:
        #    self.features, self.labels, self.metadata = self.data_loader(
        #        self.data_files[data_key],
        #        self.batch_size,
        #        task_indices=self.tasks,
        #        features_key=self.feature_key,
        #        shuffle=self.shuffle_data,
        #        ordered_num_epochs=self.ordered_num_epochs,
        #        fake_task_num=self.fake_task_num,
        #        filter_tasks=self.filter_tasks)

        inputs = self.data_loader.build_dataflow(self.batch_size, data_key)
        self.features = inputs["features"]
        self.labels = inputs["labels"]
        self.metadata = inputs["example_metadata"]
        
        # adjust tasks
        if self.tasks == []:
            self.tasks = range(self.labels.get_shape()[1])
        
        # model
        # TODO conver this to all be
        # outputs = self.model_fn(inputs, params)
        out = self.model_fn(self.features, self.labels, self.model_params,
                            is_training=is_training)
        
        return out


    def build_restore_graph_function(self, is_ensemble=False, skip=[], scope_change=None):
        """build the restore function
        """
        if is_ensemble: # this is really determined by there being more than 1 ckpt - can use as test?
            def restore_function(sess):
                # TODO adjust this function to be like below
                # for ensemble, just need to adjust scoping
                for i in xrange(len(self.checkpoints)):
                    new_scope = "model_{}/".format(i)
                    print new_scope
                    init_assign_op, init_feed_dict = restore_variables_op(
                        self.checkpoints[i],
                        skip=skip,
                        include_scope=new_scope,
                        scope_change=["", new_scope])
                    sess.run(init_assign_op, init_feed_dict)
        else:
            print self.checkpoints
            if len(self.checkpoints) > 0:
                init_assign_op, init_feed_dict = restore_variables_op(
                    self.checkpoints[0], skip=skip, scope_change=scope_change)
                def restore_function(sess):
                    sess.run(init_assign_op, init_feed_dict)
            else:
                print "WARNING NO CHECKPOINTS USED"
                
        return restore_function

    
    def restore_graph(self, sess, is_ensemble=False, skip=[], scope_change=None):
        """restore saved model from checkpoint into sess
        """
        restore_function = self.build_restore_graph_function(
            is_ensemble=is_ensemble, skip=skip, scope_change=scope_change)
        restore_function(sess)
        
        return None

    
    
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
                 inference_fn=None,
                 importances_tasks=None,
                 feature_key="features",
                 shuffle_data=True,
                 fake_task_num=0,
                 filter_tasks=[],
                 class_weighted_loss=False,
                 positives_focused_loss=False,
                 finetune=False,
                 finetune_tasks=[],
                 ordered_num_epochs=1, **kwargs): # 1 for interpretation, 100 for viz
        super(TronnNeuralNetGraph, self).__init__(
            data_files, tasks, data_loader,
            model_fn, model_params, batch_size,
            feature_key=feature_key, shuffle_data=shuffle_data,
            fake_task_num=fake_task_num,
            filter_tasks=filter_tasks,
            ordered_num_epochs=ordered_num_epochs, **kwargs)
        self.final_activation_fn = final_activation_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.metrics_fn = metrics_fn
        self.inference_fn = inference_fn
        self.importances_tasks = importances_tasks
        self.class_weighted_loss = class_weighted_loss
        self.positives_focused_loss = positives_focused_loss
        self.finetune = finetune
        self.finetune_tasks = finetune_tasks

        
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

    
    def build_inference_graph(
            self,
            inference_params,
            scan_grammars=False,
            validate_grammars=False,
            data_key="data"):
        """Build a graph with back prop ties to be able to get 
        importance scores
        """
        assert self.inference_fn is not None

        # build graph TODO if ever adjusting gradients for guided backprop, it goes here
        self.build_graph(data_key, is_training=False)

        # set up config
        if self.importances_tasks is None:
            self.importances_tasks = self.tasks if len(self.tasks) != 0 else [0]
            
        # set up negatives call (but change this - just filter through queues)
        importance_labels = []
        labels_list = tf.unstack(self.labels, axis=1)
        for task_idx in self.importances_tasks:
            importance_labels.append(labels_list[task_idx])
        importance_labels = tf.stack(importance_labels, axis=1)
        
        if len(self.labels.get_shape().as_list()) > 1:
            negative = tf.cast(
                tf.logical_not(
                    tf.cast(tf.reduce_sum(importance_labels, 1, keep_dims=True),
                            tf.bool)), tf.int32)
        else:
            negative = tf.logical_not(tf.cast(self.labels, tf.bool))

        # NOTE: if filtering, need to pass through everything associated with example that is coming back out
        # as such, everything in config needs to be in batch format to quickly filter things
        config = {
            "model": self.model_fn,
            "batch_size": self.batch_size,
            "pwms": inference_params.get("pwms"),
            "grammars": inference_params.get("grammars"),
            "importance_task_indices": self.importances_tasks,
            "importances_fn": inference_params.get("importances_fn"),
            "keep_onehot_sequence": "onehot_sequence" if True else None, # always used: filtering
            "keep_gradients": "gradients" if inference_params.get("dream") is not None else None,
            "all_grad_ys": inference_params.get("dream_pattern"),
            "keep_importances": "importances" if validate_grammars else None,
            "keep_pwm_scores_full": "pwm-scores-full" if scan_grammars else None, # used for grammars
            "keep_global_pwm_scores": "global-pwm-scores" if validate_grammars else None,
            "keep_pwm_scores": "pwm-scores" if True else None, # always used
            "keep_pwm_raw_scores": "pwm-scores-raw" if True else None,
            "keep_grammar_scores": "grammar-scores" if True else None, # always used
            "keep_grammar_scores_full": "grammar-scores-full" if True else None, # always used
            "keep_ism_scores": "ism-scores" if scan_grammars else None, # adjust this later
            "keep_dmim_scores": "dmim-scores" if scan_grammars else None, # adjust this later
            "outputs": { # these are all the batch results that must stay with their corresponding example
                "logits": self.logits,
                "importance_logits": self.logits,
                "probs": self.probs,
                "example_metadata": self.metadata,
                "subset_accuracy": self._add_task_subset_accuracy(),
                "negative": negative
            }
        }

        # don't run importances if empty net
        if self.model_params["name"] == "empty_net":
            config["use_importances"] = False
        
        # set up inference stack
        features, labels, config = self.inference_fn(
            self.features, self.labels, config, is_training=False)

        # grab desired outputs
        outputs = config["outputs"]
        
        return outputs

    
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


    def _add_task_subset_accuracy(self):
        """Given task subset, get subset accuracy
        """
        assert self.importances_tasks is not None

        # split and get subset
        labels_unstacked = tf.unstack(self.labels, axis=1)
        labels_subset = tf.stack([labels_unstacked[i] for i in self.importances_tasks], axis=1)

        probs_unstacked = tf.unstack(self.probs, axis=1)
        probs_subset = tf.stack([probs_unstacked[i] for i in self.importances_tasks], axis=1)

        # compare labels to predictions
        correctly_predicted = tf.logical_not(tf.logical_xor(tf.cast(labels_subset, tf.bool), tf.greater_equal(probs_subset, 0.5)))
        accuracy = tf.reduce_mean(tf.cast(correctly_predicted, tf.float32), 1, keep_dims=True)

        return accuracy
        

    

# long term goal - make it possible to replace the core with a high level estimator (or any other type of model)
class TronnGraphV2(object):
    """Builds out a general purpose TRONN model graph"""

    def __init__(self,
                 data_loader,
                 model_fn,
                 model_params,
                 batch_size,
                 final_activation_fn=tf.nn.sigmoid,
                 loss_fn=tf.losses.sigmoid_cross_entropy,
                 optimizer_fn=tf.train.RMSPropOptimizer,
                 optimizer_params={'learning_rate': 0.002, 'decay': 0.98, 'momentum': 0.0},
                 metrics_fn=None,
                 fake_task_num=0, # does this belong in dataloader?
                 filter_tasks=[], # does this only go in dataloader
                 ordered_num_epochs=1, # does this only go in dataloader?
                 checkpoints=[]): # changed for prediction...
        logging.info("Initialized TronnGraph")
        self.data_loader = data_loader
        self.model_fn = model_fn
        self.model_params = model_params
        self.batch_size = batch_size
        self.final_activation_fn = final_activation_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.metrics_fn = metrics_fn
        self.fake_task_num = fake_task_num # pull this out?
        self.filter_tasks = filter_tasks
        self.ordered_num_epochs = ordered_num_epochs
        self.checkpoints = checkpoints

        # to factor out
        self.class_weighted_loss = False
        self.positives_focused_loss = False
        self.finetune = False
        

    def build_training_dataflow(
            self,
            data_key="train",
            features_key="features",
            labels_key="labels",
            logits_key="logits",
            probs_key="probs"):
        """build a training dataflow
        """
        logging.info("building training dataflow")
        training_params = {}
        
        # dataloader
        inputs = self.data_loader.build_dataflow(self.batch_size, data_key)
        training_params["data_key"] = data_key

        # model
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None
        training_params["features_key"] = features_key
        training_params["labels_key"] = labels_key
        training_params["logits_key"] = logits_key
        training_params["is_training"] = True
        training_params.update(self.model_params)
        outputs, _ = self.model_fn(inputs, training_params)

        # add final activation function
        outputs[probs_key] = self.final_activation_fn(outputs[logits_key])
        
        # add loss
        outputs["loss"] = self._add_loss(
            outputs[labels_key],
            outputs[logits_key],
            data_key=data_key) # TODO remove data key?

        # add metrics
        if self.metrics_fn is not None:
            self._add_metrics(outputs[labels_key], outputs[probs_key])

        # add train op
        training_params["train_op"] = self._add_train_op()
        
        return outputs, training_params


    def build_evaluation_dataflow(
            self,
            data_key="test",
            features_key="features",
            labels_key="labels",
            logits_key="logits",
            probs_key="probs"):
        """just for evaluation
        """
        logging.info("building evaluation dataflow")
        eval_params = {}
        
        # dataloader
        inputs = self.data_loader.build_dataflow(self.batch_size, data_key)
        eval_params["data_key"] = data_key
        
        # model
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None
        eval_params["features_key"] = features_key
        eval_params["labels_key"] = labels_key
        eval_params["logits_key"] = logits_key
        eval_params["is_training"] = False
        eval_params.update(self.model_params)
        outputs, _ = self.model_fn(inputs, eval_params)

        # add final activation function
        outputs[probs_key] = self.final_activation_fn(outputs[logits_key])
        
        # add a loss
        outputs["loss"] = self._add_loss(
            outputs[labels_key],
            outputs[logits_key],
            data_key=data_key)

        # add metrics
        if self.metrics_fn is not None:
            self._add_metrics(outputs[labels_key], outputs[probs_key])

        return outputs, eval_params


    def build_prediction_dataflow(
            self,
            data_key="data",
            features_key="features",
            logits_key="logits",
            probs_key="probs"):
        """just prediction, does not require labels
        """
        logging.info("building prediction dataflow")
        prediction_params = {}
        
        # dataloader
        inputs = self.data_loader.build_dataflow(self.batch_size, data_key)
        prediction_params["data_key"] = data_key
        
        # model
        assert inputs.get(features_key) is not None
        assert inputs.get(logits_key) is not None
        prediction_params.update(self.model_params)
        prediction_params["is_training"] = False
        outputs, _ = self.model_fn(inputs, prediction_params)

        # add final activation function
        outputs[probs_key] = self.final_activation_fn(outputs[logits_key])
            
        return outputs, prediction_params


    def build_inference_dataflow(
            self,
            infer_params={},
            data_key="data",
            features_key="features",
            labels_key="labels",
            logits_key="logits",
            probs_key="probs"):
        """build an inference workflow
        """
        logging.info("building inference dataflow")
        
        # dataloader
        inputs = self.data_loader.build_dataflow(self.batch_size, data_key)
        infer_params["data_key"] = data_key
        
        # model
        assert inputs.get(features_key) is not None
        infer_params["features_key"] = features_key
        infer_params["labels_key"] = labels_key
        infer_params["logits_key"] = logits_key
        infer_params["is_training"] = False
        infer_params.update(self.model_params)
        model_outputs, _ = self.model_fn(inputs, infer_params)
        infer_params["model"] = self.model_fn
        
        # add final activation function
        model_outputs[probs_key] = self.final_activation_fn(model_outputs[logits_key])

        # add inference stack
        if infer_params.get("importance_task_indices") is None:
            infer_params["importance_task_indices"] = self.tasks if len(self.tasks) != 0 else [0]

        # TODO fix this later
        validate_grammars = False
        scan_grammars = False
            
        # inference params
        # NOTE: load pwms, grammars, etc as needed
        # also importance task indices, importance function
        infer_params.update({
            "batch_size": self.batch_size, 
            #"keep_onehot_sequence": "onehot_sequence" if True else None, # always used: filtering
            #"keep_gradients": "gradients" if infer_params.get("dream") is not None else None,
            #"all_grad_ys": infer_params.get("dream_pattern"),
            "keep_importances": "importances" if True else None,
            #"keep_pwm_scores_full": "pwm-scores-full" if scan_grammars else None, # used for grammars
            "keep_global_pwm_scores": "global-pwm-scores" if validate_grammars else None,
            "keep_pwm_scores": "pwm-scores" if True else None, # always used
            "keep_pwm_raw_scores": "pwm-scores-raw" if True else None,
            "keep_grammar_scores": "grammar-scores" if True else None, # always used
            "keep_grammar_scores_full": "grammar-scores-full" if True else None, # always used
            "keep_ism_scores": "ism-scores" if scan_grammars else None, # adjust this later
            "keep_dmim_scores": "dmim-scores" if scan_grammars else None, # adjust this later
        })

        model_outputs["importance_logits"] = model_outputs["logits"]

        # don't run importances if empty net
        if self.model_params["name"] == "empty_net":
            infer_params["use_importances"] = False
        
        # set up inference stack
        inference_fn = infer_params.get("inference_fn")
        inference_outputs, infer_params = inference_fn(model_outputs, infer_params)

        # delete certain inference outputs if not wanted
        

        return inference_outputs, infer_params


    def run_dataflow(self, driver, sess, coord, outputs, h5_file, sample_size=100000):
        """run dataflow
        """
        # set up the outlayer (tensor --> numpy)
        dataflow_driver = driver(sess, outputs, ignore_outputs=["loss"]) # Outlayer
        
        # set up the saver
        with h5py.File(h5_file, "w") as hf:
        
            h5_handler = H5Handler(
                hf, outputs, sample_size, resizable=True, batch_size=4096)
            
            # now run
            total_examples = 0
            total_visualized = 0
            passed_cutoff = 0 # debug
            try:
                while not coord.should_stop():

                    if total_examples % 1000 == 0:
                        print total_examples
                    
                    example = dataflow_driver.next()

                    #import ipdb
                    #ipdb.set_trace()
                    
                    h5_handler.store_example(example)
                    total_examples += 1
                
                    if (sample_size is not None) and (total_examples >= sample_size):
                        break

            except tf.errors.OutOfRangeError:
                print "Done reading data"

            except StopIteration:
                print "Done reading data"
                
            finally:
                h5_handler.flush()
                h5_handler.chomp_datasets()
        
        return None
    

    def build_restore_graph_function(self, is_ensemble=False, skip=[], scope_change=None):
        """build the restore function
        """
        if is_ensemble: # this is really determined by there being more than 1 ckpt - can use as test?
            def restore_function(sess):
                # TODO adjust this function to be like below
                # for ensemble, just need to adjust scoping
                for i in xrange(len(self.checkpoints)):
                    new_scope = "model_{}/".format(i)
                    print new_scope
                    init_assign_op, init_feed_dict = restore_variables_op(
                        self.checkpoints[i],
                        skip=skip,
                        include_scope=new_scope,
                        scope_change=["", new_scope])
                    sess.run(init_assign_op, init_feed_dict)
        else:
            print self.checkpoints
            if len(self.checkpoints) > 0:
                init_assign_op, init_feed_dict = restore_variables_op(
                    self.checkpoints[0], skip=skip, scope_change=scope_change)
                def restore_function(sess):
                    sess.run(init_assign_op, init_feed_dict)
            else:
                print "WARNING NO CHECKPOINTS USED"
                
        return restore_function

    
    def restore_graph(self, sess, is_ensemble=False, skip=[], scope_change=None):
        """restore saved model from checkpoint into sess
        """
        restore_function = self.build_restore_graph_function(
            is_ensemble=is_ensemble, skip=skip, scope_change=scope_change)
        restore_function(sess)
        
        return None


    def _add_train_op(self):
        """set up the optimizer and generate the training op
        """
        assert self.total_loss is not None
        assert self.optimizer_fn is not None

        # if finetuning, only train certain variables
        if self.finetune:
            variables_to_train = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, logits_key)
        else:
            variables_to_train = None # slim will use all trainable in graphkeys

        # set up optimizer and train op
        self.optimizer = self.optimizer_fn(**self.optimizer_params)
        train_op = slim.learning.create_train_op(
            self.total_loss,
            self.optimizer,
            variables_to_train=variables_to_train,
            summarize_gradients=True)

        return train_op


    def _add_loss(self, labels, logits, data_key=None):
        """set up loss function
        """
        assert not (self.class_weighted_loss and self.positives_focused_loss)

        if self.finetune:
            # adjust which labels and logits go into loss if finetuning
            labels_unstacked = tf.unstack(labels, axis=1)
            labels = tf.stack([labels_unstacked[i] for i in self.finetune_tasks], axis=1)
            logits_unstacked = tf.unstack(logits, axis=1)
            logits = tf.stack([logits_unstacked[i] for i in self.finetune_tasks], axis=1)
            print labels.get_shape()
            print logits.get_shape()

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

        return self.total_loss
    
    
    def _add_metrics(self, labels, probs):
        """set up metrics function with summaries etc
        """
        # set up metrics and values
        self.metric_values, self.metric_updates = self.metrics_fn(
            labels, probs)
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


    def _add_task_subset_accuracy(self):
        """Given task subset, get subset accuracy
        """
        assert self.importances_tasks is not None

        # split and get subset
        labels_unstacked = tf.unstack(self.labels, axis=1)
        labels_subset = tf.stack([labels_unstacked[i] for i in self.importances_tasks], axis=1)

        probs_unstacked = tf.unstack(self.probs, axis=1)
        probs_subset = tf.stack([probs_unstacked[i] for i in self.importances_tasks], axis=1)

        # compare labels to predictions
        correctly_predicted = tf.logical_not(tf.logical_xor(tf.cast(labels_subset, tf.bool), tf.greater_equal(probs_subset, 0.5)))
        accuracy = tf.reduce_mean(tf.cast(correctly_predicted, tf.float32), 1, keep_dims=True)

        return accuracy


class TronnEstimator(tf.estimator.Estimator):
    """Extended estimator to have an inference function"""

    def infer(
            self,
            input_fn,
            predict_keys=None,
            hooks=[],
            checkpoint_path=None,
            yield_single_examples=True):
        """adjust predict function to do inference
        """
        with ops.Graph().as_default() as g:
            random_seed.set_random_seed(self._config.tf_random_seed)
            self._create_and_assert_global_step(g)
            features, input_hooks = self._get_features_from_input_fn(
                input_fn, model_fn_lib.ModeKeys.PREDICT)
            estimator_spec = self._call_model_fn(
                features, None, model_fn_lib.ModeKeys.PREDICT, self.config)
            predictions = self._extract_keys(estimator_spec.predictions, predict_keys)
            all_hooks = list(input_hooks)
            all_hooks.extend(hooks)
            all_hooks.extend(list(estimator_spec.prediction_hooks or []))
            with training.MonitoredSession(
                    session_creator=training.ChiefSessionCreator(
                        checkpoint_filename_with_path=None,  # make sure it doesn't use the checkpoint path
                        master=self._config.master,
                        scaffold=estimator_spec.scaffold,
                        config=self._session_config),
                    hooks=all_hooks) as mon_sess:
                while not mon_sess.should_stop():
                    preds_evaluated = mon_sess.run(predictions)
                    if not yield_single_examples:
                        yield preds_evaluated
                    elif not isinstance(predictions, dict):
                        for pred in preds_evaluated:
                            yield pred
                    else:
                        for i in range(self._extract_batch_length(preds_evaluated)):
                            yield {
                                key: value[i]
                                for key, value in six.iteritems(preds_evaluated)
                            }
    

    
class ModelManager(object):
    """Manages the full model pipeline (utilizes Estimator)"""
    
    def __init__(
            self,
            model_fn,
            model_params):
        """Initialization keeps core of the model - inputs (from separate dataloader) 
        and model with necessary params. All other pieces are part of different graphs
        (ie, training, evaluation, prediction, inference)
        """
        self.model_fn = model_fn
        self.model_params = model_params
        
        
    def build_training_dataflow(
            self,
            inputs,
            optimizer_fn=tf.train.RMSPropOptimizer,
            optimizer_params={
                "learning_rate": 0.002,
                "decay": 0.98,
                "momentum": 0.0},
            features_key="features",
            labels_key="labels"):
        """builds the training dataflow. links up input tensors
        to the model with is_training as True.
        """
        # assertions
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None

        # build model in training mode
        self.model_params.update({"is_training": True})
        outputs, _ = self.model_fn(inputs, self.model_params)

        # add final activation, loss, and train op
        outputs["probs"] = self._add_final_activation_fn(outputs["logits"])
        loss = self._add_loss(outputs[labels_key], outputs["logits"])
        #optimizer = optimizer_fn(**optimizer_params)


        # TODO fix the train op
        train_op = self._add_train_op(loss, optimizer_fn, optimizer_params)
        #train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return outputs, loss, train_op


    def build_evaluation_dataflow(
            self,
            inputs,
            features_key="features",
            labels_key="labels"):
        """build evaluation dataflow. links up input tensors
        to the model with is_training as False
        """
        # assertions
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None

        # build model
        self.model_params.update({"is_training": False})
        outputs, _ = self.model_fn(inputs, self.model_params)

        # add final activation, loss, and metrics
        outputs["probs"] = self._add_final_activation_fn(outputs["logits"])
        loss = self._add_loss(outputs[labels_key], outputs["logits"])
        metrics = self._add_metrics(outputs[labels_key], outputs["probs"], loss)

        return outputs, loss, metrics


    def build_prediction_dataflow(
            self,
            inputs,
            features_key="features"):
        """build prediction dataflow. links up input tensors
        to the model with is_training as False
        """
        # assertions
        assert inputs.get(features_key) is not None

        # build model
        self.model_params.update({"is_training": False})
        outputs, _ = self.model_fn(inputs, self.model_params)

        # add final activation
        outputs["probs"] = self._add_final_activation_fn(outputs["logits"])
        
        return outputs


    def build_inference_dataflow(
            self,
            inputs,
            inference_fn,
            inference_params,
            features_key="features"):
        """build inference dataflow. links up input tensors
        to the model with is_training as False
        """
        # set up prediction dataflow
        outputs = self.build_prediction_dataflow(inputs, features_key=features_key)

        # get the variables to restore here
        variables_to_restore = slim.get_model_variables()
        variables_to_restore.append(tf.train.get_or_create_global_step())

        # adjust inference params as needed
        #inference_params.update({})
        
        # run inference with an inference stack
        outputs, _ = inference_fn(outputs, inference_params)
        print outputs

        return outputs, variables_to_restore

    
    def build_estimator(
            self,
            params=None,
            config=None,
            warm_start=None,
            out_dir="."):
        """build a model fn that will work in the Estimator framework
        """
        # adjust config
        if config is None:
            config=tf.estimator.RunConfig(
                save_summary_steps=30,
                save_checkpoints_secs=None,
                save_checkpoints_steps=10000000000,
                keep_checkpoint_max=None)

        # set up the model function to be called in the run
        def estimator_model_fn(features, labels, mode, params=None, config=None):
            """model fn in the Estimator framework
            """
            # set up the input dict for model fn
            #inputs = {"features": features, "labels": labels}
            inputs = features
            
            # attach necessary things and return EstimatorSpec
            if mode == tf.estimator.ModeKeys.PREDICT:
                inference_mode = params.get("inference_mode", False)
                if not inference_mode:
                    outputs = self.build_prediction_dataflow(inputs)
                    return tf.estimator.EstimatorSpec(mode, predictions=outputs)
                else:
                    outputs, variables_to_restore = self.build_inference_dataflow(
                        inputs,
                        params["inference_fn"],
                        params.get("inference_params", {}))
                    init_op, init_feed_dict = restore_variables_op(
                        params["checkpoint"],
                        skip=["pwm"])
                    init_op = control_flow_ops.group(
                        variables.global_variables_initializer(),
                        variables.local_variables_initializer(),
                        resources.initialize_resources(resources.shared_resources()),
                        init_op)

                    # TODO figure out how to collect standard scaffold and adjust just the saver
                    scaffold = monitored_session.Scaffold(
                        #init_fn=init_fn,
                        init_op=init_op,
                        #local_init_op=None,
                        init_feed_dict=init_feed_dict)
                        #local_init_op=[tf.global_variables_initializer(), tf.local_variables_initializer()], # fyi hack
                        #saver=tf.train.Saver(variables_to_restore))
                    
                    return tf.estimator.EstimatorSpec(
                        mode,
                        predictions=outputs,
                        scaffold=scaffold)
            
            elif mode == tf.estimator.ModeKeys.EVAL:
                outputs, loss, metrics = self.build_evaluation_dataflow(inputs)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            
            elif mode == tf.estimator.ModeKeys.TRAIN:
                outputs, loss, train_op = self.build_training_dataflow(inputs)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
            else:
                raise Exception, "mode does not exist!"
            
            return None
            
        # instantiate the custom estimator
        estimator = TronnEstimator(
        #estimator = tf.estimator.Estimator(
            estimator_model_fn,
            model_dir=out_dir,
            params=params,
            config=config)

        return estimator

    
    def train(
            self,
            input_fn,
            out_dir,
            config=None,
            steps=None,
            hooks=[]):
        """train an estimator. if steps is None, goes on forever or until input_fn
        runs out.
        """
        # build estimator
        estimator = self.build_estimator(
            config=config,
            out_dir=out_dir)
        
        # train until input producers are out
        estimator.train(
            input_fn=input_fn,
            max_steps=steps,
            hooks=hooks)

        # return the latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(out_dir)
        
        return latest_checkpoint


    def evaluate(
            self,
            input_fn,
            out_dir,
            config=None,
            steps=None,
            checkpoint=None,
            hooks=[]):
        """evaluate a trained estimator
        """
        # build evaluation estimator
        estimator = self.build_estimator(
            config=config,
            out_dir=out_dir)
        
        # evaluate
        eval_metrics = estimator.evaluate(
            input_fn=input_fn,
            steps=steps,
            checkpoint_path=checkpoint)
        logging.info("EVAL: {}".format(eval_metrics))

        return eval_metrics


    def predict(
            self,
            input_fn,
            out_dir,
            config=None,
            predict_keys=None,
            checkpoint=None,
            hooks=[],
            yield_single_examples=True):
        """predict on a trained estimator
        """
        # build prediction estimator
        estimator = self.build_estimator(
            config=config,
            out_dir=out_dir)
        
        # evaluate
        return estimator.predict(
            input_fn=input_fn,
            checkpoint_path=checkpoint)

    
    def infer(
            self,
            input_fn,
            out_dir,
            inference_fn,
            inference_params={},
            config=None,
            predict_keys=None,
            checkpoint=None,
            hooks=[],
            yield_single_examples=True):
        """infer on a trained estimator
        """
        # build inference estimator
        params = {
            "checkpoint": inference_params.get("checkpoint"), # TODO factor out somehow?
            "inference_mode": True,
            "inference_fn": inference_fn,
            "inference_params": inference_params} # TODO pwms etc etc

        # TODO may need to adjust the scaffold with a different restore functoin

        #restore_hook = RestoreHook(
        #    warm_start,
        #    warm_start_params)
        #training_hooks.append(restore_hook)
        
        # build estimator
        estimator = self.build_estimator(
            params=params,
            config=config,
            out_dir=out_dir)

        # return generator
        return estimator.infer( # adjusted here for tronn
            input_fn=input_fn,
            checkpoint_path=checkpoint)

    
    def train_and_evaluate_with_early_stopping(
            self,
            train_input_fn,
            eval_input_fn,
            out_dir,
            max_epochs=20,
            early_stopping_metric="mean_auprc",
            epoch_patience=2,
            warm_start=None,
            warm_start_params={}):
        """run full training loop with evaluation for early stopping
        """
        # set up stopping conditions
        stopping_log = "{}/stopping.log".format(out_dir)
        if os.path.isfile(stopping_log):
            # if there is a stopping log, restart using the info in the log
            with open(stopping_log, "r") as fp:
                start_epoch, best_metric_val, consecutive_bad_epochs, best_checkpoint = map(
                    float, fp.readline().strip().split())
        else:
            # fresh run
            start_epoch = 0
            best_metric_val = None
            consecutive_bad_epochs = 0
            best_checkpoint = None

        # run through epochs
        for epoch in xrange(start_epoch, max_epochs):
            logging.info("EPOCH {}".format(epoch))
            
            # restore from transfer as needed
            training_hooks = []
            if (epoch == 0) and warm_start is not None:
                logging.info("Restoring from {}".format(warm_start))
                restore_hook = RestoreHook(
                    warm_start,
                    warm_start_params)
                training_hooks.append(restore_hook)

            # train
            latest_checkpoint = self.train(
                train_input_fn,
                "{}/train".format(out_dir),
                steps=None,
                hooks=training_hooks)
            
            # eval
            eval_metrics = self.evaluate(
                eval_input_fn,
                "{}/eval".format(out_dir),
                steps=1000,
                checkpoint=latest_checkpoint)

            # early stopping and saving best model
            # TODO fix the early stopping logic conditions
            if best_metric_val is None or ('loss' in early_stopping_metric) != (
                    eval_metrics[early_stopping_metric] > best_metric_val):
                best_metric_val = eval_metrics[early_stopping_metric]
                consecutive_bad_epochs = 0
                best_checkpoint = latest_checkpoint
                with open(os.path.join(out_dir, 'best.log'), 'w') as fp:
                    fp.write('epoch %d\n'%epoch)
                    fp.write("checkpoint path: {}\n".format(best_checkpoint))
                    fp.write(str(eval_metrics))
                with open(stopping_log, 'w') as out:
                    out.write("{}\t{}\t{}\t{}".format(
                        epoch,
                        best_metric_val,
                        consecutive_bad_epochs,
                        best_checkpoint))
            else:
                # break
                consecutive_bad_epochs += 1
                if consecutive_bad_epochs > epoch_patience:
                    logging.info("early stopping triggered")
                    break

        # return the best checkpoint
        return best_checkpoint

    
    def _add_final_activation_fn(
            self,
            logits,
            activation_fn=tf.nn.sigmoid):
        """add final activation function
        """
        return activation_fn(logits)

    
    def _add_loss(self, labels, logits, loss_fn=tf.losses.sigmoid_cross_entropy):
        """add loss
        """
        if False:
        #if self.finetune:
            # adjust which labels and logits go into loss if finetuning
            labels_unstacked = tf.unstack(labels, axis=1)
            labels = tf.stack([labels_unstacked[i] for i in self.finetune_tasks], axis=1)
            logits_unstacked = tf.unstack(logits, axis=1)
            logits = tf.stack([logits_unstacked[i] for i in self.finetune_tasks], axis=1)
            print labels.get_shape()
            print logits.get_shape()

        # split out getting the positive weights so that only the right ones go into the loss function

        if False:
        #if self.class_weighted_loss:
            pos_weights = get_positive_weights_per_task(self.data_files[data_key])
            if self.finetune:
                pos_weights = [pos_weights[i] for i in self.finetune_tasks]
            self.loss = class_weighted_loss_fn(
                self.loss_fn, labels, logits, pos_weights)
        elif False:
        #elif self.positives_focused_loss:
            task_weights, class_weights = get_task_and_class_weights(self.data_files[data_key])
            if self.finetune:
                task_weights = [task_weights[i] for i in self.finetune_tasks]
            if self.finetune:
                class_weights = [class_weights[i] for i in self.finetune_tasks]
            self.loss = positives_focused_loss_fn(
                self.loss_fn, labels, logits, task_weights, class_weights)
        else:
            # this is the function
            loss = loss_fn(labels, logits)

        # this is all registered losses
        total_loss = tf.losses.get_total_loss()

        return total_loss

    
    def _add_train_op(
            self,
            loss,
            optimizer_fn,
            optimizer_params):
        """set up the optimizer and generate the training op
        """
        # TODO adjust this to remove reliance on contrib
        optimizer = optimizer_fn(**optimizer_params)
        train_op = slim.learning.create_train_op(
            loss,
            optimizer,
            variables_to_train=None, # use this for finetune or training subset
            summarize_gradients=True)

        return train_op

    
    
    def _add_metrics(self, labels, probs, loss, metrics_fn=get_global_avg_metrics):
        """set up metrics function with summaries etc
        """
        if False:
            # set up metrics and values
            metric_values, metric_updates = metrics_fn(
                labels, probs)
            for update in metric_updates: tf.add_to_collection(
                tf.GraphKeys.UPDATE_OPS, update)

        metric_map = metrics_fn(labels, probs)
        # Add losses to metrics
        #mean_loss, _ = tf.metrics.mean(
        #    loss, updates_collections=tf.GraphKeys.UPDATE_OPS)
        ##metric_map.update({
        #    "total_loss": loss,
        #    "mean_loss": mean_loss
        #})

        return metric_map


    def _add_summaries(self):
        """add things you want to track on tensorboard
        """
        

        
        return

    

class SlimManager(ModelManager):
    """Utilize tf-slim, this is for backwards compatibility"""

    def train(
            self,
            input_fn,
            out_dir,
            checkpoint=None,
            steps=None):
        """training with tf-slim
        """
        with tf.Graph().as_default():

            # build_graph
            outputs, loss, train_op = super(
                SlimManager, self).build_training_dataflow(inputs)

            # add summaries

            # print a param count

            # create the init fn


            # pass it all to tf-slim

        # return latest checkpoint
        return None

    
    def evaluate(
            self,
            input_fn,
            out_dir,
            steps=None,
            checkpoint=None):
        """evaluation with tf-slim
        """
        pass


    def predict(self):
        pass


    def infer(self):
        pass

    
class MetaGraphManager(ModelManager):
    """Model manager for metagraphs, uses Estimator framework"""

    def __init__():
        pass


    def train():
        pass

    def eval():
        pass

    def infer():
        pass

    
    def _build_metagraph_model_fn(self, metagraph):
        """given a metagraph file, build a model function around it
        """
        # TODO use a hook instead to put it in?

        
        def metagraph_model_fn(inputs, params):
            """wrapper for a loading a metagraph based model
            """
            assert params.get("sess") is not None

            # restore the model
            saver = tf.train.import_meta_graph(metagraph)
            saver.restore(params["sess"], metagraph)

            # pull out logits
            outputs["logits"] = tf.get_collection("logits")
            
            return outputs, params
        
        return metagraph_model_fn
    
    
    

    

class KerasManager(ModelManager):
    """Model manager for Keras models, uses Estimator framework"""
    
    def __init__():
        pass


    def train():
        pass


    def eval():
        pass


    def predict():
        pass


    def infer():
        pass







def infer_and_save_to_hdf5(generator, h5_file, sample_size):
    """wrapper routine to run inference and save the results out
    """
    # generate first set of outputs to know shapes
    print "starting inference"
    first_example = generator.next() # put in try except

    # set up the saver
    with h5py.File(h5_file, "w") as hf:
        
        h5_handler = H5Handler(
            hf,
            first_example,
            sample_size,
            resizable=True,
            batch_size=4096,
            is_tensor_input=False)

        # and store first outputs
        h5_handler.store_example(first_example)
        
        # now run
        total_examples = 1
        total_visualized = 0
        passed_cutoff = 0 # debug
        try:
            #while not coord.should_stop():
            for i in xrange(1, sample_size):
                
                if total_examples % 1000 == 0:
                    print total_examples
                    
                example = generator.next()
                
                #import ipdb
                #ipdb.set_trace()
                    
                h5_handler.store_example(example)
                total_examples += 1
                
                #if (sample_size is not None) and (total_examples >= sample_size):
                #    break

        except tf.errors.OutOfRangeError:
            print "Done reading data"

        except StopIteration:
            print "Done reading data"
                
        finally:
            h5_handler.flush()
            h5_handler.chomp_datasets()
            
    return None
