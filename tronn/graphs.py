"""Contains functions to make running tensorflow graphs easier
"""

import logging
import h5py

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_ops import restore_variables_op
from tronn.util.tf_ops import class_weighted_loss_fn
from tronn.util.tf_ops import positives_focused_loss_fn

#from tronn.datalayer import get_task_and_class_weights
#from tronn.datalayer import get_positive_weights_per_task

from tronn.outlayer import H5Handler


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
            "keep_onehot_sequence": "onehot_sequence" if True else None, # always used: filtering
            "keep_gradients": "gradients" if infer_params.get("dream") is not None else None,
            "all_grad_ys": infer_params.get("dream_pattern"),
            "keep_importances": "importances" if True else None,
            "keep_pwm_scores_full": "pwm-scores-full" if scan_grammars else None, # used for grammars
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
        
