"""Contains functions to make running tensorflow graphs easier
"""

import os
import json
import logging
import h5py

import six

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim # TODO try deprecate!

from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.keras import _clone_and_build_model as build_keras_model

from tensorflow.python.keras import models

from tensorflow.python.training import monitored_session

from tronn.util.tf_ops import restore_variables_op
from tronn.util.tf_ops import class_weighted_loss_fn
from tronn.util.tf_ops import positives_focused_loss_fn

from tronn.util.tf_utils import print_param_count

from tronn.outlayer import H5Handler

from tronn.learn.estimator import TronnEstimator

from tronn.learn.evaluation import get_global_avg_metrics
from tronn.learn.evaluation import get_regression_metrics

from tronn.learn.learning import RestoreHook
from tronn.learn.learning import KerasRestoreHook
from tronn.learn.learning import DataSetupHook
from tronn.learn.learning import DataCleanupHook

from tronn.nets.nets import net_fns

from tronn.contrib.pytorch.nets import net_fns as pytorch_net_fns

from tronn.interpretation.interpret import visualize_region
from tronn.interpretation.dreaming import dream_one_sequence

from tronn.visualization import visualize_debug

from tronn.util.utils import DataKeys
from tronn.util.formats import write_to_json

        
_TRAIN_PHASE = "train"
_EVAL_PHASE = "eval"


class ModelManager(object):
    """Manages the full model pipeline (utilizes Estimator framework)"""
    
    def __init__(
            self,
            model,
            name="nn_model"):
        """Initialization keeps core of the model - inputs (from separate dataloader) 
        and model with necessary params. All other pieces are part of different graphs
        (ie, training, evaluation, prediction, inference)

        Args:
          model: a dict of name, params, etc
        """
        # set up from model dict
        self.name = model["name"]
        self.model_fn = net_fns[self.name]
        self.model_params = model.get("params", {})
        self.model_dir = model["model_dir"]
        self.model_checkpoint = model.get("checkpoint")

        
    def describe(self):
        """return a dictionary that can be saved out to json
        for future loading
        """
        model = {
            "name": self.name,
            "params": self.model_params,
            "model_dir": self.model_dir,
            "checkpoint": self.model_checkpoint
        }
        
        return model
        
        
    def build_training_dataflow(
            self,
            inputs,
            optimizer_fn=tf.train.RMSPropOptimizer,
            optimizer_params={
                "learning_rate": 0.002,
                "decay": 0.98,
                "momentum": 0.0},
            features_key=DataKeys.FEATURES,
            labels_key=DataKeys.LABELS,
            logits_key=DataKeys.LOGITS,
            probs_key=DataKeys.PROBABILITIES,
            regression=False,
            logit_indices=[]):
        """builds the training dataflow. links up input tensors
        to the model with is_training as True.
        """
        # assertions
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None

        # build model in training mode
        self.model_params.update({"is_training": True})
        outputs, _ = self.model_fn(inputs, self.model_params)

        # if adjusting logits, need to be done here, before feeding into loss
        if len(logit_indices) > 0:
            outputs[logits_key] = tf.gather(outputs[logits_key], logit_indices, axis=1)
        
        # add final activation + loss
        if regression:
            loss = self._add_loss(
                outputs[labels_key],
                outputs[logits_key],
                loss_fn=tf.losses.mean_squared_error)
        else:
            outputs[probs_key] = self._add_final_activation_fn(
                outputs[DataKeys.LOGITS])
            loss = self._add_loss(
                outputs[labels_key],
                outputs[logits_key])

        # add train op
        # TODO fix the train op so that doesn't rely on slim
        train_op = self._add_train_op(loss, optimizer_fn, optimizer_params)
        
        return outputs, loss, train_op


    def build_evaluation_dataflow(
            self,
            inputs,
            features_key=DataKeys.FEATURES,
            labels_key=DataKeys.LABELS,
            logits_key=DataKeys.LOGITS,
            probs_key=DataKeys.PROBABILITIES,
            regression=False,
            logit_indices=[]):
        """build evaluation dataflow. links up input tensors
        to the model with is_training as False
        """
        # assertions
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None

        # build model
        self.model_params.update({"is_training": False})
        outputs, _ = self.model_fn(inputs, self.model_params)
        
        # if adjusting logits, need to be done here, before feeding into loss
        if len(logit_indices) > 0:
            outputs[logits_key] = tf.gather(outputs[logits_key], logit_indices, axis=1)

        # add loss
        loss = self._add_loss(
            outputs[labels_key],
            outputs[logits_key])

        # add final activateion + metrics
        if regression:
            metrics = self._add_metrics(
                outputs[labels_key],
                outputs[logits_key],
                loss,
                metrics_fn=get_regression_metrics)
        else:
            outputs[probs_key] = self._add_final_activation_fn(
                outputs[logits_key])
            metrics = self._add_metrics(
                outputs[labels_key],
                outputs[probs_key],
                loss)

        return outputs, loss, metrics


    def build_prediction_dataflow(
            self,
            inputs,
            features_key=DataKeys.FEATURES,
            logits_key=DataKeys.LOGITS,
            probs_key=DataKeys.PROBABILITIES,
            regression=False,
            logit_indices=[]):
        """build prediction dataflow. links up input tensors
        to the model with is_training as False
        """
        # assertions
        assert inputs.get(features_key) is not None

        # build model
        self.model_params.update({"is_training": False})
        outputs, _ = self.model_fn(inputs, self.model_params)

        # if adjusting logits, need to be done here
        if len(logit_indices) > 0:
            outputs[logits_key] = tf.gather(outputs[logits_key], logit_indices, axis=1)

        # add final activation
        if not regression:
            outputs[probs_key] = self._add_final_activation_fn(
                outputs[logits_key])
        
        return outputs


    def build_inference_dataflow(
            self,
            inputs,
            inference_fn,
            inference_params,
            features_key=DataKeys.FEATURES,
            logits_key=DataKeys.LOGITS,
            probs_key=DataKeys.PROBABILITIES,
            regression=False,
            logit_indices=[]):
        """build inference dataflow. links up input tensors
        to the model with is_training as False
        """
        # assertions
        assert inputs.get(features_key) is not None
        
        # set up prediction dataflow
        outputs = self.build_prediction_dataflow(
            inputs,
            features_key=features_key,
            logits_key=logits_key,
            probs_key=probs_key,
            regression=regression,
            logit_indices=logit_indices)

        # get the variables to restore here
        # TODO remove reliance on slim
        variables_to_restore = slim.get_model_variables()
        variables_to_restore.append(tf.train.get_or_create_global_step())
        
        # run inference with an inference stack
        outputs, _ = inference_fn(outputs, inference_params)

        return outputs, variables_to_restore

    
    def build_estimator(
            self,
            params=None,
            config=None,
            warm_start=None,
            regression=False,
            logit_indices=[],
            out_dir="."):
        """build a model fn that will work in the Estimator framework
        """
        # adjust config
        if config is None:
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = False #True
            config = tf.estimator.RunConfig(
                save_summary_steps=30,
                save_checkpoints_secs=None,
                save_checkpoints_steps=10000000000,
                keep_checkpoint_max=None,
                session_config=session_config)

        # set up the model function to be called in the run
        def estimator_model_fn(
                features,
                labels,
                mode,
                params=None,
                config=config):
            """model fn in the Estimator framework
            """
            # set up the input dict for model fn
            # note that all input goes through features (including labels)
            inputs = features
            
            # attach necessary things and return EstimatorSpec
            if mode == tf.estimator.ModeKeys.PREDICT:
                inference_mode = params.get("inference_mode", False)
                if not inference_mode:
                    # prediction mode
                    outputs = self.build_prediction_dataflow(
                        inputs, regression=regression, logit_indices=logit_indices)
                    return tf.estimator.EstimatorSpec(mode, predictions=outputs)
                else:
                    # inference mode
                    outputs, variables_to_restore = self.build_inference_dataflow(
                        inputs,
                        params["inference_fn"],
                        params,
                        regression=regression,
                        logit_indices=logit_indices)

                    # adjust model weight loading
                    if self.model_checkpoint is None:
                        # this is important for py_func wrapped models like pytorch
                        # and/or keras models where we use a py_func op for loading
                        print "WARNING: NO CHECKPOINT BEING USED"
                        return tf.estimator.EstimatorSpec(mode, predictions=outputs)
                    else:
                        # create custom init fn for tensorflow
                        init_op, init_feed_dict = restore_variables_op(
                            self.model_checkpoint,
                            skip=["pwm"])
                        def init_fn(scaffold, sess):
                            sess.run(init_op, init_feed_dict)
                        # custom scaffold to load checkpoint
                        scaffold = monitored_session.Scaffold(
                            init_fn=init_fn)
                        return tf.estimator.EstimatorSpec(
                            mode, predictions=outputs, scaffold=scaffold)
            
            elif mode == tf.estimator.ModeKeys.EVAL:
                # evaluation mode
                outputs, loss, metrics = self.build_evaluation_dataflow(
                    inputs, regression=regression, logit_indices=logit_indices)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            
            elif mode == tf.estimator.ModeKeys.TRAIN:
                # training mode
                outputs, loss, train_op = self.build_training_dataflow(
                    inputs, regression=regression, logit_indices=logit_indices)
                print_param_count()
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
            else:
                raise Exception, "mode does not exist!"
            
            return None
            
        # instantiate the custom estimator
        estimator = TronnEstimator(
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
            hooks=[],
            regression=False,
            logit_indices=[]):
        """train an estimator. if steps is None, goes on forever or until input_fn
        runs out.
        """
        # build estimator and train
        estimator = self.build_estimator(
            config=config,
            out_dir=out_dir,
            regression=regression,
            logit_indices=logit_indices)
        estimator.train(input_fn=input_fn, max_steps=steps, hooks=hooks)
        
        return tf.train.latest_checkpoint(out_dir)


    def evaluate(
            self,
            input_fn,
            out_dir,
            config=None,
            steps=None,
            checkpoint=None,
            hooks=[],
            regression=False,
            logit_indices=[]):
        """evaluate a trained estimator
        """
        # build evaluation estimator and evaluate
        estimator = self.build_estimator(
            config=config,
            out_dir=out_dir,
            regression=regression,
            logit_indices=logit_indices)
        eval_metrics = estimator.evaluate(
            input_fn=input_fn,
            steps=steps,
            checkpoint_path=checkpoint,
            hooks=hooks)
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
            yield_single_examples=True,
            logit_indices=[]):
        """predict on a trained estimator
        """
        # build prediction estimator
        estimator = self.build_estimator(
            config=config,
            out_dir=out_dir,
            logit_indices=logit_indices)

        # return prediction generator
        return estimator.predict(
            input_fn=input_fn,
            checkpoint_path=checkpoint,
            hooks=hooks)

    
    def infer(
            self,
            input_fn,
            out_dir,
            inference_params={},
            config=None,
            predict_keys=None,
            checkpoint=None,
            hooks=[],
            yield_single_examples=True,
            logit_indices=[]):
        """infer on a trained estimator
        """
        hooks.append(DataSetupHook())

        # convert inference fn
        inference_params["inference_fn"] = net_fns[
            inference_params["inference_fn_name"]]
        
        # build estimator
        estimator = self.build_estimator(
            params=inference_params,
            config=config,
            out_dir=out_dir,
            logit_indices=logit_indices)

        # return generator
        return estimator.infer(
            input_fn=input_fn,
            checkpoint_path=checkpoint,
            hooks=hooks)

    
    def dream(
            self,
            dream_dataset,
            input_fn,
            feed_dict,
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
            "checkpoint": checkpoint,
            "inference_mode": True,
            "inference_fn": inference_fn,
            "inference_params": inference_params} # TODO pwms
        
        # build estimator
        estimator = self.build_estimator(
            params=params,
            config=config,
            out_dir=out_dir)

        # return generator
        return estimator.dream_generator(
            dream_dataset,
            input_fn,
            feed_dict,
            predict_keys=predict_keys)

    
    @staticmethod
    def _setup_train_summary(train_summary):
        """maintain a dictionary that summarizes training
        
        Args:
          train_summary: dictionary (may be empty) of training status

        Returns:
          train_summary: filled out dictionary covering missing gaps
        """
        train_summary.update({
            "start_epoch": train_summary.get("start_epoch", 0),
            "best_metric_val": train_summary.get("best_metric_val"),
            "consecutive_bad_epochs": int(train_summary.get("consecutive_bad_epochs", 0)),
            "best_epoch": train_summary.get("best_epoch"),
            "best_checkpoint": train_summary.get("best_checkpoint"),
            "last_phase": train_summary.get("last_phase", _EVAL_PHASE)
        })
        
        if train_summary["best_metric_val"] is not None:
            train_summary["best_metric_val"] == float(train_summary["best_metric_val"])
        
        return train_summary
    
    
    def train_and_evaluate(
            self,
            train_input_fn,
            eval_input_fn,
            out_dir,
            max_epochs=30,
            early_stopping_metric="mean_auprc",
            epoch_patience=3,
            train_steps=None,
            eval_steps=1000,
            warm_start=None,
            warm_start_params={},
            regression=False,
            model_summary_file=None,
            train_summary_file=None,
            early_stopping=True,
            multi_gpu=False,
            logit_indices=[]):
        """run full training loop with evaluation for early stopping
        """
        # assertions
        assert model_summary_file is not None
        assert train_summary_file is not None

        # pull in train summary if it exists
        if os.path.isfile(train_summary_file):
            with open(train_summary_file, "r") as fp:
                train_summary = json.load(fp)
            print train_summary
        else:
            train_summary = {}

        # fill out as needed
        train_summary = ModelManager._setup_train_summary(train_summary)

        # pull out needed variables
        start_epoch = train_summary["start_epoch"]
        best_metric_val = train_summary["best_metric_val"]
        consecutive_bad_epochs = train_summary["consecutive_bad_epochs"]
        best_checkpoint = train_summary["best_checkpoint"]
        last_phase = train_summary["last_phase"]
        
        # update model summary and training summary
        self.model_checkpoint = best_checkpoint
        write_to_json(self.describe(), model_summary_file)
        write_to_json(train_summary, train_summary_file)
        
        # if early stopping and criterion met, don't run training
        if early_stopping and (consecutive_bad_epochs >= epoch_patience):
            return best_checkpoint
            
        # set up run config
        # requires 1.9+ (prefer 1.11 bc of batch norm issues in 1.9 and 1.10)
        if multi_gpu:
            distribution = tf.contrib.distribute.MirroredStrategy()
            config = tf.estimator.RunConfig(
                save_summary_steps=30,
                save_checkpoints_secs=None,
                save_checkpoints_steps=10000000000,
                keep_checkpoint_max=None,
                train_distribute=distribution)
        else:
            config = None

        # run through epochs
        for epoch in xrange(start_epoch, max_epochs):
            logging.info("EPOCH {}".format(epoch))
            train_summary["start_epoch"] = epoch
            write_to_json(train_summary, train_summary_file)
            
            # restore from transfer as needed
            training_hooks = []
            if (epoch == 0) and warm_start is not None:
                logging.info("Restoring from {}".format(warm_start))
                restore_hook = RestoreHook(
                    warm_start,
                    warm_start_params)
                training_hooks.append(restore_hook)

            # train
            if last_phase == _EVAL_PHASE:
                self.train(
                    train_input_fn,
                    "{}/train".format(out_dir),
                    steps=train_steps,
                    config=config,
                    hooks=training_hooks,
                    regression=regression,
                    logit_indices=logit_indices)
                
                last_phase = _TRAIN_PHASE
                train_summary["last_phase"] = last_phase
                write_to_json(train_summary, train_summary_file)
            
            # eval
            if last_phase == _TRAIN_PHASE:
                latest_checkpoint = tf.train.latest_checkpoint(
                    "{}/train".format(out_dir))
                eval_metrics = self.evaluate(
                    eval_input_fn,
                    "{}/eval".format(out_dir),
                    steps=eval_steps,
                    checkpoint=latest_checkpoint,
                    regression=regression,
                    logit_indices=logit_indices)

                last_phase = _EVAL_PHASE
                train_summary["last_phase"] = last_phase
                write_to_json(train_summary, train_summary_file)

            # determine if epoch of training was good
            if best_metric_val is None:
                is_good_epoch = True
            elif early_stopping_metric in ["loss", "mse"]:
                is_good_epoch = eval_metrics[early_stopping_metric] < best_metric_val
            else:
                is_good_epoch = eval_metrics[early_stopping_metric] > best_metric_val

            # if good epoch, save out new metrics
            if is_good_epoch:
                best_metric_val = eval_metrics[early_stopping_metric]
                consecutive_bad_epochs = 0
                best_checkpoint = latest_checkpoint
                train_summary["best_metric_val"] = best_metric_val
                train_summary["consecutive_bad_epochs"] = consecutive_bad_epochs
                train_summary["best_epoch"] = epoch
                train_summary["best_checkpoint"] = best_checkpoint
                train_summary["metrics"] = eval_metrics
                self.model_checkpoint = best_checkpoint
            else:
                # increase bad epoch count
                consecutive_bad_epochs += 1
                train_summary["consecutive_bad_epochs"] = consecutive_bad_epochs
                
            # save out model summary and train summary
            write_to_json(self.describe(), model_summary_file)
            write_to_json(train_summary, train_summary_file)
            
            # stop if early stopping
            if early_stopping:
                if consecutive_bad_epochs >= epoch_patience:
                    logging.info(
                        "early stopping triggered "
                        "on epoch {} "
                        "with patience {}".format(epoch, epoch_patience))
                    break

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
        # for finetune - just choose which ones go into loss

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
        # TODO adjust this to remove reliance on slim
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
        metric_map = metrics_fn(labels, probs)

        return metric_map


    def _add_summaries(self):
        """add things you want to track on tensorboard
        """
        return None

    
    @staticmethod
    def infer_and_save_to_h5(generator, h5_file, sample_size, debug=False):
        """wrapper routine to run inference and save the results out
        """
        if debug:
            viz_dir = "{}/viz.debug".format(os.path.dirname(h5_file))
            os.system("mkdir -p {}".format(viz_dir))
        
        # generate first set of outputs to know shapes
        print "starting inference"
        first_example = generator.next()
        
        # set up the saver
        with h5py.File(h5_file, "w") as hf:

            h5_handler = H5Handler(
                hf,
                first_example,
                sample_size,
                resizable=True,
                batch_size=min(4096, sample_size),
                is_tensor_input=False)

            # and store first outputs
            h5_handler.store_example(first_example)

            # now run
            total_examples = 1
            try:
                for i in xrange(1, sample_size):
                    if total_examples % 1000 == 0:
                        print total_examples

                    example = generator.next()
                    h5_handler.store_example(example)
                    total_examples += 1
                    
                    if debug:
                        # here, generate useful graphs
                        # (2) pwm scores, raw or not (pwm x pos with pwm vector)
                        # (3) dmim - dmim vector?
                        import ipdb
                        ipdb.set_trace()
                        
                        prefix = "{}/{}".format(viz_dir, os.path.basename(h5_file).split(".h5")[0])
                        visualize_debug(example, prefix)

            except StopIteration:
                print "Done reading data"

            finally:
                h5_handler.flush()
                h5_handler.chomp_datasets()

        return None

    
    @staticmethod
    def dream_and_save_to_h5(generator, h5_handle, group, sample_size=100000):
        """wrapper routine to run dreaming and save results out
        """
        logging.info("starting dream")
        first_example = generator.next()
        
        # set up saver
        h5_handler = H5Handler(
            h5_handle,
            first_example,
            sample_size,
            group=group,
            resizable=True,
            batch_size=4096,
            is_tensor_input=False)

        # and score first output
        h5_handler.store_example(first_example)

        # now run
        total_examples = 1
        try:
            for i in xrange(1, sample_size):
                if total_examples % 1000 == 0:
                    print total_examples

                example = generator.next()
                h5_handler.store_example(example)
                total_examples += 1

        except StopIteration:
            print "Done reading data"

        finally:
            h5_handler.flush()
            h5_handler.chomp_datasets()
        
        return None


class KerasModelManager(ModelManager):
    """Model manager for Keras models"""

    def __init__(
            self,
            keras_model=None,
            keras_model_path=None,
            custom_objects=None,
            model_dir=None,
            model_params=None):
        """extract the keras model fn and keep the model fn, to be called by 
        tronn estimator. also set up first checkpoint
        
        Note that for inference you do not use variable scope because Keras
        was not built with variable reuse in mind

        Draws liberally from tf.keras.estimator.model_to_estimator fn
        """
        self.model_dir = model_dir

        # check mutually exclusive
        if not (keras_model or keras_model_path):
            raise ValueError(
                'Either `keras_model` or `keras_model_path` needs to be provided.')
        if keras_model and keras_model_path:
            raise ValueError(
                'Please specity either `keras_model` or `keras_model_path`, '
                'but not both.')

        # set up keras model
        if not keras_model:
            self.keras_model = models.load_model(keras_model_path)
        else:
            self.keras_model = keras_model
        
        # set up model fn
        def keras_estimator_fn(inputs, model_fn_params):
            """wrap up keras model call
            """
            features = inputs[DataKeys.FEATURES]
            labels = inputs[DataKeys.LABELS]
            outputs = dict(inputs)
            
            # for mahfuza models
            features = tf.squeeze(features, axis=1)

            # build keras model
            if model_fn_params.get("is_training", False):
                mode = model_fn_lib.ModeKeys.TRAIN
            else:
                mode = model_fn_lib.ModeKeys.PREDICT
            model = build_keras_model(mode, self.keras_model, model_fn_params, features, labels)
            model.layers.pop() # remove the sigmoid activation
            model = tf.keras.models.Model(model.input, model.layers[-1].output)
            #print [layer.name for layer in model.layers]

            # set up outputs
            outputs.update(dict(zip(model.output_names, model.outputs)))
            out_layer_key = "dense_3" # is 12 just the layer num?
            outputs[DataKeys.LOGITS] = outputs[out_layer_key]

            # add to collection to make sure restored correctly
            is_inference = self.model_params.get("inference_mode", False)
            if not is_inference:
                # add variables to model variables collection to make
                # sure they are restored correctly
                for v in tf.trainable_variables():
                    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
            else:
                # set up a py_func op for now that has the init fn
                # and set up a hook to restore from there
                # build an init fn or init op to run in a KerasRestoreHook
                def init_fn():
                    model.set_weights(self.keras_weights)
                init_op = tf.py_func(
                    func=init_fn,
                    inp=[],
                    Tout=[],
                    stateful=False,
                    name="init_keras")
                tf.get_collection("KERAS_INIT")
                tf.add_to_collection("KERAS_INIT", init_op)
            
            return outputs, model_fn_params

        # store fn and params for later
        self.model_fn = keras_estimator_fn
        self.model_params = model_params
        self.model_params.update({"model_reuse": False})
        
        # set up weights
        # TODO figure out how to check if weights were even initialized
        self.keras_weights = self.keras_model.get_weights()
        self.model_checkpoint = self.save_checkpoint_from_keras_model()
        
        # debug check
        #print self.keras_model.input_names
        #print self.keras_model.output_names


    def save_checkpoint_from_keras_model(self):
        """create a checkpoint from the keras model

        draws heavily from _save_first_checkpoint in keras to estimator fn
        from tensorflow 1.8
        """
        keras_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if not keras_checkpoint:
            keras_checkpoint = "{}/keras_model.ckpt".format(self.model_dir)
            with tf.Graph().as_default() as g:
                tf.train.create_global_step(g)
                mode = model_fn_lib.ModeKeys.TRAIN
                model = build_keras_model(
                    mode, self.keras_model, self.model_params)
                with tf.Session() as sess:
                    model.set_weights(self.keras_weights)
                    # TODO - check if adding to model variables necessary here
                    for v in tf.trainable_variables():
                        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
                    if False:
                        print tf.trainable_variables()
                        print [len(layer.weights) for layer in model.layers]
                        print sum([len(layer.weights) for layer in model.layers])
                        print len([layer.name for layer in model.layers])
                        print len(tf.trainable_variables())
                        print len(self.keras_weights)
                        print self.keras_weights[0][:,:,0]
                        print sess.run(tf.trainable_variables()[0])[:,:,0]
                    saver = tf.train.Saver()
                    saver.save(sess, keras_checkpoint)

        # test
        #keras_checkpoint = "/srv/scratch/dskim89/mahfuza/models/mouse.all.fbasset/train/model.ckpt-1"
        #keras_checkpoint = "/srv/scratch/dskim89/mahfuza/models/mouse.all.fbasset/test.ckpt"
        with tf.Graph().as_default() as g:
            tf.train.create_global_step(g)
            #mode = model_fn_lib.ModeKeys.TRAIN
            #model = build_keras_model(
            #    mode,
            #    self.keras_model,
            #    self.model_params)
            #model.layers.pop() # remove the sigmoid - but check this
            #model = tf.keras.models.Model(model.input, model.layers[-1].output)
            features = tf.placeholder(tf.float32, shape=[64,1,1000,4])
            labels = tf.placeholder(tf.float32, shape=[64,279])
            inputs = {"features": features, "labels": labels}
            self.model_fn(inputs, {})
            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(sess, keras_checkpoint)
                print tf.trainable_variables()
                print sess.run(tf.trainable_variables()[0])[:,:,0]

        return keras_checkpoint

    
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
        """adjust a few things for keras
        """
        hooks.append(KerasRestoreHook())
        
        return super(KerasModelManager, self).infer(
            input_fn,
            out_dir,
            inference_fn,
            inference_params=inference_params,
            config=config,
            predict_keys=predict_keys,
            checkpoint=checkpoint,
            hooks=hooks,
            yield_single_examples=yield_single_examples)
    
    
    
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


class PyTorchModelManager(ModelManager):
    """Model manager for pytorch models"""

    def __init__(
            self,
            model,
            name="pytorch_model"):
        """Initialization keeps core of the model - inputs (from separate dataloader) 
        and model with necessary params. All other pieces are part of different graphs
        (ie, training, evaluation, prediction, inference)
        """
        self.name = model["name"]
        self.model_params = model.get("params", {})
        self.model_checkpoint = model.get("checkpoint")
        self.model_dir = model["model_dir"]

        # convert model
        pytorch_model = pytorch_net_fns[model["name"]]()
        
        def converted_model_fn(inputs, params):
            outputs = dict(inputs)
            seq = inputs[DataKeys.FEATURES]
            seq = tf.squeeze(seq)
            
            max_batch_size = seq.get_shape().as_list()[0]
            if True:
                rna = np.ones((max_batch_size, 1630))
            else:
                rna = inputs["FEATURES.RNA"]

            # run model through pyfunc and set shape
            pytorch_inputs = [seq, rna, max_batch_size]
            Tout = tf.float32
            outputs[DataKeys.LOGITS] = tf.py_func(
                pytorch_model.output,
                pytorch_inputs,
                Tout,
                #stateful="False",
                name="pytorch_model_logits")
            outputs[DataKeys.LOGITS] = tf.reshape(
                outputs[DataKeys.LOGITS],
                (max_batch_size, 1))
            
            # also get gradients and set shape
            pytorch_inputs = [seq, rna, False, max_batch_size]
            outputs[DataKeys.IMPORTANCE_GRADIENTS] = tf.py_func(
                pytorch_model.importance_score,
                pytorch_inputs,
                Tout,
                #stateful="False",
                name="pytorch_model_gradients")
            outputs[DataKeys.IMPORTANCE_GRADIENTS] = tf.reshape(
                outputs[DataKeys.IMPORTANCE_GRADIENTS],
                seq.get_shape())
            outputs[DataKeys.IMPORTANCE_GRADIENTS] = tf.expand_dims(
                outputs[DataKeys.IMPORTANCE_GRADIENTS],
                axis=1)
            
            return outputs, params

        self.model_fn = converted_model_fn

        
def setup_model_manager(args):
    """wrapper function to make loading
    models from different sources consistent
    """
    if args.model_framework == "tensorflow":
        model_manager = ModelManager(model=args.model)
    elif args.model_framework == "keras":

        args.model["name"] = "keras_transfer"
        with open(args.transfer_keras) as fp:
            args.model_info = json.load(fp)
            model_manager = KerasModelManager(
                keras_model_path=args.model_info["checkpoint"],
                model_params=args.model_info.get("params", {}),
                model_dir=args.out_dir)
            args.transfer_model_checkpoint = model_manager.model_checkpoint
            warm_start_params = {}

        
        model_manager = KerasModelManager(model=args.model)
    elif args.model_framework == "pytorch":
        model_manager = PyTorchModelManager(model=args.model)
    else:
        raise ValueError, "unrecognized deep learning framework!"
    
    return model_manager
