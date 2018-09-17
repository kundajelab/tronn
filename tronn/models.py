"""Contains functions to make running tensorflow graphs easier
"""

import os
import json
import logging
import h5py

import six

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed

from tensorflow.python.keras import models

from tensorflow.python.estimator.keras import _clone_and_build_model as build_keras_model

from tensorflow.python.training import monitored_session
from tensorflow.python.training import training

from tronn.util.tf_ops import restore_variables_op
from tronn.util.tf_ops import class_weighted_loss_fn
from tronn.util.tf_ops import positives_focused_loss_fn

from tronn.util.tf_utils import print_param_count

from tronn.outlayer import H5Handler

from tronn.learn.evaluation import get_global_avg_metrics
from tronn.learn.evaluation import get_regression_metrics

from tronn.learn.learning import RestoreHook
from tronn.learn.learning import KerasRestoreHook
from tronn.learn.learning import DataCleanupHook

from tronn.interpretation.interpret import visualize_region
from tronn.interpretation.dreaming import dream_one_sequence

from tronn.visualization import visualize_debug

from tronn.util.utils import DataKeys


# TODO move into learn.estimator
class TronnEstimator(tf.estimator.Estimator):
    """Extended estimator to have extra capabilities"""

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
                print "session created"
                while not mon_sess.should_stop():
                    preds_evaluated = mon_sess.run(predictions)
                    #for key in preds_evaluated.keys():
                    #    print key, preds_evaluated[key].shape
                    print "run session"
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

                            
    def dream_generator(
            self,
            array,
            input_fn,
            feed_dict,
            predict_keys=None,
            dream_key="dream.results",
            hooks=[],
            checkpoint_path=None):
        """given array of onehot sequences, run dream
        """
        num_examples = array.shape[0]

        # set up graph
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
                # run through examples
                for example_idx in xrange(num_examples):
                    preds_evaluated = dream_one_sequence(
                        np.expand_dims(array[example_idx][:], axis=0),
                        mon_sess,
                        feed_dict,
                        predictions,
                        dream_key,
                        max_iter=1,
                        num_bp_per_iter=10)
                    yield preds_evaluated



    # TODO keep this for now as starting code for ensembling
    def build_restore_graph_function_old(self, checkpoints, is_ensemble=False, skip=[], scope_change=None):
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
                        checkpoints[i],
                        skip=skip,
                        include_scope=new_scope,
                        scope_change=["", new_scope])
                    sess.run(init_assign_op, init_feed_dict)
        else:
            print checkpoints
            if len(checkpoints) > 0:
                init_assign_op, init_feed_dict = restore_variables_op(
                    checkpoints[0], skip=skip, scope_change=scope_change)
                def restore_function(sess):
                    sess.run(init_assign_op, init_feed_dict)
            else:
                print "WARNING NO CHECKPOINTS USED"
                
        return restore_function

    
    def restore_graph_old(self, sess, checkpoints, is_ensemble=False, skip=[], scope_change=None):
        """restore saved model from checkpoint into sess
        """
        restore_function = self.build_restore_graph_function(
            checkpoints, is_ensemble=is_ensemble, skip=skip, scope_change=scope_change)
        restore_function(sess)
        
        return None


class ModelManager(object):
    """Manages the full model pipeline (utilizes Estimator framework)"""
    
    def __init__(
            self,
            model_fn,
            model_params,
            model_checkpoint=None,
            model_dir=None):
        """Initialization keeps core of the model - inputs (from separate dataloader) 
        and model with necessary params. All other pieces are part of different graphs
        (ie, training, evaluation, prediction, inference)
        """
        self.model_fn = model_fn
        self.model_params = model_params
        self.model_checkpoint = model_checkpoint
        self.model_dir = model_dir
        
        
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
            regression=False):
        """builds the training dataflow. links up input tensors
        to the model with is_training as True.
        """
        # assertions
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None

        # build model in training mode
        self.model_params.update({"is_training": True})
        outputs, _ = self.model_fn(inputs, self.model_params)

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
            regression=False):
        """build evaluation dataflow. links up input tensors
        to the model with is_training as False
        """
        # assertions
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None

        # build model
        self.model_params.update({"is_training": False})
        outputs, _ = self.model_fn(inputs, self.model_params)

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
            regression=False):
        """build prediction dataflow. links up input tensors
        to the model with is_training as False
        """
        # assertions
        assert inputs.get(features_key) is not None

        # build model
        self.model_params.update({"is_training": False})
        outputs, _ = self.model_fn(inputs, self.model_params)

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
            regression=False):
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
            regression=regression)

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
                    outputs = self.build_prediction_dataflow(inputs, regression=regression)
                    return tf.estimator.EstimatorSpec(mode, predictions=outputs)
                else:
                    # inference mode
                    inference_params = params.get("inference_params", {})
                    model_reuse = params.get("model_reuse", False)
                    inference_params.update({"model_reuse": model_reuse})
                    outputs, variables_to_restore = self.build_inference_dataflow(
                        inputs,
                        params["inference_fn"],
                        inference_params)
                    
                    # create custom init fn
                    init_op, init_feed_dict = restore_variables_op(
                        params["checkpoint"],
                        skip=["pwm"])
                    def init_fn(scaffold, sess):
                        sess.run(init_op, init_feed_dict)

                    # custom scaffold to load checkpoint
                    scaffold = monitored_session.Scaffold(
                        init_fn=init_fn)
                    
                    return tf.estimator.EstimatorSpec(
                        mode,
                        predictions=outputs,
                        scaffold=scaffold)
            
            elif mode == tf.estimator.ModeKeys.EVAL:
                # evaluation mode
                outputs, loss, metrics = self.build_evaluation_dataflow(inputs, regression=regression)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            
            elif mode == tf.estimator.ModeKeys.TRAIN:
                # training mode
                outputs, loss, train_op = self.build_training_dataflow(inputs, regression=regression)
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
            regression=False):
        """train an estimator. if steps is None, goes on forever or until input_fn
        runs out.
        """
        # if there is a data close fn, add in as hook
        hooks.append(DataCleanupHook())
        print "appended hook"
        
        # build estimator and train
        estimator = self.build_estimator(
            config=config, out_dir=out_dir, regression=regression)
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
            regression=False):
        """evaluate a trained estimator
        """
        # if there is a data close fn, add in as hook
        hooks.append(DataCleanupHook())
        print "appended hook"
        
        # build evaluation estimator and evaluate
        estimator = self.build_estimator(
            config=config, out_dir=out_dir, regression=regression)
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
            yield_single_examples=True):
        """predict on a trained estimator
        """
        # if there is a data close fn, add in as hook
        hooks.append(DataCleanupHook())
        print "appended hook"

        # build prediction estimator
        estimator = self.build_estimator(config=config, out_dir=out_dir)

        # return prediction generator
        return estimator.predict(
            input_fn=input_fn,
            checkpoint_path=checkpoint,
            hooks=hooks)

    
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
        # if there is a data close fn, add in as hook
        hooks.append(DataCleanupHook())
        print "appended hook"
        
        # build inference estimator
        self.model_params["inference_mode"] = True
        params = {
            "checkpoint": checkpoint,
            "inference_mode": True,
            "inference_fn": inference_fn,
            "inference_params": inference_params,
            "model_reuse": self.model_params.get(
                "model_reuse", True)} # TODO pwms etc etc
        
        # build estimator
        estimator = self.build_estimator(
            params=params,
            config=config,
            out_dir=out_dir)

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

    
    def train_and_evaluate_with_early_stopping(
            self,
            train_input_fn,
            eval_input_fn,
            out_dir,
            max_epochs=20,
            early_stopping_metric="mean_auprc",
            epoch_patience=2,
            eval_steps=1000,
            warm_start=None,
            warm_start_params={},
            regression=False,
            model_info={},
            early_stopping=True):
        """run full training loop with evaluation for early stopping
        """
        # adjust for regression
        if regression:
            early_stopping_metric = "mse"
        
        # set up stopping conditions
        stopping_log = "{}/stopping.log".format(out_dir)
        if os.path.isfile(stopping_log):
            # if there is a stopping log, restart using the info in the log
            with open(stopping_log, "r") as fp:
                start_epoch, best_metric_val, consecutive_bad_epochs, best_checkpoint = fp.readline().strip().split()
                start_epoch, consecutive_bad_epochs =  map(
                    int, [start_epoch, consecutive_bad_epochs])
                best_metric_val = float(best_metric_val)
                start_epoch += 1
            if consecutive_bad_epochs >= epoch_patience:
                # move start epochs to max epoch so training doesn't run
                start_epoch = max_epochs
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
                steps=None, # TODO here calculate steps?
                hooks=training_hooks,
                regression=regression)
            
            # eval
            eval_metrics = self.evaluate(
                eval_input_fn,
                "{}/eval".format(out_dir),
                steps=eval_steps,
                checkpoint=latest_checkpoint,
                regression=regression)

            # determine if epoch of training was good
            if best_metric_val is None:
                is_good_epoch = True
            elif early_stopping_metric in ["loss", "mse"]:
                is_good_epoch = eval_metrics[early_stopping_metric] < best_metric_val
            else:
                is_good_epoch = eval_metrics[early_stopping_metric] > best_metric_val

            # if good epoch, save out new metrics and model info
            if is_good_epoch:
                best_metric_val = eval_metrics[early_stopping_metric]
                consecutive_bad_epochs = 0
                best_checkpoint = latest_checkpoint
                with open(os.path.join(out_dir, 'best.log'), 'w') as fp:
                    fp.write('epoch %d\n'%epoch)
                    fp.write("checkpoint path: {}\n".format(best_checkpoint))
                    fp.write(str(eval_metrics))
                model_info["checkpoint"] = best_checkpoint
                with open("{}/model_info.json".format(out_dir), "w") as fp:
                    json.dump(model_info, fp, sort_keys=True, indent=4)
            else:
                # break if consecutive bad epochs are too high
                consecutive_bad_epochs += 1
                if consecutive_bad_epochs > epoch_patience:
                    logging.info(
                        "early stopping triggered "
                        "on epoch {} "
                        "with patience {}".format(epoch, epoch_patience))
                    if early_stopping:
                        break

            # save to stopping log
            with open(stopping_log, 'w') as out:
                out.write("{}\t{}\t{}\t{}".format(
                    epoch,
                    best_metric_val,
                    consecutive_bad_epochs,
                    best_checkpoint))

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
                batch_size=4096,
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


# TODO(dk)
def setup_model_manager():
    """wrapper to keep things consistent
    """

    return model_manager
