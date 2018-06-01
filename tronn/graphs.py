"""Contains functions to make running tensorflow graphs easier
"""

import os
import logging
import h5py

import six

import numpy as np
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

from tronn.outlayer import H5Handler
from tronn.learn.evaluation import get_global_avg_metrics

from tronn.learn.learning_2 import RestoreHook

from tronn.interpretation.interpret import visualize_region
from tronn.interpretation.dreaming import dream_one_sequence

#from tronn.util.tf_utils import setup_tensorflow_session
#from tronn.util.tf_utils import close_tensorflow_session


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
                    print example_idx
                    new_sequence = dream_one_sequence(
                        np.expand_dims(array[example_idx][:], axis=0),
                        mon_sess,
                        feed_dict,
                        predictions,
                        max_iter=1,
                        num_bp_per_iter=10)
                    yield new_sequence

                    
    def build_restore_graph_function(self, checkpoints, is_ensemble=False, skip=[], scope_change=None):
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

    
    def restore_graph(self, sess, checkpoints, is_ensemble=False, skip=[], scope_change=None):
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

        # TODO fix the train op so that doesn't rely on slim
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
        # assertions
        assert inputs.get(features_key) is not None
        
        # set up prediction dataflow
        outputs = self.build_prediction_dataflow(inputs, features_key=features_key)

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
            # note that all input goes through features (including labels)
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
        # build estimator and train
        estimator = self.build_estimator(config=config, out_dir=out_dir)
        estimator.train(input_fn=input_fn, max_steps=steps, hooks=hooks)
        
        return tf.train.latest_checkpoint(out_dir)


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
        # build evaluation estimator and evaluate
        estimator = self.build_estimator(config=config, out_dir=out_dir)
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
        estimator = self.build_estimator(config=config, out_dir=out_dir)

        # return prediction generator
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
            "checkpoint": checkpoint,
            "inference_mode": True,
            "inference_fn": inference_fn,
            "inference_params": inference_params} # TODO pwms etc etc
        
        # build estimator
        estimator = self.build_estimator(
            params=params,
            config=config,
            out_dir=out_dir)

        # return generator
        return estimator.infer(
            input_fn=input_fn,
            checkpoint_path=checkpoint)

    
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
            feed_dict)

    
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
            if best_metric_val is None or ('loss' in early_stopping_metric) != (
                    eval_metrics[early_stopping_metric] > best_metric_val):
                best_metric_val = eval_metrics[early_stopping_metric]
                consecutive_bad_epochs = 0
                best_checkpoint = latest_checkpoint
                with open(os.path.join(out_dir, 'best.log'), 'w') as fp:
                    fp.write('epoch %d\n'%epoch)
                    fp.write("checkpoint path: {}\n".format(best_checkpoint))
                    fp.write(str(eval_metrics))
            else:
                # break if consecutive bad epochs are too high
                consecutive_bad_epochs += 1
                if consecutive_bad_epochs > epoch_patience:
                    logging.info(
                        "early stopping triggered on epoch {} with patience {}".format(epoch, epoch_patience))
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
        if False:
            # set up metrics and values
            metric_values, metric_updates = metrics_fn(
                labels, probs)
            for update in metric_updates: tf.add_to_collection(
                tf.GraphKeys.UPDATE_OPS, update)

        metric_map = metrics_fn(labels, probs)

        return metric_map


    def _add_summaries(self):
        """add things you want to track on tensorboard
        """
        return None

    
    @staticmethod
    def infer_and_save_to_h5(generator, h5_file, sample_size):
        """wrapper routine to run inference and save the results out
        """
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
            #total_visualized = 0
            #passed_cutoff = 0 # debug
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
    
