"""functions to support learning
"""

import logging

import numpy as np
import tensorflow as tf

from tronn.util.tf_ops import restore_variables_op


class DataSetupHook(tf.train.SessionRunHook):
    """Hook to initialize tf.data.Dataset initializer"""
    
    def after_create_session(self, session, coord):
        
        initialize_ops = tf.get_collection("DATASETUP")
        staging_ops = tf.get_collection("STAGING_OPS")
        print len(initialize_ops)
        #session.run(initialize_ops)
        for i, op in enumerate(initialize_ops):
            print i
            session.run(initialize_ops[i])
            session.run(staging_ops[:i])
        print "initialized datasets"


class DataSetupHook_NEW(tf.train.SessionRunHook):
    """Hook to initialize tf.data.Dataset initializer"""
    
    def after_create_session(self, session, coord):
        staging_ops = tf.get_collection("STAGING_OPS")
        print staging_ops
        for i, op in enumerate(staging_ops):
            session.run(staging_ops[i])
        print "set up stages"

        

class DataCleanupHook(tf.train.SessionRunHook):
    """Hook to cleanup data threads as needed"""
        
    def end(self, session):
        close_ops = tf.get_collection("DATACLEANUP")
        session.run(close_ops)


class RestoreHook(tf.train.SessionRunHook):
    """Hook that runs initialization functions at beginning of session"""

    def __init__(self, warm_start, warm_start_params):
        self.warm_start = warm_start
        self.skip = warm_start_params.get("skip", [])
        self.include_scope = warm_start_params.get("include_scope", "")
        self.scope_change = warm_start_params.get("scope_change", None)
        #self.init_assign_op = init_assign_op
        #self.init_feed_dict = init_feed_dict

        
    def begin(self):
        """set up the init functions and feed dict
        """
        self.init_assign_op, self.init_feed_dict = restore_variables_op(
            self.warm_start,
            skip=self.skip,
            include_scope=self.include_scope,
            scope_change=self.scope_change)

        
    def after_create_session(self, session, coord=None):
        """Restore the model from checkpoint, given params above
        """
        session.run(self.init_assign_op, self.init_feed_dict)
        print "RESTORED FROM CHECKPOINT"
        #print session.run(tf.trainable_variables()[0])[:,:,0]


        
class KerasRestoreHook(tf.train.SessionRunHook):
    """Hook that restores Keras saved model variables"""

    def after_create_session(self, session, coord=None):
        init_ops = tf.get_collection("KERAS_INIT")
        session.run(init_ops)

        
class EarlyStoppingHook(tf.train.SessionRunHook):
    """Hook that requests stop based on early stopping criteria"""

    def __init__(self, monitor="mean_auprc", min_delta=0, patience=0, mode="auto"):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode, RuntimeWarning)
            mode = 'auto'
        
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
            
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def begin(self):
        # Convert names to tensors if given
        graph = tf.get_default_graph()

        self.monitor = tf.get_collection("auprc")[0]
        print self.monitor
        
        #self.monitor = graph.as_graph_element(self.monitor)
        if isinstance(self.monitor, tf.Operation):
            self.monitor = self.monitor.outputs[0]
                
    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs(self.monitor)
    
    def after_run(self, run_context, run_values):
        current = run_values.results
        
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                run_context.request_stop()


# TODO DON'T DELETE YET
# port into manager to run continuous evaluation
# and stop when loss stops dropping
def train_and_evaluate_with_early_stopping_OLD(
        model_manager,
        train_input_fn,
        eval_input_fn,
        metrics_fn=None, # TODO adjust this to just get loss?
        out_dir=".",
        warm_start=None):
    """wrapper for training
    """
    # run config
    run_config = tf.estimator.RunConfig(
        save_summary_steps=30,
        save_checkpoints_secs=None,
        save_checkpoints_steps=10000000000,
        keep_checkpoint_max=None) # TODO add summary here?

    # set up estimator
    estimator = model_manager.build_estimator(
        config=run_config,
        params={
            "optimizer_fn": tf.train.RMSPropOptimizer,
            "optimizer_params": {
                "learning_rate": 0.002,
                "decay": 0.98,
                "momentum": 0.0}},
        out_dir=out_dir)

    # restore from transfer as needed
    restore_hook = RestoreHook(
        warm_start,
        skip=["logit"],
        include_scope="",
        scope_change=["", "basset/"])

    # train until input producers are out
    estimator.train(
        input_fn=train_input_fn,
        max_steps=None,
        hooks=[restore_hook])

    quit()

    # hooks
    early_stopping_hook = EarlyStoppingHook(
        monitor="mean_auprc",
        patience=2)

    # TODO set up warm start hook here
    
    # set up estimator specs
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=442648+100,
        hooks=[restore_hook])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=100,
        start_delay_secs=60000000,
        throttle_secs=60000000,
        hooks=[early_stopping_hook])

    # utilize train and evaluate functionality
    tf.logging.set_verbosity("DEBUG")
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec)
    
    return None
