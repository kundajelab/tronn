"""description: contains adjusted Estimator
"""

import time

import tensorflow as tf

import six

from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.training import training


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
            if False:
                print "USING QUEUES FOR DATA LOADING"
                features, input_hooks = self._get_features_from_input_fn(
                    input_fn, model_fn_lib.ModeKeys.PREDICT)
            else:
                # this is to make sure the dataset gets initialized
                iterator = input_fn().make_one_shot_iterator()
                features, _ = iterator.get_next()
                input_hooks = []
            
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
                init_time = time.time()
                batches_run = 0
                warmup_num = 10
                while not mon_sess.should_stop():
                    if batches_run == warmup_num:
                        init_time = time.time() # warmup
                    preds_evaluated = mon_sess.run(predictions)
                    new_time = time.time()
                    batches_run += 1
                    if warmup_num != batches_run:
                        time_per_batch = (new_time - init_time) / (batches_run - warmup_num)
                    #time_per_batch = new_time - init_time
                    print "run session {}".format(time_per_batch)
                    #init_time = new_time
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
