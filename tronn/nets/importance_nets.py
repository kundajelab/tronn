"""Description: graphs that transform importance scores to other representations
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.nets.sequence_nets import generate_dinucleotide_shuffles
from tronn.nets.sequence_nets import pad_data
from tronn.nets.sequence_nets import generate_scaled_inputs

from tronn.nets.filter_nets import rebatch
from tronn.nets.filter_nets import filter_and_rebatch

from tronn.nets.sequence_nets import remove_shuffles
from tronn.nets.sequence_nets import clear_shuffles

from tronn.nets.threshold_nets import threshold_shufflenull

from tronn.nets.normalization_nets import normalize_to_weights
from tronn.nets.normalization_nets import normalize_to_weights_w_shuffles
from tronn.nets.normalization_nets import normalize_to_absolute_one

from tronn.nets.threshold_nets import clip_edges

from tronn.util.utils import DataKeys


class FeatureImportanceExtractor(object):
    """Follows a kind of tf Hooks framework - do things before, during, after model run"""
    
    def __init__(
            self,
            model_fn,
            feature_type="sequence",
            num_shuffles=7,
            keep_shuffles=True):
        """initialize object
        """
        self.model_fn = model_fn
        self.num_shuffles = num_shuffles
        self.keep_shuffles = keep_shuffles
        self.feature_type = feature_type
        
        
    def preprocess(self, inputs, params):
        """assertions and any feature preprocessing
        """
        # run some generic assertions that apply to all here
        assert inputs.get(DataKeys.FEATURES) is not None

        # generate shuffles if sequence
        if self.feature_type == "sequence":
            
            # save out the original sequence
            inputs[DataKeys.ORIG_SEQ] = inputs[DataKeys.FEATURES]

            # update params
            params.update({"aux_key": DataKeys.ORIG_SEQ_SHUF})
            params.update({"num_shuffles": self.num_shuffles})
            
            # TODO add assert for is sequence
            inputs, params = generate_dinucleotide_shuffles(inputs, params)

            # and attach the shuffles (aux key points to ORIG_SEQ_SHUF)
            inputs, params = attach_auxiliary_tensors(inputs, params)

        return inputs, params
            
            
    def run_model_and_get_anchors(self, inputs, params, layer_key):
        """run the model fn with preprocessed inputs
        """
        reuse = params.get("model_reuse", True)
        with tf.variable_scope("", reuse=reuse):
            logging.info("Calling model fn")
            model_outputs, params = self.model_fn(inputs, params)
        
        # gather anchors
        inputs[DataKeys.IMPORTANCE_ANCHORS] = tf.gather(
            model_outputs[layer_key],
            params["importance_task_indices"],
            axis=1)
        
        return inputs, params

        
    def get_singletask_feature_importances(inputs, params):
        """implement in child class
        """
        raise NotImplementedError, "implement in derived class!"

    
    @classmethod
    def get_multitask_feature_importances(cls, inputs, params):
        """gets feature importances from multiple tasks
        can be called standalone (in an inference stack for example)
        """
        outputs = dict(inputs)
        anchors = inputs[DataKeys.IMPORTANCE_ANCHORS]

        # NOTE: cannot use map_fn here, breaks the connection to the gradients
        importances = []
        gradients = []
        for task_idx in xrange(anchors.get_shape().as_list()[1]):
            inputs_anchor = anchors[:,task_idx]
            inputs.update({"anchor": inputs_anchor})
            results, _ = cls.get_singletask_feature_importances(inputs, params)
            importances.append(results[DataKeys.FEATURES])
            gradients.append(results[DataKeys.IMPORTANCE_GRADIENTS])
        importances = tf.concat(importances, axis=1)
        gradients = tf.concat(gradients, axis=1)
        
        # save out
        outputs[DataKeys.FEATURES] = importances
        outputs[DataKeys.WEIGHTED_SEQ] = importances
        outputs[DataKeys.IMPORTANCE_GRADIENTS] = gradients
        
        return outputs, params
        
        
    def postprocess(
            self,
            inputs,
            params,
            keep_shuffle_keys=[
                DataKeys.ORIG_SEQ,
                DataKeys.WEIGHTED_SEQ,
                DataKeys.LOGITS]):
        """post processing 
        """
        default_params = {
            "pval_thresh": 0.01,
            "filter_window": 7,
            "filter_window_fract": float(2)/7,
            "normalize_weight_key": DataKeys.LOGITS,
            "left_clip": 420,
            "right_clip": 580,
            "required_nonzero_bp": 10}
        
        from tronn.nets.inference_nets import build_inference_stack
        
        if self.feature_type == "sequence":

            params.update({"num_aux_examples": self.num_shuffles})
            params.update({"rebatch_name": "detach_dinuc_seq"})
            
            # update params
            params.update(
                {"keep_shuffles": self.keep_shuffles,
                 "keep_shuffle_keys": keep_shuffle_keys,
                 "process_shuffles": True})

            # is it better to
            # 1) detach shuffles
            # 2) threshold by shuffled null
            # 3) reattach shuffles
            # 4) filter singles
            # 5) normalize to weights
            # 6) clip edges
            # 7) detach shuffles
            # 8) filter
            
            # postprocess stack
            inference_stack = [
                # TODO figure out if there's a cleaner way to manage shuffles
                (remove_shuffles, {}), # move shuffles out of the way
                (threshold_shufflenull, {"pval_thresh": 0.01, "shuffle_key": DataKeys.WEIGHTED_SEQ_SHUF}),
                (filter_singles_twotailed_w_shuffles, {"window": 7, "min_fract": float(2)/7}), # this is losing p63?
                (normalize_to_weights_w_shuffles, {"weight_key": DataKeys.LOGITS}), # do this after clipping weights?
                (clip_edges, {"left_clip": 420, "right_clip": 580}),

                #(clear_shuffles, {}),
                (filter_by_importance, {"cutoff": 10, "positive_only": True}) # TODO move this out?

            ]

            # build the postproces stack
            outputs, params = build_inference_stack(
                inputs, params, inference_stack)

            # save out updates to weighted seq key
            outputs[DataKeys.WEIGHTED_SEQ_ACTIVE] = outputs[DataKeys.FEATURES]
            
        return outputs, params


    def extract(self, inputs, params):
        """put all the pieces together
        """
        layer_key = params.get("layer_key", DataKeys.LOGITS)
        
        outputs, params = self.preprocess(inputs, params)
        outputs, params = self.run_model_and_get_anchors(outputs, params, layer_key)
        outputs, params = self.get_multitask_feature_importances(outputs, params)
        outputs, params = self.postprocess(outputs, params)
        
        return outputs, params

    

    def extract_old(self, layer_key=DataKeys.LOGITS):
        """put all the pieces together
        """
        self.preprocess()
        self.run_model_and_get_anchors(layer_key)
        self.outputs, _ = self.get_multitask_feature_importances(self.inputs, self.params)
        results, params = self.postprocess()
        
        return results, params

        
class InputxGrad(FeatureImportanceExtractor):
    """Runs input x grad"""

    @staticmethod
    def get_singletask_feature_importances(inputs, params):

        # inputs
        features = inputs[DataKeys.FEATURES]
        anchor = inputs["anchor"]
        outputs = dict(inputs)
        
        # gradients
        [feature_gradients] = tf.gradients(anchor, [features])
        # [feature_grad] = tf.gradients(anchor, [features], grad_ys=params["grad_ys"])
            
        # input x grad
        results = tf.multiply(features, feature_gradients, 'input_x_grad')

        outputs[DataKeys.FEATURES] = results
        outputs[DataKeys.IMPORTANCE_GRADIENTS] = feature_gradients

        return outputs, params

    
class IntegratedGradients(InputxGrad):
    """Run integrated gradients"""

    def preprocess(self):
        """generate scaled inputs
        """
        # call super (generate shuffles etc)
        super(IntegratedGradients, self).preprocess()
        
        # and then make scaled input steps here
        self.inputs, self.params = generate_scaled_inputs(self.inputs, self.params)

        return self.inputs, self.params

    
    @staticmethod
    def get_singletask_feature_importances(inputs, params):
        """call input x grad and then integrate
        """
        assert params.get("num_scaled_inputs") is not None
        
        # params
        num_scaled_inputs = params["num_scaled_inputs"]

        # input x grad
        outputs, params = InputxGrad.get_singletask_feature_importances(inputs, params)

        # integrate
        integrations = []
        for i in xrange(0, batch_size, num_scaled_inputs):
            print i
            integrated = tf.reduce_mean(
                outputs[DataKeys.FEATURES][i:i+num_scaled_inputs], axis=0, keep_dims=True)
            integrations.append(integrated)

        outputs["features"] = tf.concat(integrations, axis=0)

        return outputs, params
        

    def postprocess(self):
        """remove steps
        """
        # remove steps here
        self.params.update(
                {"keep_shuffles": self.keep_shuffles,
                 "keep_shuffle_keys": keep_shuffle_keys})

        shuffle_params = {"num_shuffles": self.params["num_scaled_inputs"],
                          "keep_shuffles": False,
                          "keep_shuffle_keys": []}
        self.outputs, _ = remove_shuffles(self.outputs, shuffle_params)
        
        # and then call super
        self.outputs, self.params = super(IntegratedGradients, self).postprocess()

        return self.outputs, self.params


class DeepLift(FeatureImportanceExtractor):
    """deep lift for sequence"""


    @classmethod
    def get_diff_from_ref(features, shuffle_num=7):
        """Get the diff from reference, but maintain batch
        only remove shuffles at the end of the process. For deeplift
        """
        batch_size = features.get_shape().as_list()[0]
        assert batch_size % (shuffle_num + 1) == 0

        example_num = batch_size / (shuffle_num + 1)

        # unstack to get diff from ref
        features = [tf.expand_dims(example, axis=0)
                    for example in tf.unstack(features, axis=0)]
        for i in xrange(example_num):
            idx = (shuffle_num + 1) * (i)
            actual = features[idx]
            references = features[idx+1:idx+shuffle_num+1]
            diff = tf.subtract(actual, tf.reduce_mean(references, axis=0))
            features[idx] = diff

        # restack
        features = tf.concat(features, axis=0)

        return features

    
    @classmethod
    def build_deeplift_multiplier(cls, x, y, multiplier=None, shuffle_num=7):
        """Takes input and activations to pass down
        """
        # TODO figure out how to use sets for name matching
        linear_names = set(["Conv", "conv", "fc"])
        nonlinear_names = set(["Relu", "relu"])

        if "Relu" in y.name:
            # rescale rule
            delta_y = cls.get_diff_from_ref(y, shuffle_num=shuffle_num)
            delta_x = cls.get_diff_from_ref(x, shuffle_num=shuffle_num)
            multiplier = tf.divide(
                delta_y, delta_x)
        elif "Conv" in y.name or "fc" in y.name:
            # linear rule
            [weights] = tf.gradients(y, x, grad_ys=multiplier)
            #delta_x = get_diff_from_ref(x, shuffle_num=shuffle_num)
            #multiplier = tf.multiply(
            #    weights, delta_x)
            multiplier = weights
        else:
            # TODO implement reveal cancel rule?
            print y.name, "not recognized"
            quit()

        return multiplier


    @classmethod
    def get_singletask_feature_importances(cls, inputs, params):
        """run deeplift on a single anchor
        """
        
        # inputs
        features = inputs[DataKeys.FEATURES]
        anchor = inputs["anchor"]
        outputs = dict(inputs)
        activations = tf.get_collection("DEEPLIFT_ACTIVATIONS")
        activations = [features] + activations

        # go backwards through the variables
        activations.reverse()
        for i in xrange(len(activations)):
            current_activation = activations[i]

            if i == 0:
                previous_activation = tf.identity(
                    anchor, name="fc.anchor")
                multiplier = None
            else:
                previous_activation = activations[i-1]

            # function here to build multiplier and pass down
            multiplier = cls.build_deeplift_multiplier(
                current_activation,
                previous_activation,
                multiplier=multiplier,
                shuffle_num=shuffle_num)
            
        outputs[DataKeys.FEATURES] = multiplier
        
        return outputs, params


class DeltaFeatureImportanceMapper(InputxGrad):
    """DFIM - given mutational results, get importance scores 
    and return the delta results"""

    def preprocess(self, inputs, params):
        """preprocess for delta features
        """
        assert inputs.get(DataKeys.FEATURES) is not None
        assert inputs.get(DataKeys.MUT_MOTIF_SEQ) is not None
        
        # don't generate shuffles, assume that the shuffles are still attached from before
        
        # attach the mutations (interleaved)
        orig_features = inputs[DataKeys.FEATURES] # {N, 1, 1000, 4}
        orig_features = tf.expand_dims(orig_features, axis=1) # {N, 1, 1, 1000, 4}
        mut_features = inputs[DataKeys.MUT_MOTIF_SEQ] # {N, mut_M, 1, 1000, 4}
        features = tf.concat([orig_features, mut_features], axis=1) # {N, 1+mut_M, 1, 1000, 4}
        features = tf.reshape(features, [-1]+orig_features.get_shape().as_list()[2:]) # {N*(1+mut_M), 1, 1000, 4}
        inputs[DataKeys.FEATURES] = features
        
        # track which parts of the batch are what
        params["num_shuffles"] = inputs[DataKeys.MUT_MOTIF_SEQ].get_shape().as_list()[1]
        batch_size = params["num_shuffles"] + 1
        params["batch_size"] = orig_features.get_shape().as_list()[0]
        
        # and pad everything
        outputs, _ = pad_data(
            inputs,
            {"num_shuffles": params["num_shuffles"],
             "ignore": [DataKeys.FEATURES]})

        # and rebatch
        outputs, _ = rebatch(outputs, {"name": "rebatch_dfim", "batch_size": batch_size})

        return outputs, params

    
    def postprocess(self, inputs, params):
        """postprocess
        """

        from tronn.nets.inference_nets import build_inference_stack

        # TODO need to calculate the delta logits (but also keep the actual logits too)
        # use delta logits to get statistical sig on whether mutation had an effect
        # {N, mutM, logit} - test is for every mutM for every logit (not in NN)
        # remember to first grab the subset where the motif is actually present
        # {N, mutM} this means run a for loop across the mutM
        # result: {muM, logit}
        
        # postprocess stack
        inference_stack = [
            (threshold_shufflenull, {"pval_thresh": 0.01, "shuffle_key": DataKeys.WEIGHTED_SEQ_SHUF}),
            
            (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}),
            (normalize_to_absolute_one, {}),
            (clip_edges, {"left_clip": 420, "right_clip": 580}),

            (remove_shuffles, {})

            # importance filter? to remove examples w no change?
            # but may be important to keep these for statistics
        ]

        # HERE - calculate the delta

        # and after calculating the delta, normalize the delta
        # to the diff in the logits
        
        # build the postproces stack
        outputs, params = build_inference_stack(
            inputs, params, inference_stack)

        print outputs[DataKeys.FEATURES]
        
        # Q: is there a way to get the significance of a delta score even here?
        # ie, what is the probability of a delta score by chance?
        # could build a distribution?

        # this returns the delta importances as features

        return outputs, params
    
    
    
def get_task_importances(inputs, params):
    """per desired task, get the importance scores
    
    methods:
    input x grad
    integrated gradients
    deeplift
    saturation mutagenesis

    """
    backprop = params["backprop"]
    
    # all this should be is a wrapper
    if backprop == "input_x_grad":
        extractor = InputxGrad(params["model_fn"])
    elif backprop == "integrated_gradients":
        extractor = IntegratedGradients(params["model_fn"])
    elif backprop == "deeplift":
        extractor = DeepLift(params["model_fn"])
    elif backprop == "saturation_mutagenesis":
        pass
    else:
        # TODO switch to exception
        print "method does not exist/not yet implemented"
        quit()
    
    outputs, params = extractor.extract(inputs, params)

    return outputs, params


def run_dfim(inputs, params):
    """wrapper for functional calls
    """
    extractor = DeltaFeatureImportanceMapper(params["model_fn"])

    outputs, params = extractor.extract(inputs, params)
    
    return outputs, params



def filter_singles(inputs, params):
    """Filter out singlets to remove noise
    """
    # features
    features = inputs[DataKeys.FEATURES]
    outputs = dict(inputs)

    # params
    window = params.get("window", 7)
    min_features = params.get("min_fract", float(2)/7)

    # binarize features and get fraction in windows
    features_present = tf.cast(tf.not_equal(features, 0), tf.float32)
    feature_counts_in_window = tf.layers.average_pooling2d(
        features_present, [1, window], [1,1], padding="same")

    # and mask
    feature_mask = tf.cast(tf.greater_equal(feature_counts_in_window, min_features), tf.float32)
    features = tf.multiply(features, feature_mask)
    outputs[DataKeys.FEATURES] = features
    
    return outputs, params


def filter_singles_twotailed(inputs, params):
    """Filter out singlets, removing positive and negative singlets separately
    """
    assert inputs.get(DataKeys.FEATURES) is not None
    
    # get features
    features = inputs[DataKeys.FEATURES]
    outputs = dict(inputs)
    
    # split features
    pos_features = tf.cast(tf.greater(features, 0), tf.float32)
    neg_features = tf.cast(tf.less(features, 0), tf.float32)

    # get masks
    pos_mask = filter_singles({DataKeys.FEATURES: pos_features}, params)[0][DataKeys.FEATURES]
    neg_mask = filter_singles({DataKeys.FEATURES: neg_features}, params)[0][DataKeys.FEATURES]
    keep_mask = tf.add(pos_mask, neg_mask)

    # mask features
    features = tf.multiply(features, keep_mask)

    # TODO - what do I use this for?
    # output for later
    num_positive_features = tf.reduce_sum(
        tf.cast(
            tf.greater(
                tf.reduce_max(features, axis=[1,3]), [0]),
            tf.float32), axis=1, keepdims=True)

    # save desired outputs
    outputs[DataKeys.FEATURES] = features
    outputs["positive_importance_bp_sum"] = num_positive_features

    return outputs, params


def filter_singles_twotailed_w_shuffles(inputs, params):
    """also filter the singles out of the shuffles
    """
    # first run the normal one
    outputs, _ = filter_singles_twotailed(inputs, params)

    # and then also filter the shuffles
    shuffles = inputs[DataKeys.WEIGHTED_SEQ_SHUF]
    shuf_shape = shuffles.get_shape().as_list()
    shuffles = tf.reshape(
        shuffles,
        [shuf_shape[0],
         shuf_shape[1]*shuf_shape[2],
         shuf_shape[3],
         shuf_shape[4]])
    
    inputs.update({DataKeys.FEATURES: shuffles})
    shuffles = filter_singles_twotailed(
        inputs, params)[0][DataKeys.FEATURES]
    outputs[DataKeys.WEIGHTED_SEQ_SHUF] = tf.reshape(shuffles, shuf_shape)
    
    return outputs, params


def filter_by_importance(inputs, params):
    """Filter out low importance examples, not interesting
    """
    # features
    features = inputs[DataKeys.FEATURES]
    
    # params
    cutoff = params.get("cutoff", 20)
    positive_only = params.get("positive_only", False)

    if positive_only:
        # get condition mask
        feature_sums = tf.reduce_max(
            tf.reduce_sum(
                tf.cast(tf.greater(features, 0), tf.float32),
                axis=[2, 3]),
            axis=1) # shape {N}
    else:
        # get condition mask
        feature_sums = tf.reduce_max(
            tf.reduce_sum(
                tf.cast(tf.not_equal(features, 0), tf.float32),
                axis=[2, 3]),
            axis=1) # shape {N}

    inputs["condition_mask"] = tf.greater(feature_sums, cutoff)
    params.update({"name": "importances_filter"})
    outputs, params = filter_and_rebatch(inputs, params)
    
    return outputs, params
