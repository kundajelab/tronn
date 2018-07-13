"""Description: graphs that transform importance scores to other representations
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.nets.sequence_nets import generate_dinucleotide_shuffles
from tronn.nets.sequence_nets import generate_scaled_inputs
from tronn.nets.sequence_nets import unpad_examples

from tronn.nets.filter_nets import rebatch
from tronn.nets.filter_nets import filter_and_rebatch

from tronn.nets.sequence_nets import remove_shuffles
from tronn.nets.sequence_nets import clear_shuffles

from tronn.nets.threshold_nets import threshold_shufflenull

from tronn.nets.normalization_nets import normalize_to_weights

from tronn.nets.threshold_nets import clip_edges



from tronn.util.utils import DataKeys


def input_x_grad(inputs, params):
    """Layer-wise Relevance Propagation (Batch et al), implemented
    as input * gradient (equivalence is demonstrated in deepLIFT paper,
    Shrikumar et al). Returns the raw scores, adjust/normalize as needed.
    
    Requires:
      features: {N, 1, seqlen, 4} the onehot sequence
      anchor: {N, 1} the desired anchor neurons (normally logits)
    
    Returns:
      Input tensor weighted by gradient backpropagation.
      optionally the gradients also
    """
    assert inputs.get("features") is not None
    assert params["is_training"] == False
    assert params.get("anchor") is not None

    # pull features and send all others to output
    features = inputs.get("features")
    outputs = dict(inputs)

    print features

    
    # params
    anchor = params.get("anchor")
    use_relu = params.get("relu", False)

    print anchor
    
    # gradients
    if params.get("grad_ys") is None:
        [feature_grad] = tf.gradients(anchor, [features])
    else:
        [feature_grad] = tf.gradients(anchor, [features], grad_ys=params["grad_ys"])


    print feature_grad
        
    # input x grad
    features = tf.multiply(features, feature_grad, 'input_x_grad')

    if use_relu:
        features = tf.nn.relu(features)

    # save out
    outputs["features"] = features
    
    # keep gradients if you'd like them
    if params.get("keep_gradients") is not None:
        outputs["gradients"] = feature_grad
        
    return outputs, params


def integrated_gradients(inputs, params):
    """Integrated gradients as proposed by Sundararajan 2017
    """
    # assertions
    assert params.get("num_scaled_inputs") is not None
    assert params.get("batch_size") is not None

    batch_size = params["batch_size"]
    num_scaled_inputs = params["num_scaled_inputs"]

    assert batch_size % num_scaled_inputs == 0

    # run input_x_grad
    outputs, params = input_x_grad(inputs, params)

    # for features, reduce mean according to the groups
    all_mean_features = []
    for i in xrange(0, batch_size, num_scaled_inputs):
        print i
        mean_features = tf.reduce_mean(
            outputs["features"][i:i+num_scaled_inputs], axis=0, keep_dims=True)
        all_mean_features.append(mean_features)

    outputs["features"] = tf.concat(all_mean_features, axis=0)

    # and remove the steps
    params.update({"ignore": ["features"]})
    outputs, params = unpad_examples(outputs, params)

    return features, labels, config


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


def build_deeplift_multiplier(x, y, multiplier=None, shuffle_num=7):
    """Takes input and activations to pass down
    """
    # TODO figure out how to use sets for name matching
    linear_names = set(["Conv", "conv", "fc"])
    nonlinear_names = set(["Relu", "relu"])

    if "Relu" in y.name:
        # rescale rule
        delta_y = get_diff_from_ref(y, shuffle_num=shuffle_num)
        delta_x = get_diff_from_ref(x, shuffle_num=shuffle_num)
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


def deeplift(features, labels, config, is_training=False):
    """Basic deeplift in raw tensorflow
    """
    assert is_training == False
    assert config.get("anchor") is not None
    anchor = config.get("anchor")
    shuffle_num = config.get("shuffle_num", 7)
    
    # deepnet needs to be adding activations (for EVERY layer)
    # to the DEEPLIFT_ACTIVATIONS collection
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
        multiplier = build_deeplift_multiplier(
            current_activation,
            previous_activation,
            multiplier=multiplier,
            shuffle_num=shuffle_num)
        
    features = multiplier
 
    return features, labels, config


# TODO deprecate
def multitask_importances(inputs, params):
    """Set up importances coming from multiple tasks
    """
    assert inputs.get("features") is not None, "Feature tensor does not exist"
    assert inputs.get("importance_logits") is not None, "Importance logits do not exist"
    assert params.get("importance_task_indices") is not None
    assert params["is_training"] == False
    
    # pull features and send the rest through
    features = inputs["features"]
    outputs = dict(inputs)

    # TODO call model fn again here, and explicitly grab the logits

    
    # params
    anchors = inputs.get("importance_logits")
    task_indices = params.get("importance_task_indices")
    backprop = params.get("backprop", "input_x_grad")
    
    if backprop == "input_x_grad":
        importances_fn = input_x_grad
    elif backprop == "integrated_gradients":
        importances_fn = integrated_gradients
    elif backprop == "deeplift":
        importances_fn = deeplift
    else:
        print "method does not exist/not implemented!"
        quit()

    print importances_fn
    
    # split out anchors by task
    anchors = [tf.expand_dims(tensor, axis=1)
               for tensor in tf.unstack(anchors, axis=1)] # {N, 1}
    
    # get task specific importances
    task_importances = []
    task_gradients = []
    for i in xrange(len(task_indices)):
        task_params = dict(params)
        anchor_idx = task_indices[i]
        task_params["anchor"] = anchors[anchor_idx]
        if params.get("keep_gradients") is not None:
            task_params["grad_ys"] = params["all_grad_ys"][i]
        task_outputs, task_params = importances_fn(inputs, task_params)
        task_importances.append(task_outputs["features"])
        
        if params.get("keep_gradients") is not None:
            task_gradients.append(task_outputs["gradients"])
            
    features = tf.concat(task_importances, axis=1) # {N, task, pos, C}
    
    outputs["features"] = features
    if params.get("keep_gradients") is not None:
        outputs["gradients"] = tf.reduce_mean(
            tf.concat(task_gradients, axis=1), axis=1, keep_dims=True)
    
    # unpad as needed
    if backprop == "integrated_gradients":
        # TODO switch to other function
        params.update({"ignore": ["features"]})
        outputs, params = unpad_examples(outputs, params)
    #elif backprop == "deeplift":
        # TODO - actually remove shuffles later, make dependent on shuffle null and data input loader
        #features, labels, config = remove_shuffles(features, labels, config)
        #features, labels, config = rebatch(features, labels, config)
        
    return outputs, params


# ABOVE IS OLD



class FeatureImportanceExtractor(object):
    """Follows a kind of tf Hooks framework - do things before, during, after model run"""
    
    def __init__(
            self,
            model_fn,
            inputs,
            params,
            feature_type="sequence",
            keep_shuffles=True):
        """initialize object
        """
        self.model_fn = model_fn
        self.inputs = inputs
        self.params = params
        self.keep_shuffles = keep_shuffles
        self.feature_type = feature_type
        
        
    def preprocess(self):
        """assertions and any feature preprocessing
        """
        # run some generic assertions that apply to all here
        assert self.inputs.get(DataKeys.FEATURES) is not None

        # generate shuffles if sequence
        if self.feature_type == "sequence":
            # add assert for is sequence
            self.inputs, self.params = generate_dinucleotide_shuffles(
                self.inputs, self.params)
            print self.inputs[DataKeys.FEATURES]

            
    def run_model_and_get_anchors(self, layer_key):
        """run the model fn with preprocessed inputs
        """
        with tf.variable_scope("", reuse=True):
            self.model_outputs, self.params = self.model_fn(self.inputs, self.params)

        # gather anchors
        self.inputs[DataKeys.IMPORTANCE_ANCHORS] = tf.gather(
            self.model_outputs[layer_key],
            self.params["importance_task_indices"],
            axis=1)

        
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
        # transpose anchors for  (num_tasks, batch_size)
        #anchors_transposed = tf.transpose(inputs[DataKeys.IMPORTANCE_ANCHORS])

        # NOTE: cannot use map_fn here, breaks the connection to the gradients
        importances = []
        for task_idx in xrange(anchors.get_shape().as_list()[1]):
            inputs_anchor = anchors[:,task_idx]
            inputs.update({"anchor": inputs_anchor})
            results, _ = cls.get_singletask_feature_importances(inputs, params)
            importances.append(results[DataKeys.FEATURES])
        importances = tf.concat(importances, axis=1)

        # save out
        outputs[DataKeys.FEATURES] = importances

        return outputs, params
        
        
    def postprocess(self, keep_shuffle_keys=[DataKeys.FEATURES, DataKeys.LOGITS]):
        """post processing 
        """
        from tronn.nets.inference_nets import build_inference_stack
        
        if self.feature_type == "sequence":

            # put multitask into axis 1
            #self.outputs[DataKeys.FEATURES] = tf.squeeze(self.outputs[DataKeys.FEATURES], axis=2)
            
            # update params
            self.params.update(
                {"keep_shuffles": self.keep_shuffles,
                 "keep_shuffle_keys": keep_shuffle_keys})

            # postprocess stack
            inference_stack = [
                (remove_shuffles, {}),
                (threshold_shufflenull, {"pval_thresh": 0.05}),
                (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}),
                (normalize_to_weights, {"weight_key": "logits"}),
                (clip_edges, {"left_clip": 420, "right_clip": 580}),
                (filter_by_importance, {"cutoff": 10, "positive_only": True}),
                (clear_shuffles, {})
            ]

            # build the postproces stack
            self.outputs, self.params = build_inference_stack(
                self.outputs, self.params, inference_stack)
            
        return self.outputs, self.params
    

    def extract(self, layer_key=DataKeys.LOGITS):
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
        outputs["gradients"] = feature_gradients

        return outputs, params

    
class IntegratedGradients(InputxGrad):
    """Run integrated gradients"""

    def preprocess(self):
        # call super (generate shuffles etc)
        super(IntegratedGradients, self).preprocess()
        
        # and then make steps here
        self.inputs, self.params = generate_scaled_inputs(self.inputs, self.params)


    def postprocess(self):
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

    

def get_task_importances(inputs, params):
    """per desired task, get the importance scores
    
    methods:
    input x grad
    integrated gradients
    deeplift
    saturation mutagenesis

    """
    assert inputs.get(DataKeys.FEATURES) is not None
    assert params.get("importance_task_indices") is not None
    
    backprop = params["backprop"]
    
    # all this should be is a wrapper
    if backprop == "input_x_grad":
        extractor = InputxGrad(params["model_fn"], inputs, params)
    elif backprop == "integrated_gradients":
        extractor = IntegratedGradients(params["model_fn"], inputs, params)
    elif backprop == "deeplift":
        extractor = DeepLift(params["model_fn"], inputs, params)
    elif backprop == "saturation_mutagenesis":
        pass
    else:
        # TODO switch to exception
        print "method does not exist/not yet implemented"
        quit()
    
    outputs, params = extractor.extract()

    return outputs, params




# OLD CODE BELOW?







# TODO is this deprecated?
def multitask_global_importance(features, labels, config, is_training=False):
    """Also get global importance. does a check to see that the feature is at least
    observed twice (count_thresh)
    """
    assert is_training == False
    append = config.get("append", True)
    count_thresh = config.get("count_thresh", 2)
    
    # per example, only keep positions that have been seen more than once
    features_by_example = [tf.expand_dims(tensor, axis=0)
                           for tensor in tf.unstack(features)] # {1, task, M}

    # for each example, get sum across tasks
    # TODO separate this out as different function
    masked_features_list = []
    for example_features in features_by_example:
        motif_counts = tf.reduce_sum(
            tf.cast(tf.not_equal(example_features, 0), tf.float32),
            axis=1, keep_dims=True) # sum across tasks {1, 1, M}
        #motif_max = tf.reduce_max(
        #    motif_counts, axis=[1, 3], keep_dims=True) # {1, 1, pos, 1}
        # then mask based on max position
        motif_mask = tf.cast(tf.greater_equal(motif_counts, count_thresh), tf.float32)
        # and mask
        masked_features = tf.multiply(motif_mask, example_features)
        masked_features_list.append(masked_features)

    # stack
    features = tf.concat(masked_features_list, axis=0)
    
    # TODO could do a max - min scoring system?
    reduce_type = config.get("reduce_type", "sum")

    if reduce_type == "sum":
        features_max = tf.reduce_sum(features, axis=1, keep_dims=True)
    elif reduce_type == "max":
        features_max = tf.reduce_max(features, axis=1, keep_dims=True)
    elif reduce_type == "mean":
        features_max = tf.reduce_mean(features, axis=1, keep_dims=True)

    # append or replace
    if append:
        features = tf.concat([features, features_max], axis=1)
    else:
        features = features_max

    # things to keep
    if config.get("keep_global_pwm_scores") is not None:
        # attach to config
        config["outputs"][config["keep_global_pwm_scores"]] = features_max #{N, pos, motif}
        config["keep_global_pwm_scores"] = None # TODO fix this, this is because there is overwriting
        
    return features, labels, config


# KEEP BELOW HERE

def filter_singles(inputs, params):
    """Filter out singlets to remove noise
    """
    # get features and pass rest through
    features = inputs["features"]
    outputs = dict(inputs)
    
    window = params.get("window", 5)
    min_features = params.get("min_fract", 0.4)
    
    features_present = tf.cast(tf.not_equal(features, 0), tf.float32)

    # check the windows
    feature_counts_in_window = slim.avg_pool2d(features_present, [1, window], stride=[1, 1], padding="SAME")
    feature_mask = tf.cast(tf.greater_equal(feature_counts_in_window, min_features), tf.float32)
    
    # wherever the window is 0, blank out that region
    features = tf.multiply(features, feature_mask)

    outputs["features"] = features
    
    return outputs, params




def filter_singles_twotailed(inputs, params):
    """Filter out singlets, removing positive and negative singlets separately
    """
    # get features and pass rest through
    features = inputs["features"]
    outputs = dict(inputs)
    
    # split features
    pos_features = tf.cast(tf.greater(features, 0), tf.float32)
    neg_features = tf.cast(tf.less(features, 0), tf.float32)

    # get masks
    pos_mask, _ = filter_singles({"features": pos_features}, params)
    neg_mask, _ = filter_singles({"features": neg_features}, params)
    keep_mask = tf.add(pos_mask["features"], neg_mask["features"])

    # mask features
    features = tf.multiply(features, keep_mask)

    # output for later
    num_positive_features = tf.reduce_sum(
        tf.cast(
            tf.greater(
                tf.reduce_max(features, axis=[1,3]), [0]),
            tf.float32), axis=1, keepdims=True)

    # save desired outputs
    outputs["features"] = features
    outputs["positive_importance_bp_sum"] = num_positive_features

    return outputs, params





def filter_by_importance(inputs, params):
    """Filter out low importance examples, not interesting
    """
    assert inputs.get("features") is not None
    
    # features
    features = inputs["features"]
    
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
