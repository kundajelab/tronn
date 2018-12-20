"""Description: graphs that transform importance scores to other representations
"""

import logging

import tensorflow as tf

from tronn.nets.sequence_nets import generate_dinucleotide_shuffles
from tronn.nets.sequence_nets import generate_scaled_inputs
from tronn.nets.sequence_nets import decode_onehot_sequence

from tronn.nets.filter_nets import filter_and_rebatch

from tronn.nets.normalization_nets import interpolate_logits_to_labels
from tronn.nets.normalization_nets import normalize_to_importance_logits
from tronn.nets.normalization_nets import normalize_to_absolute_one

from tronn.nets.stats import get_gaussian_confidence_intervals
from tronn.nets.stats import check_confidence_intervals

from tronn.nets.threshold_nets import build_null_distribution_threshold_fn
from tronn.nets.threshold_nets import get_threshold_mask_twotailed

from tronn.nets.util_nets import build_stack
from tronn.nets.util_nets import transform_auxiliary_tensor
from tronn.nets.util_nets import attach_auxiliary_tensors
from tronn.nets.util_nets import detach_auxiliary_tensors
from tronn.nets.util_nets import rebatch

from tronn.util.utils import DataKeys


def threshold_w_dinuc_shuffle_null(inputs, params):
    """get a threshold using dinuc-shuffle-derived 
    importances as a null background
    """
    logging.info("LAYER: thresholding with dinuc shuffle null")
    
    # features and shuffles
    features = inputs[DataKeys.FEATURES]
    shuffles = inputs[DataKeys.ACTIVE_SHUFFLES] # {N, task, aux, pos, 4}
    outputs = dict(inputs)

    # params
    # for a sequence of 1Kb with pval 0.01, 10 bp will pass by chance
    pval_thresh = params.get("pval_thresh", 0.01)
    thresholds_key = params["thresholds_key"]
    
    # adjust shuffles so that you get a threshold
    # for every example, for every task
    #shuffles = tf.transpose(shuffles, perm=[0,2,1,3,4]) # {N, task, aux, pos, 4}
    shuffles_shape = shuffles.get_shape().as_list()
    shuffles = tf.reshape(shuffles, [-1]+shuffles_shape[2:])
    shuffles = tf.reduce_sum(shuffles, axis=3) # {N*task, shuf, pos}

    # get thresholds for each example, for each task
    threshold_fn = build_null_distribution_threshold_fn(pval_thresh)
    thresholds = tf.map_fn(
        threshold_fn,
        shuffles,
        back_prop=False,
        parallel_iterations=1)
    
    # readjust shape and save
    feature_shape = features.get_shape().as_list()
    thresholds = tf.reshape(
        thresholds,
        feature_shape[0:2] + [1 for i in xrange(len(feature_shape[2:]))])
    outputs[thresholds_key] = thresholds
    logging.debug("...thresholds: {}".format(thresholds.get_shape()))

    
    # apply
    threshold_mask = get_threshold_mask_twotailed(
        {DataKeys.FEATURES: features, thresholds_key: thresholds},
        {"thresholds_key": thresholds_key})[0][DataKeys.FEATURES]
    outputs[DataKeys.FEATURES] = tf.multiply(threshold_mask, features)

    logging.debug("RESULTS: {}".format(outputs[DataKeys.FEATURES].get_shape()))
    
    return outputs, params



class FeatureImportanceExtractor(object):
    """Follows a kind of tf Hooks framework - do things before, during, after model run"""
    
    def __init__(
            self,
            model_fn,
            feature_type="sequence",
            num_shuffles=7):
        """initialize object
        """
        self.model_fn = model_fn
        self.num_shuffles = num_shuffles
        self.feature_type = feature_type
        
        
    def preprocess(self, inputs, params):
        """assertions and any feature preprocessing
        """
        logging.info("feature importance extractor: preprocess")
        # run some generic assertions that apply to all here
        assert inputs.get(DataKeys.FEATURES) is not None

        # generate shuffles if sequence
        if self.feature_type == "sequence":

            # TODO assert is sequence
            
            # save out the original sequence
            inputs[DataKeys.ORIG_SEQ] = inputs[DataKeys.FEATURES]

            # update params for where to put the dinuc shuffles
            params.update({"aux_key": DataKeys.ORIG_SEQ_SHUF})
            params.update({"num_shuffles": self.num_shuffles})
            inputs, params = generate_dinucleotide_shuffles(inputs, params)

            # and attach the shuffles (aux_key points to ORIG_SEQ_SHUF)
            params.update({"name": "attach_dinuc_seq"})
            inputs, params = attach_auxiliary_tensors(inputs, params)

        return inputs, params
            
            
    def run_model_and_get_anchors(self, inputs, params):
        """run the model fn with preprocessed inputs
        """
        logging.debug("LAYER: call model and set up backprop")
        layer_key = params.get("layer_key", DataKeys.LOGITS)
        
        reuse = params.get("model_reuse", True)
        with tf.variable_scope("", reuse=reuse):
            logging.info("Calling model fn")
            model_params = params["model"].model_params
            model_params.update({"is_inferring": True})
            model_outputs, _ = self.model_fn(inputs, model_params)

        # gather anchors
        inputs[DataKeys.IMPORTANCE_ANCHORS] = tf.gather(
            model_outputs[layer_key],
            params["importance_task_indices"],
            axis=-1)

        # save out logits 
        inputs[DataKeys.LOGITS] = model_outputs[DataKeys.LOGITS]
        if "ensemble" in params["model"].name:
            inputs[DataKeys.LOGITS_MULTIMODEL] = model_outputs[DataKeys.LOGITS_MULTIMODEL]

        # adjust logits for normalization
        if params.get("prediction_sample") is not None:
            # adjust logits here. 
            inputs, _  = interpolate_logits_to_labels(inputs, params)

        # debug
        logging.debug("FEATURES: {}".format(inputs[DataKeys.FEATURES].get_shape()))
        
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


    # TODO deprecate, this doesn't get used
    @classmethod
    def get_multimodel_multitask_feature_importances(cls, inputs, params):
        """do multimodel importances
        """
        anchors = inputs[DataKeys.IMPORTANCE_ANCHORS] # {N, model, logit}
        outputs = dict(inputs)
        
        num_models = params["num_models"]
        num_gpus = params["num_gpus"]
        
        importances = []
        gradients = []
        for model_idx in range(num_models):
            pseudo_count = num_gpus - (num_models % num_gpus) - 1 # ex 10 models, 3 gpus gives 1. 10m, 4g = 1, 10m, 5g=0
            device = "/gpu:{}".format((num_models + pseudo_count - model_idx) % num_gpus)
            print device
            with tf.device(device):
                inputs[DataKeys.IMPORTANCE_ANCHORS] = anchors[:,model_idx] # {N, logit}
                results, _ = cls.get_multitask_feature_importances(inputs, params)

            importances.append(results[DataKeys.WEIGHTED_SEQ]) # list of {N, task, seqlen, 4}
            gradients.append(results[DataKeys.IMPORTANCE_GRADIENTS]) # list of {N, task, seqlen, 4}

        importances = tf.stack(importances, axis=1) # {N, model, task, seqlen, 4}
        gradients = tf.stack(gradients, axis=1)
        
        # save out
        outputs["importances.multimodel"] = importances # in general dont save this, it's big
        outputs[DataKeys.FEATURES] = tf.reduce_mean(importances, axis=1)
        outputs[DataKeys.WEIGHTED_SEQ] = tf.reduce_mean(importances, axis=1)
        outputs[DataKeys.IMPORTANCE_GRADIENTS] = tf.reduce_mean(gradients, axis=1)
                
        return outputs, params
    
    

    def threshold(self, inputs, params):
        """threshold using dinuc shuffles
        """
        logging.info("TRANSFORM: threshold features")
        
        SHUFFLE_PVAL = 0.01
        #SHUFFLE_PVAL = 0.05

        # threshold main features
        inputs.update({DataKeys.ACTIVE_SHUFFLES: inputs[DataKeys.WEIGHTED_SEQ_SHUF]})
        params.update({"pval_thresh": SHUFFLE_PVAL})
        params.update({"thresholds_key": DataKeys.WEIGHTED_SEQ_THRESHOLDS})
        outputs, params = threshold_w_dinuc_shuffle_null(inputs, params)

        # apply to shuffles
        threshold_mask = get_threshold_mask_twotailed(
            {DataKeys.FEATURES: outputs[DataKeys.WEIGHTED_SEQ_SHUF],
             DataKeys.WEIGHTED_SEQ_THRESHOLDS: outputs[DataKeys.WEIGHTED_SEQ_THRESHOLDS]},
            params)[0][DataKeys.FEATURES]
        outputs[DataKeys.WEIGHTED_SEQ_SHUF] = tf.multiply(
            threshold_mask,
            outputs[DataKeys.WEIGHTED_SEQ_SHUF])
        
        logging.debug("...TRANSFORM RESULTS: {}".format(outputs[DataKeys.FEATURES].get_shape()))
        logging.debug("...TRANSFORM RESULTS: {}".format(outputs[DataKeys.WEIGHTED_SEQ_SHUF].get_shape()))
        
        return outputs, params


    def denoise(self, inputs, params):
        """perform any de-noising as desired
        """
        logging.info("TRANSFORM: denoise by removing alone bps")
        SINGLES_FILT_WINDOW = 7
        SINGLES_FILT_WINDOW_MIN = float(2) / SINGLES_FILT_WINDOW
        
        params.update({"window": SINGLES_FILT_WINDOW, "min_fract": SINGLES_FILT_WINDOW_MIN})
        outputs, params = filter_singles_twotailed(inputs, params)
        outputs[DataKeys.WEIGHTED_SEQ_SHUF] = transform_auxiliary_tensor(
            filter_singles_twotailed, DataKeys.WEIGHTED_SEQ_SHUF, outputs, params, aux_axis=2)

        logging.debug("...TRANSFORM RESULTS: {}".format(outputs[DataKeys.FEATURES].get_shape()))
        
        return outputs, params


    def normalize(self, inputs, params):
        """normalize
        """
        logging.info("TRANSFORM: normalize importance scores")
        # normalize to logits
        params.update({"weight_key": DataKeys.LOGITS})
        outputs, params = normalize_to_importance_logits(inputs, params)
        params.update({"weight_key": DataKeys.LOGITS_SHUF})
        params.update({"task_axis": 1})
        outputs[DataKeys.WEIGHTED_SEQ_SHUF] = normalize_to_importance_logits(
            {DataKeys.FEATURES: outputs[DataKeys.WEIGHTED_SEQ_SHUF],
             DataKeys.LOGITS_SHUF: outputs[DataKeys.LOGITS_SHUF]}, params)[0][DataKeys.FEATURES]
        del params["task_axis"]
        
        logging.info("...TRANSFORM RESULTS: {}".format(outputs[DataKeys.FEATURES].get_shape()))
        
        return outputs, params
    

    def clip_sequences(self, inputs, params):
        """clip sequences
        """
        assert params.get("left_clip") is not None
        assert params.get("right_clip") is not None
        
        LEFT_CLIP = params["left_clip"]
        RIGHT_CLIP = params["right_clip"]
        
        # create shallow copy
        outputs = dict(inputs)

        # clip weighted sequences (real and shuf)
        outputs[DataKeys.FEATURES] = outputs[DataKeys.FEATURES][:,:,LEFT_CLIP:RIGHT_CLIP,:]
        outputs[DataKeys.WEIGHTED_SEQ_ACTIVE] = outputs[DataKeys.FEATURES] # save out from features
        outputs[DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF] = outputs[
            DataKeys.WEIGHTED_SEQ_SHUF][:,:,:,LEFT_CLIP:RIGHT_CLIP,:]

        # clip original sequences (real and shuf)
        outputs[DataKeys.ORIG_SEQ_ACTIVE] = outputs[
            DataKeys.ORIG_SEQ][:,:,LEFT_CLIP:RIGHT_CLIP,:]            
        outputs[DataKeys.ORIG_SEQ_ACTIVE_SHUF] = outputs[
            DataKeys.ORIG_SEQ_SHUF][:,:,:,LEFT_CLIP:RIGHT_CLIP,:]

        return outputs, params

    
    def postprocess(self, inputs, params):
        """post processing 
        """
        logging.info("feature importance extractor: postprocess")
        if self.feature_type == "sequence":

            # remove the shuffles - weighted seq (orig seq shuffles are already saved)
            params.update({"save_aux": {
                DataKeys.FEATURES: DataKeys.WEIGHTED_SEQ_SHUF,
                DataKeys.LOGITS: DataKeys.LOGITS_SHUF}})
            params.update({"rebatch_name": "detach_dinuc_seq"})
            outputs, params = detach_auxiliary_tensors(inputs, params)

            # adjust the axes for the saved shuffles so that task comes before shuffle
            outputs[DataKeys.ORIG_SEQ_SHUF] = tf.transpose(
                outputs[DataKeys.ORIG_SEQ_SHUF], perm=[0,2,1,3,4]) # {N, task, aux, pos, 4}
            outputs[DataKeys.WEIGHTED_SEQ_SHUF] = tf.transpose(
                outputs[DataKeys.WEIGHTED_SEQ_SHUF], perm=[0,2,1,3,4])
            outputs[DataKeys.LOGITS_SHUF] = tf.transpose(
                outputs[DataKeys.LOGITS_SHUF], perm=[0,2,1])

            # threshold
            outputs, params = self.threshold(outputs, params)
            
            # denoise <- TODO move this later if doing ensemble?
            outputs, params = self.denoise(outputs, params)
            
            # normalize
            outputs, params = self.normalize(outputs, params)

            # clip
            outputs, params = self.clip_sequences(outputs, params)
            
        return outputs, params


    def cleanup(self, inputs, params):
        """any clean up operations to perform after all work is done
        """
        del inputs[DataKeys.IMPORTANCE_ANCHORS]
        del inputs[DataKeys.ORIG_SEQ_SHUF]
        del inputs[DataKeys.WEIGHTED_SEQ_SHUF]
        del inputs[DataKeys.ACTIVE_SHUFFLES]

        # to consider: do I need original weighted seq, 1000 bp?
        
        return inputs, params


    def multimodel_merge(self, multimodel_outputs):
        """given a list of tensor dicts, pull out the ones that need to be merged
        and merge, and otherwise just grab from model_0.
        """
        outputs = dict(multimodel_outputs[0])
        
        # concat
        concat_keys = [
            DataKeys.WEIGHTED_SEQ_THRESHOLDS,
            DataKeys.FEATURES,
            DataKeys.IMPORTANCE_GRADIENTS,
            DataKeys.WEIGHTED_SEQ,
            DataKeys.WEIGHTED_SEQ_ACTIVE,
            DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
            DataKeys.LOGITS_SHUF]
        for key in concat_keys:
            concat_output = []
            for model_outputs in multimodel_outputs:
                concat_output.append(model_outputs[key])
            concat_output = tf.stack(concat_output, axis=1)
            outputs[key] = concat_output

        # keep out a vector of importances for weighted seq active?
        if False:
            # TODO delete this
            outputs["importances.multimodel"] = outputs[DataKeys.FEATURES]
        
        if True:
            # TODO separate this out to different fn
            # get confidence interval
            outputs["multimodel.importances.tmp"] = tf.reduce_sum(
                outputs[DataKeys.WEIGHTED_SEQ_ACTIVE], axis=-1)
            ci_params = {
                "ci_in_key": "multimodel.importances.tmp",
                "ci_out_key": DataKeys.WEIGHTED_SEQ_ACTIVE_CI,
                "std_thresh": 2.576} # 90% confidence interval 1.645, 95% is 1.96
            outputs, _ = get_gaussian_confidence_intervals(
                outputs, ci_params)
            del outputs["multimodel.importances.tmp"]

            # threshold
            thresh_params = {
                "ci_out_key": DataKeys.WEIGHTED_SEQ_ACTIVE_CI,
                "ci_pass_key": DataKeys.WEIGHTED_SEQ_ACTIVE_CI_THRESH}
            outputs, _ = check_confidence_intervals(outputs, thresh_params)
            outputs[DataKeys.WEIGHTED_SEQ_ACTIVE_CI_THRESH] = tf.logical_not(
                outputs[DataKeys.WEIGHTED_SEQ_ACTIVE_CI_THRESH])
            outputs[DataKeys.WEIGHTED_SEQ_ACTIVE_CI_THRESH] = tf.expand_dims(
                outputs[DataKeys.WEIGHTED_SEQ_ACTIVE_CI_THRESH], axis=-1)

            # TODO maybe need to do this for shuffles also...?
            # to make it consistent for the motif scanning?
            
        # of concat, some get merged
        merge_keys = [
            DataKeys.FEATURES,
            DataKeys.IMPORTANCE_GRADIENTS,
            DataKeys.WEIGHTED_SEQ,
            DataKeys.WEIGHTED_SEQ_ACTIVE,
            DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
            DataKeys.LOGITS_SHUF]
        for key in merge_keys:
            outputs[key] = tf.reduce_mean(outputs[key], axis=1)

        # TODO and then filter the features and weighted seq
        outputs[DataKeys.WEIGHTED_SEQ_ACTIVE] = tf.multiply(
            outputs[DataKeys.WEIGHTED_SEQ_ACTIVE],
            tf.cast(outputs[DataKeys.WEIGHTED_SEQ_ACTIVE_CI_THRESH], tf.float32))

        outputs[DataKeys.FEATURES] = tf.multiply(
            outputs[DataKeys.FEATURES],
            tf.cast(outputs[DataKeys.WEIGHTED_SEQ_ACTIVE_CI_THRESH], tf.float32))

        
        return outputs
    

    def extract(self, inputs, params):
        """put all the pieces together
        """
        outputs, params = self.preprocess(inputs, params)

        if "ensemble" in params["model"].name:
            # TODO potentially split the stack up entirely at this level...
            if False:
                # TODO deprecate this version
                params.update({"layer_key": DataKeys.LOGITS_MULTIMODEL})
                outputs, params = self.run_model_and_get_anchors(outputs, params)
                outputs, params = self.get_multimodel_multitask_feature_importances(outputs, params)
                outputs, params = self.postprocess(outputs, params)
            else:
                # separate complete stack here
                params.update({"layer_key": DataKeys.LOGITS_MULTIMODEL})
                outputs, params = self.run_model_and_get_anchors(outputs, params)

                # pull out key outputs that get used per model
                anchors = outputs[DataKeys.IMPORTANCE_ANCHORS]
                if outputs.get(DataKeys.LOGITS_MULTIMODEL_NORM) is not None:
                    logits = outputs[DataKeys.LOGITS_MULTIMODEL_NORM]                    
                else:
                    logits = outputs[DataKeys.LOGITS_MULTIMODEL]

                # get num models and num gpus
                num_models = params["model"].model_params["num_models"]
                num_gpus = params["model"].model_params["num_gpus"]

                # extract relevant keys on a per model basis
                multimodel_outputs = []
                for model_idx in range(num_models):
                    pseudo_count = num_gpus - (num_models % num_gpus) - 1
                    device = "/gpu:{}".format((num_models + pseudo_count - model_idx) % num_gpus)
                    print device
                    with tf.device(device):
                        outputs[DataKeys.IMPORTANCE_ANCHORS] = anchors[:,model_idx] # {N, logit}
                        outputs[DataKeys.LOGITS] = logits[:,model_idx] # {N, logit}
                        params.update({"model_string": "model_{}".format(model_idx)})
                        model_outputs, model_params = self.get_multitask_feature_importances(outputs, params)
                        model_outputs, _ = self.postprocess(model_outputs, model_params)
                        multimodel_outputs.append(model_outputs)

                # merge the outputs
                outputs = self.multimodel_merge(multimodel_outputs)
                if outputs.get(DataKeys.LOGITS_MULTIMODEL_NORM) is not None:
                    outputs[DataKeys.LOGITS] = outputs[DataKeys.LOGITS_NORM]
                
                # TODO build a function to calculate correlations across pairs

        else:
            outputs, params = self.run_model_and_get_anchors(outputs, params)
            outputs, params = self.get_multitask_feature_importances(outputs, params)
            outputs, params = self.postprocess(outputs, params)

        # cleanup
        outputs, params = self.cleanup(outputs, params)
        
        return outputs, params

        
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
    
    
class PyTorchInputxGrad(FeatureImportanceExtractor):
    """Runs input x grad"""

    @staticmethod
    def get_singletask_feature_importances(inputs, params):

        # TODO this is not generalizable to multitask, fix later
        # probably pass in the index or something
        
        # inputs
        features = inputs[DataKeys.FEATURES]
        anchor = inputs["anchor"]
        outputs = dict(inputs)
        
        # input x grad
        outputs[DataKeys.FEATURES] = tf.multiply(
            features,
            inputs[DataKeys.IMPORTANCE_GRADIENTS], 'input_x_grad')

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
    def get_diff_from_ref(cls, activations, num_shuffles):
        """Get the diff from reference, but maintain batch
        only remove shuffles at the end of the process. For deeplift
        """
        # get actual example num
        batch_size = activations.get_shape().as_list()[0]
        assert batch_size % (num_shuffles + 1) == 0
        example_num = batch_size / (num_shuffles + 1)
        
        # go through sub batches (actual + shuffle refs)
        diff_from_ref = []
        for i in xrange(example_num):
            idx = (num_shuffles + 1) * (i)
            actual = activations[idx:idx+num_shuffles+1]
            references = activations[idx+1:idx+num_shuffles+1]
            diff = tf.subtract(
                actual,
                tf.reduce_mean(references, axis=0, keepdims=True))
            diff_from_ref.append(diff)
            
        # restack
        diff_from_ref = tf.concat(diff_from_ref, axis=0)
        
        return diff_from_ref

    
    @classmethod
    def _is_nonlinear_var(cls, var):
        """check if variable is nonlinear, use names
        """
        nonlinear_names = ["Relu", "relu"]
        is_nonlinear = False
        for nonlinear_name in nonlinear_names:
            if nonlinear_name in var.name:
                is_nonlinear = True
                break

        return is_nonlinear

    
    @classmethod
    def _is_linear_var(cls, var):
        """check if variable is linear, use names
        """
        linear_names = set(["Conv", "conv", "fc", "Reshape", "reshape", "anchor"])
        is_linear = False
        for linear_name in linear_names:
            if linear_name in var.name:
                is_linear = True
                break

        return is_linear
    
    
    @classmethod
    def build_deeplift_multipliers(
            cls, x, y,
            num_shuffles,
            old_multipliers,
            nonlinear_fn="rescale"):
        """Takes input and activations to pass down
        """
        if cls._is_nonlinear_var(y):
            delta_y = cls.get_diff_from_ref(y, num_shuffles)
            delta_x = cls.get_diff_from_ref(x, num_shuffles)
            if nonlinear_fn == "rescale":
                # rescale
                multipliers = tf.divide(delta_y, delta_x)
            elif nonlinear_fn == "reveal_cancel":
                # reveal cancel
                positive_multipliers = tf.divide(
                    tf.nn.relu(delta_y),
                    tf.nn.relu(delta_x))
                negative_multipliers = tf.divide(
                    tf.nn.relu(-delta_y),
                    tf.nn.relu(-delta_x))
                multipliers = tf.add(positive_multipliers, negative_multipliers)
            else:
                raise ValueError, "nonlinear fn not implemented!"
            
            # NOTE that paper says to use gradients when delta_x is small
            # for numerical instability
            multipliers = tf.where(
                tf.less(tf.abs(delta_x), 1e-8),
                x=tf.gradients(y, x)[0],
                y=multipliers)

            # and chain rule
            multipliers = tf.multiply(multipliers, old_multipliers)
            
        elif cls._is_linear_var(y):
            # linear rule
            [weights] = tf.gradients(y, x, grad_ys=old_multipliers)
            multipliers = weights
            
        else:
            # just use linear rule if not recognized?
            raise ValueError, "{} not recognized as linear or nonlinear".format(y.name)
        
        return multipliers


    @classmethod
    def get_singletask_feature_importances(cls, inputs, params):
        """run deeplift on a single anchor
        """
        # inputs
        features = inputs[DataKeys.FEATURES]
        anchor = tf.identity(inputs["anchor"], name="anchor")
        outputs = dict(inputs)
        num_shuffles = params.get("num_shuffles")
        activations = tf.get_collection("DEEPLIFT_ACTIVATIONS")
        model_string = params.get("model_string", "")
        activations = [features] + [activation for activation in activations
                       if model_string in activation.name] + [anchor]
        
        # go backwards through the variables
        activations.reverse()
        for i in xrange(len(activations)):
            current_activation = activations[i]
            print current_activation
            
            if i == 0:
                previous_activation = tf.identity(
                    anchor, name="fc.anchor")
                multipliers = None
            else:
                previous_activation = activations[i-1]

            # function here to build multiplier and pass down
            multipliers = cls.build_deeplift_multipliers(
                current_activation,
                previous_activation,
                num_shuffles,
                multipliers)
            
        # save out multipliers as gradients
        outputs[DataKeys.IMPORTANCE_GRADIENTS] = multipliers
        outputs[DataKeys.FEATURES] = tf.multiply(features, multipliers)
        
        return outputs, params


class DeltaFeatureImportanceMapper(InputxGrad):
    """DFIM - given mutational results, get importance scores 
    and return the delta results"""

    def preprocess(self, inputs, params):
        """preprocess for delta features
        """
        assert inputs.get(DataKeys.FEATURES) is not None
        assert inputs.get(DataKeys.MUT_MOTIF_ORIG_SEQ) is not None
        
        # don't generate shuffles, assume that the shuffles are still attached from before
        # attach the mutations (interleaved)
        logging.info("LAYER: attaching {}".format(DataKeys.MUT_MOTIF_ORIG_SEQ))
        params.update({"aux_key": DataKeys.MUT_MOTIF_ORIG_SEQ})        
        params.update({"name": "attach_mut_motif_seq"})
        inputs, params = attach_auxiliary_tensors(inputs, params)
        logging.debug("After attaching aux: {}".format(inputs[DataKeys.FEATURES].get_shape()))
        
        return inputs, params

    
    def postprocess(self, inputs, params):
        """postprocess
        """
        LEFT_CLIP = 420
        RIGHT_CLIP = 580

        # remove the mutations - weighted seq (orig seq shuffles are already saved)
        params.update({"save_aux": {
            DataKeys.FEATURES: DataKeys.MUT_MOTIF_WEIGHTED_SEQ,
            DataKeys.LOGITS: DataKeys.MUT_MOTIF_LOGITS}})
        params.update({"rebatch_name": "detach_mut_motif_seq"})
        outputs, params = detach_auxiliary_tensors(inputs, params)
        logging.debug("FEATURES: {}".format(outputs[DataKeys.FEATURES].get_shape()))
        logging.debug("MUT_MOTIF_WEIGHTED_SEQ: {}".format(
            outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ].get_shape()))
        logging.debug("MUT_MOTIF_LOGITS: {}".format(outputs[DataKeys.MUT_MOTIF_LOGITS].get_shape()))
        
        # apply thresholds to normal and mut sequences
        # NOTE: for now, only deal with positives
        thresholds = outputs[DataKeys.WEIGHTED_SEQ_THRESHOLDS]
        
        features = outputs[DataKeys.FEATURES]
        pass_positive_thresh = tf.cast(tf.greater(features, thresholds), tf.float32)
        pass_negative_thresh = tf.cast(tf.less(features, -thresholds), tf.float32)
        passed_thresholds = tf.add(pass_positive_thresh, pass_negative_thresh)
        outputs[DataKeys.FEATURES] = tf.multiply(
            passed_thresholds, outputs[DataKeys.FEATURES])
        outputs[DataKeys.FEATURES] = tf.nn.relu(outputs[DataKeys.FEATURES])

        thresholds = tf.expand_dims(thresholds, axis=1)
        features = outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ]
        pass_positive_thresh = tf.cast(tf.greater(features, thresholds), tf.float32)
        pass_negative_thresh = tf.cast(tf.less(features, -thresholds), tf.float32)
        passed_thresholds = tf.add(pass_positive_thresh, pass_negative_thresh)
        outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ] = tf.multiply(
            passed_thresholds, outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ])
        outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ] = tf.nn.relu(outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ])
        
        # denoise - this is same as above?
        # can it be made the same by attach.detach?
        logging.info("TRANSFORM: denoise by removing alone bps")
        SINGLES_FILT_WINDOW = 7
        SINGLES_FILT_WINDOW_MIN = float(2) / SINGLES_FILT_WINDOW
        
        params.update({"window": SINGLES_FILT_WINDOW, "min_fract": SINGLES_FILT_WINDOW_MIN})
        outputs, params = filter_singles_twotailed(outputs, params)
        outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ] = transform_auxiliary_tensor( # except this is not
            filter_singles_twotailed, DataKeys.MUT_MOTIF_WEIGHTED_SEQ, outputs, params, aux_axis=2)

        # normalize
        params.update({"weight_key": DataKeys.LOGITS})        
        outputs, params = normalize_to_importance_logits(outputs, params)
        params.update({"weight_key": DataKeys.MUT_MOTIF_LOGITS})
        outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ] = normalize_to_importance_logits(
            {DataKeys.FEATURES: outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ],
             DataKeys.MUT_MOTIF_LOGITS: outputs[DataKeys.MUT_MOTIF_LOGITS]},
            params)[0][DataKeys.FEATURES]
        
        # TODO
        # (1) features divide by total, multiply by logits - this gives you features normalized to logits
        # (2) 1ST OUTPUT - dy/dx, where you get dy by subtracting orig from mut response, and
        #     dx is subtracting orig from mut at the mut position(s). this is used for permute test
        # (3) 2ND OUTPUT - just the subtraction, which is used for ranking/vis
        # normalization - NOTE: this function normalizes while blanking the motif site
        outputs[DataKeys.FEATURES] = tf.expand_dims(outputs[DataKeys.FEATURES], axis=1)
        if False:
            outputs, params = normalize_to_absolute_one(outputs, params)
            outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ] = normalize_to_absolute_one(
                {DataKeys.FEATURES: outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ],
                 DataKeys.MUT_MOTIF_POS: outputs[DataKeys.MUT_MOTIF_POS]},
                params)[0][DataKeys.FEATURES]
        
        # for both raw and weighted, both shuf and not
        # (1) clip edges
        # if all is correct, only need to chop the features and the mut motif seqs
        # this is same as above
        # and clip the position mask also (for blanking at the motif level)
        outputs[DataKeys.FEATURES] = outputs[DataKeys.FEATURES][:,:,:,LEFT_CLIP:RIGHT_CLIP,:]
        outputs[DataKeys.MUT_MOTIF_ORIG_SEQ] = outputs[DataKeys.MUT_MOTIF_ORIG_SEQ][
            :,:,:,LEFT_CLIP:RIGHT_CLIP,:]
        outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ] = outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ][
            :,:,:,LEFT_CLIP:RIGHT_CLIP,:]
        outputs[DataKeys.MUT_MOTIF_POS] = outputs[DataKeys.MUT_MOTIF_POS][:,:,:,LEFT_CLIP:RIGHT_CLIP,:]

        params.update({"decode_key": DataKeys.MUT_MOTIF_ORIG_SEQ})
        outputs, _ = decode_onehot_sequence(outputs, params)
        
        # calculate deltas scores (DFIM). leave as aux (to attach later)
        # this is dy
        outputs[DataKeys.DFIM_SCORES] = tf.subtract(
            outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ],
            outputs[DataKeys.FEATURES])

        # dx
        #  {N, mut, pos, 4}
        orig_x = tf.reduce_sum(tf.multiply(
            outputs[DataKeys.FEATURES],
            outputs[DataKeys.MUT_MOTIF_POS]), axis=[2,3,4])

        mut_x = tf.reduce_sum(tf.multiply(
            outputs[DataKeys.MUT_MOTIF_WEIGHTED_SEQ],
            outputs[DataKeys.MUT_MOTIF_POS]), axis=[2,3,4])

        # can keep this separate for now
        outputs[DataKeys.DFIM_SCORES_DX] = tf.subtract(mut_x, orig_x)
        
        # zero out the ones that shouldn't have responded
        # they already give low responses, but easiest to just zero them out
        if True:
            mut_motif_present = tf.cast(outputs[DataKeys.MUT_MOTIF_PRESENT], tf.float32)

            # NOTE: turn on for synergy scores!!
            if params["cmd_name"] == "synergy":
                print "WARNING USING SYNERGY SCORE VERSION"
                mut_motif_present = outputs[DataKeys.MUT_MOTIF_PRESENT]
                mut_motif_present = tf.cast(
                    tf.reduce_all(mut_motif_present, axis=1, keepdims=True),
                    tf.float32) # {N, 1}
            else:
                # TURN THIS OFF FOR DMIM
                print "WARNING USING DMIM VERSION"
            
            mut_motif_shape = mut_motif_present.get_shape().as_list()
            feature_shape = outputs[DataKeys.DFIM_SCORES].get_shape().as_list()
            mut_motif_present = tf.reshape(
                mut_motif_present,
                mut_motif_shape + [1 for i in xrange(len(feature_shape) - len(mut_motif_shape))])
            outputs[DataKeys.DFIM_SCORES] = tf.multiply(
                mut_motif_present,
                outputs[DataKeys.DFIM_SCORES])
        
        outputs[DataKeys.FEATURES] = outputs[DataKeys.DFIM_SCORES]
        
        # calculate the delta logits later, in post analysis
        
        # Q: is there a way to get the significance of a delta score even here?
        # ie, what is the probability of a delta score by chance?
        
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
    model_fn = params["model"].model_fn
    
    # all this should be is a wrapper
    if backprop == "input_x_grad":
        extractor = InputxGrad(model_fn)
    elif backprop == "pytorch_input_x_grad":
        extractor = PyTorchInputxGrad(model_fn)
    elif backprop == "integrated_gradients":
        extractor = IntegratedGradients(model_fn)
    elif backprop == "deeplift":
        extractor = DeepLift(model_fn)
    elif backprop == "saturation_mutagenesis":
        raise ValueError, "backprop method not implemented!"
    else:
        raise ValueError, "backprop method not implemented!"
    
    outputs, params = extractor.extract(inputs, params)
    
    # filter by importance
    if params.get("use_filtering", True):
        print "using filtering"
        MIN_IMPORTANCE_BP = 10
        outputs, _ = filter_by_importance(
            outputs,
            {"cutoff": MIN_IMPORTANCE_BP,
             "positive_only": True})
        logging.info("OUT: {}".format(outputs[DataKeys.FEATURES].get_shape()))

    return outputs, params


def run_dfim(inputs, params):
    """wrapper for functional calls
    """
    extractor = DeltaFeatureImportanceMapper(params["model"].model_fn)
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
    outputs[DataKeys.FEATURES] = tf.greater(features, 0)
    
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

    # reduce to put all bases in one dimension
    pos_features = tf.reduce_sum(pos_features, axis=3, keepdims=True)
    neg_features = tf.reduce_sum(pos_features, axis=3, keepdims=True)

    # get masks
    pos_mask = filter_singles({DataKeys.FEATURES: pos_features}, params)[0][DataKeys.FEATURES]
    neg_mask = filter_singles({DataKeys.FEATURES: neg_features}, params)[0][DataKeys.FEATURES]
    keep_mask = tf.cast(tf.logical_or(pos_mask, neg_mask), tf.float32)

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
        
    inputs["positive_importance_bp_sum"] = feature_sums
    inputs["condition_mask"] = tf.greater(feature_sums, cutoff)
    params.update({"name": "importances_filter"})
    outputs, _ = filter_and_rebatch(inputs, params)
    
    return outputs, params
