"""Contains nets that perform PWM convolutions
"""

import h5py
import logging

import numpy as np
import tensorflow as tf

from tronn.nets.threshold_nets import build_null_distribution_threshold_fn
from tronn.nets.threshold_nets import get_threshold_mask_twotailed

from tronn.nets.filter_nets import filter_and_rebatch

from tronn.util.initializers import pwm_simple_initializer
from tronn.util.tf_utils import get_fan_in

from tronn.nets.util_nets import pad_inputs
from tronn.nets.util_nets import rebatch

from tronn.util.utils import DataKeys


class MotifScanner(object):
    """base class, scans sequence and produces pwm maps {motif, position}"""
    
    def __init__(self, features_key=DataKeys.FEATURES, pwms_key="pwms"):
        self.features_key = features_key
        self.pwms_key = pwms_key

        
    @staticmethod
    def pwm_conv2d(inputs, params):
        """Convolve with PWMs and normalize with vector projection:
          projection = a dot b / | b |
        """
        # TODO options for running (1) FWD, (2) REV, (3) COMB
        assert inputs.get(DataKeys.FEATURES) is not None
        assert params.get("pwms") is not None
        assert params.get("filter_width") is not None

        # features
        features = inputs[DataKeys.FEATURES]
        outputs = dict(inputs)

        # params
        pwm_list = params["pwms"]
        max_size = params["filter_width"]
        num_filters = len(pwm_list)
        reuse = params.get("reuse_pwm_layer", False)

        # make the convolution net for dot product, normal filters
        # here, the pwms are already normalized to unit vectors for vector projection
        kernel_size = [1, max_size]
        with tf.variable_scope("pwm_layer", reuse=reuse):
            features = tf.layers.conv2d(
                features,
                num_filters,
                kernel_size,
                padding="valid",
                activation=None,
                kernel_initializer=pwm_simple_initializer(
                    kernel_size,
                    pwm_list,
                    get_fan_in(features),
                    unit_vector=True,
                    length_norm=True),
                use_bias=False,
                bias_initializer=None,
                trainable=False,
                name="conv_pwm") # {N, task, seq_len, num_filters}

        outputs[DataKeys.FEATURES] = features

        return outputs, params

    
    @staticmethod
    def pwm_convolve_twotailed(inputs, params):
        """Convolve both pos and negative scores with PWMs. Prevents getting scores
        when the negative correlation is stronger.
        """
        assert inputs.get(DataKeys.FEATURES) is not None
        
        # features
        features = inputs[DataKeys.FEATURES]
        outputs = dict(inputs)

        # positive weighted sequence
        pos_scores = MotifScanner.pwm_conv2d(
            {DataKeys.FEATURES: tf.nn.relu(features)},
            params)[0][DataKeys.FEATURES]
        pos_scores = tf.nn.relu(pos_scores)

        # negative weighted sequence
        params.update({"reuse_pwm_layer": True})
        neg_scores = MotifScanner.pwm_conv2d(
            {DataKeys.FEATURES: tf.nn.relu(-features)},
            params)[0][DataKeys.FEATURES]
        neg_scores = tf.nn.relu(neg_scores)

        # condition: is positive score higher than neg score
        is_positive_max = tf.greater_equal(pos_scores, neg_scores)
        
        # fill in accordingly: returns both positive and negative values
        outputs[DataKeys.FEATURES] = tf.where(is_positive_max, pos_scores, -neg_scores)
        
        return outputs, params

    
    def preprocess(self, inputs, params):
        """preprocessing
        """
        # assertions
        assert params.get(self.pwms_key) is not None

        # determine filter width
        pwm_list = params[self.pwms_key]
        num_filters = len(pwm_list)
        max_size = 0
        for pwm in pwm_list:
            if pwm.weights.shape[1] > max_size:
                max_size = pwm.weights.shape[1]
        params["filter_width"] = max_size

        # adjust features
        inputs[DataKeys.FEATURES] = inputs[self.features_key]

        return inputs, params

    
    def convolve_motifs(self, inputs, params):
        """run convolution
        """
        return MotifScanner.pwm_convolve_twotailed(inputs, params)

    
    def postprocess(self, inputs, params):
        # things to run after convolving

        # TODO consider whether to re-weight by number of basepairs that were marked as important (nonzero)

        # TODO consider smoothing across time (an average pool with height, no width)
        
        # get the global best scores across tasks?
        #params.update({"append": True, "count_thresh": 1})
        #outputs, params = multitask_global_pwm_scores(inputs, params)
        #print outputs[DataKeys.FEATURES]


        
        return inputs, params


    def scan(self, inputs, params):
        """put all the pieces together
        """
        # run preprocess
        inputs, params = self.preprocess(inputs, params)
        # convolve: NOTE this returns pos and neg scores!
        outputs, params = self.convolve_motifs(inputs, params)
        # postprocess
        outputs, params = self.postprocess(outputs, params)

        # returns {N, task, pos, M}
        return outputs, params


class ShufflesMotifScanner(MotifScanner):
    """scans shuffles to get pvals"""
    
    def preprocess(self, inputs, params):
        """adjust shuffle dimensions
        """
        self.shuffle_shape = inputs[self.features_key].get_shape().as_list()
        shuffles_reshaped = tf.reshape(
            inputs[self.features_key],
            [self.shuffle_shape[0],
             self.shuffle_shape[1]*self.shuffle_shape[2],
             self.shuffle_shape[3],
             self.shuffle_shape[4]])

        # adjust features
        inputs[DataKeys.FEATURES] = shuffles_reshaped

        return inputs, params

    
    def postprocess(self, inputs, params):
        """adjust shuffle dimensions back
        """
        pwm_scores = inputs[DataKeys.FEATURES]
        self.shuffle_shape[3:] = pwm_scores.get_shape().as_list()[2:]
        inputs[DataKeys.FEATURES] = tf.reshape(pwm_scores, self.shuffle_shape)

        return inputs, params

    
class MotifScannerWithThresholds(MotifScanner):
    """add in thresholding"""

    def __init__(
            self,
            features_key=DataKeys.FEATURES,
            shuffles_key=DataKeys.ACTIVE_SHUFFLES,
            out_scores_key=DataKeys.ORIG_SEQ_PWM_SCORES,
            out_hits_key=DataKeys.ORIG_SEQ_PWM_HITS,
            out_scores_thresh_key=DataKeys.ORIG_SEQ_PWM_SCORES_THRESH,
            pval=0.05,
            two_sided_thresh=False,
            **kwargs):
        """init
        """
        super(MotifScannerWithThresholds, self).__init__(
            features_key=features_key, **kwargs)
        self.shuffles_key = shuffles_key
        self.out_scores_key = out_scores_key
        self.out_hits_key = out_hits_key
        self.out_scores_thresh_key = out_scores_thresh_key
        self.pval = pval
        self.two_sided_thresh = two_sided_thresh

        
    @staticmethod
    def threshold(inputs, params):
        """threshold scores using motif scores from 
        shuffled sequence (ie null distribution)
        """
        # requires you have scores on the examples AND the shuffles
        assert inputs.get(DataKeys.ACTIVE_SHUFFLES) is not None
        assert inputs.get(DataKeys.FEATURES) is not None

        # features and shuffles
        features = inputs[DataKeys.FEATURES] # {N, task, pos, M}
        shuffles = inputs[DataKeys.ACTIVE_SHUFFLES] # {N, task, shuf, pos, M}
        outputs = dict(inputs)
        
        # params
        pval_thresh = params.get("pval_thresh", 0.05)
        thresholds_key = params["thresholds_key"]

        # adjust features
        features = tf.transpose(features, perm=[0,3,1,2]) # {N, M, task, pos}
        features_shape = features.get_shape().as_list()
        
        # adjust shuffles and features for thresholding
        perm = [0,4,1,2,3]
        shuffles = tf.transpose(shuffles, perm=perm) # becomes {N, M, task, shuf, pos}
        shuffles_shape = shuffles.get_shape().as_list()
        shuffles = tf.reshape(shuffles, [-1]+shuffles_shape[3:]) # {N*M*task, shuf, pos}
        
        # get thresholds
        threshold_fn = build_null_distribution_threshold_fn(pval_thresh)
        thresholds = tf.map_fn(
            threshold_fn,
            shuffles) # {N*M*task}

        # readjust shape and save
        thresholds = tf.reshape(
            thresholds,
            features_shape[0:3]+[1]) # back to {N, M, task, 1}
        outputs[thresholds_key] = tf.transpose(thresholds, perm=[0,2,3,1]) # {N, task, 1, M}
        
        # apply
        threshold_mask = get_threshold_mask_twotailed(
            {DataKeys.FEATURES: features, thresholds_key: thresholds},
            {"thresholds_key": thresholds_key})[0][DataKeys.FEATURES]
        threshold_mask = tf.transpose(threshold_mask, perm=[0,2,3,1]) # {N, task, pos, M}

        outputs[DataKeys.FEATURES] = threshold_mask

        return outputs, params

    
    def scan(self, inputs, params):
        """run the shuffle scanner and get the results
        """
        # get sequence results
        outputs = super(MotifScannerWithThresholds, self).scan(inputs, params)[0]
        
        # get shuffle results, only keep positive values in active shuffles
        shuffle_scanner = ShufflesMotifScanner(features_key=self.shuffles_key)
        shuffle_results = shuffle_scanner.scan(inputs, params)[0]
        outputs[DataKeys.ACTIVE_SHUFFLES] = tf.nn.relu(shuffle_results[DataKeys.FEATURES])
        
        # get thresholds
        params.update({"pval": self.pval})
        outputs[self.out_hits_key] = MotifScannerWithThresholds.threshold(
            outputs, params)[0][DataKeys.FEATURES]
        outputs[self.out_scores_thresh_key] = tf.multiply(
            outputs[self.out_hits_key],
            outputs[DataKeys.FEATURES])
        
        # also save out raw scores
        outputs[self.out_scores_key] = outputs[DataKeys.FEATURES]

        # delete active shuffles
        del outputs[DataKeys.ACTIVE_SHUFFLES]
        
        return outputs, params


class DeltaMotifImportanceMapper(MotifScanner):
    """DMIM - given mutational results, scan for motifs and return results"""

    def preprocess(self, inputs, params):
        """assertions, and blank the motif site
        """
        assert inputs.get(DataKeys.FEATURES) is not None # {N, mutM, task, pos, 4}
        assert inputs.get(DataKeys.MUT_MOTIF_POS) is not None # {N, mut_M, 1, pos, 1}

        # features - these are dfim scores
        features = inputs[DataKeys.FEATURES]
        mask = tf.cast(tf.greater(inputs[DataKeys.MUT_MOTIF_POS], 0), tf.float32)
        outputs = dict(inputs)
        
        # blank out motif sites
        # TODO extend the blanking, if using single?
        features = tf.multiply(features, mask) # {N, mutM, task, pos, 4}
        
        # join dims for scanning
        features_shape = features.get_shape().as_list()
        params["batch_size"] = features_shape[0]
        features = tf.reshape(features, [-1]+features_shape[2:]) # {N*mutM, task, pos, 4}
        new_batch_size = features.get_shape().as_list()[0]
        outputs[DataKeys.FEATURES] = features
        
        # and pad
        params.update({"num_aux_examples": features_shape[1]-1})
        params.update({"ignore_keys": [DataKeys.FEATURES]})
        outputs, _ = pad_inputs(outputs, params)

        # rebatch to mut size
        # TODO rebatch multiple mutants
        outputs, _ = rebatch(outputs, {"name": "rebatch_dfim", "batch_size": features_shape[1]})

        outputs, params = super(DeltaMotifImportanceMapper, self).preprocess(outputs, params)
        
        return outputs, params

    
    def postprocess(self, inputs, params):
        """make sure you only keep the hits and reduce sum?
        """
        # utilizing existing pwm hits, mask the scores
        features = tf.multiply(
            inputs[DataKeys.FEATURES],
            inputs[DataKeys.ORIG_SEQ_PWM_HITS])

        # adjust shape
        features_shape = features.get_shape().as_list()
        features = tf.reshape(features, [1, -1] + features_shape[1:])

        # TODO unpad based on multiple mutant rebatch
        # and then unpad
        outputs = {}
        for key in inputs.keys():
            outputs[key] = tf.gather(inputs[key], [0])
            
        # get sum across positions. keep both positive and negative scores
        features = tf.reduce_sum(features, axis=3) # {N, mutM, task, M}
        outputs[DataKeys.FEATURES] = features
        print outputs[DataKeys.FEATURES]

        # TODO also save out to a separate aux tensor

        # two tailed, drop or increase
        # the permutation test would be:
        # per mut motif, per task, per motif
        # (ie a mut_M, M pair at a task):
        # look at label shuffling N vs N
        # just keep the diffs, and run permutation test on the diffs
        # perm test is sign flip
        
        return outputs, params
    

def get_pwm_scores(inputs, params):
    """scan raw and weighted sequence with shuffles, and threshold
    using shuffles as null distribution
    """
    # note: need to get (1) raw scores, (2) raw hits, (3) thresholded scores
    
    # run original sequence
    scanner = MotifScannerWithThresholds(
        features_key=DataKeys.ORIG_SEQ_ACTIVE,
        shuffles_key=DataKeys.ORIG_SEQ_ACTIVE_SHUF,
        out_scores_key=DataKeys.ORIG_SEQ_PWM_SCORES,
        out_hits_key=DataKeys.ORIG_SEQ_PWM_HITS,
        out_scores_thresh_key=DataKeys.ORIG_SEQ_PWM_SCORES_THRESH)
    outputs, params = scanner.scan(inputs, params)
    
    # run weighted sequence
    # if onesided (default), then will only return positive scores
    scanner = MotifScannerWithThresholds(
        features_key=DataKeys.WEIGHTED_SEQ_ACTIVE,
        shuffles_key=DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
        out_scores_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES,
        out_hits_key=DataKeys.WEIGHTED_SEQ_PWM_HITS,
        out_scores_thresh_key=DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH)
    outputs.update(scanner.scan(inputs, params)[0]) 
    
    debug = False
    if debug:
        print "results"
        print outputs[DataKeys.ORIG_SEQ_PWM_HITS]
        print outputs[DataKeys.ORIG_SEQ_PWM_SCORES] # check this
        print outputs[DataKeys.ORIG_SEQ_PWM_SCORES_THRESH]

        print outputs[DataKeys.WEIGHTED_SEQ_PWM_HITS]
        print outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES]
        print outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH]
    
    # mask weighted scores with raw hits
    outputs[DataKeys.WEIGHTED_SEQ_PWM_HITS] = tf.multiply(
        outputs[DataKeys.WEIGHTED_SEQ_PWM_HITS],
        outputs[DataKeys.ORIG_SEQ_PWM_HITS])

    outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES] = tf.multiply(
        outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES],
        outputs[DataKeys.ORIG_SEQ_PWM_HITS])
    
    outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH] = tf.multiply(
        outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH],
        outputs[DataKeys.ORIG_SEQ_PWM_HITS])

    # TODO mask raw scores with weighted hits?
    
    
    # squeeze the raw and weighted
    # this happens here because you need to filter the scores
    # with the hits from raw sequence first
    # TODO can try max instead of sum?
    outputs[DataKeys.ORIG_SEQ_PWM_SCORES_SUM] = tf.reduce_sum(
        outputs[DataKeys.ORIG_SEQ_PWM_SCORES_THRESH], axis=2) # {N, 1, M}

    # relu on the weighted scores for now
    outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH] = tf.nn.relu(
        outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH]) # {N, task, pos, M}

    outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM] = tf.reduce_sum(
        outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH], axis=2) # {N, task, M}
    
    # features coming out: the summed weighted scores
    outputs[DataKeys.FEATURES] = outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM] # {N, task, M}
    
    return outputs, params


def get_motif_densities(inputs, params):
    """use an avg pool to get density within window
    and also save out the max motif density val
    for now only do on original sequence
    """
    assert inputs.get(DataKeys.ORIG_SEQ_PWM_HITS) is not None
    outputs = dict(inputs)

    window = params.get("density_window", 20)

    # get densities within windows
    outputs[DataKeys.ORIG_SEQ_PWM_DENSITIES] = tf.layers.average_pooling2d(
        inputs[DataKeys.ORIG_SEQ_PWM_HITS], [1, window], [1,1], padding="same") # {N, 1, pos, M}

    # get max density in the region
    outputs[DataKeys.ORIG_SEQ_PWM_MAX_DENSITIES] = tf.reduce_max(
        outputs[DataKeys.ORIG_SEQ_PWM_DENSITIES], axis=2)

    return outputs, params


def filter_for_significant_pwms(inputs, params):
    """
    """
    assert inputs.get(DataKeys.ORIG_SEQ_PWM_HITS) is not None # {N, task, pos, M}
    assert params.get("manifold") is not None # {cluster, task, M}

    # features
    features = inputs[DataKeys.ORIG_SEQ_PWM_HITS] # {N, 1, pos, M}
    features = tf.reduce_sum(features, axis=2) # {N, 1, M}
    features = tf.expand_dims(features, axis=1) # {N, 1, 1, M}
    outputs = dict(inputs)

    # pwm masks
    with h5py.File(params["manifold"], "r") as hf:
        pwm_masks = hf[DataKeys.MANIFOLD_PWM_SIG_CLUST][:] # {cluster, M}
    pwm_masks = (pwm_masks > 0).astype(int) # make sure bool
    pwm_masks = np.expand_dims(pwm_masks, axis=1) # {cluster, 1, M}
    pwm_masks = np.expand_dims(pwm_masks, axis=0) # {1, cluster, 1, M}
    pwm_totals = np.sum(pwm_masks, axis=3) # {1, cluster, 1}
    pwm_totals = np.squeeze(pwm_totals, axis=2) # {1, cluster}
    
    # multiply
    pwms_present = tf.multiply(features, pwm_masks) # {N, cluster, task, M}
    pwms_present = tf.cast(tf.greater(pwms_present, 0), tf.float32)
    pwms_present_sums = tf.reduce_sum(pwms_present, axis=3) # {N, cluster, task}

    # and get max present across tasks
    pwms_present_sums = tf.reduce_max(pwms_present_sums, axis=2) # {N, cluster}
    
    # passes threshold
    passed_thresholds = tf.cast(tf.equal(pwms_present_sums, pwm_totals), tf.float32) # {N, cluster}
 
    # TODO
    # save out
    outputs["sig_pwms_present"] = passed_thresholds # {N, cluster}

    # check condition
    outputs["condition_mask"] = tf.reduce_any(
        tf.greater(passed_thresholds, 0), axis=1)
        
    params.update({"name": "sig_pwm_filter"})
    outputs, params = filter_and_rebatch(outputs, params)
        
    return outputs, params


def run_dmim(inputs, params):
    """run dmim
    """
    # scan, no shuffles or filtering
    scanner = DeltaMotifImportanceMapper(features_key=DataKeys.FEATURES)
    outputs, params = scanner.scan(inputs, params)

    
    
    return outputs, params






# OLD BELOW = CHECK TO SEE WHAT TO KEEP




# TODO deprecate
def pwm_position_squeeze(inputs, params):
    """Squeeze position
    """
    assert inputs.get("features") is not None

    # features
    features = inputs["features"]
    outputs = dict(inputs)
    
    squeeze_type = params.get("squeeze_type", "max")
    if squeeze_type == "max":
        max_vals = tf.reduce_max(tf.abs(features), axis=2, keepdims=True) # {N, task, 1, M}
        max_mask = tf.cast(
            tf.greater_equal(tf.abs(features), max_vals),
            tf.float32) #{N, task, pos, M}
        features = tf.reduce_sum(
            tf.multiply(max_mask, features), axis=2) # {N, task, M}
    elif squeeze_type == "mean":
        features = tf.reduce_mean(features, axis=2)
    elif squeeze_type == "sum":
        features = tf.reduce_sum(features, axis=2)

    outputs["features"] = features

    return outputs, params


# tODO deprecate
def pwm_relu(inputs, params):
    """Only keep positive
    """
    assert inputs.get("features") is not None

    # features
    features = inputs["features"]
    outputs = dict(inputs)

    # relu
    features = tf.nn.relu(features)

    outputs["features"] = features
    
    return outputs, params



# TODO consider whether this is actually useful
def multitask_global_pwm_scores(inputs, params):
    """Also get global pwm scores
    """
    features = inputs["features"]
    outputs = dict(inputs)

    append = params.get("append", True)
    count_thresh = params.get("count_thresh", 2)
    
    # per example, only keep positions that have been seen more than once
    features_by_example = [tf.expand_dims(tensor, axis=0)
                           for tensor in tf.unstack(features)] # {1, task, M}

    # for each example, get sum across tasks
    # TODO separate this out as different function
    masked_features_list = []
    for example_features in features_by_example:
        motif_counts = tf.reduce_sum(
            tf.cast(tf.not_equal(example_features, 0), tf.float32),
            axis=1, keepdims=True) # sum across tasks {1, 1, M}
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
    reduce_type = params.get("reduce_type", "sum")

    if reduce_type == "sum":
        features_max = tf.reduce_sum(features, axis=1, keepdims=True)
    elif reduce_type == "max":
        features_max = tf.reduce_max(features, axis=1, keepdims=True)
    elif reduce_type == "mean":
        features_max = tf.reduce_mean(features, axis=1, keepdims=True)

    # append or replace
    if append:
        features = tf.concat([features, features_max], axis=1)
    else:
        features = features_max

    outputs["features"] = features
        
    # things to keep
    if params.get("keep_global_pwm_scores") is not None:
        # attach to config
        outputs[params["keep_global_pwm_scores"]] = features_max #{N, pos, motif}
        params["keep_global_pwm_scores"] = None # TODO fix this, this is because there is overwriting
        
    return outputs, params
