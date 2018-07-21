"""Contains nets that perform PWM convolutions
"""

import logging

import tensorflow as tf

from tronn.nets.threshold_nets import build_null_distribution_threshold_fn
from tronn.util.initializers import pwm_simple_initializer
from tronn.util.tf_utils import get_fan_in

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
    def positive_threshold(inputs, params):
        """threshold scores using motif scores from 
        shuffled sequence (ie null distribution)
        """
        # requires you have scores on the examples AND the shuffles
        assert inputs.get(DataKeys.ACTIVE_SHUFFLES) is not None
        assert inputs.get(DataKeys.FEATURES) is not None

        # adjust the shuffle key so that when running map_fn
        # you get a threshold for each example for each task
        shuffles = tf.transpose(
            inputs[DataKeys.ACTIVE_SHUFFLES],
            perm=[0,2,4,1,3]) # becomes {N, task, M, shuf, pos}
        shuffles_shape = shuffles.get_shape().as_list()
        shuffles = tf.reshape(
            shuffles,
            [shuffles_shape[0]*shuffles_shape[1]*shuffles_shape[2],
             shuffles_shape[3]*shuffles_shape[4]])

        # features
        features = inputs[DataKeys.FEATURES]
        outputs = dict(inputs)

        # params
        pval_thresh = params.get("pval_thresh", 0.05)
        threshold_fn = build_null_distribution_threshold_fn(pval_thresh)

        # get thresholds for each example, for each task
        thresholds = tf.map_fn(
            threshold_fn,
            shuffles)

        # and apply (remember to reshape and transpose back to feature dim ordering)
        feature_shape = features.get_shape().as_list()
        thresholds = tf.reshape(thresholds, shuffles_shape[0:3]+[1])
        thresholds = tf.transpose(thresholds, perm=[0,1,3,2]) # {N, task, pos, M}

        # this should be a onesided threshold, since raw seq should just get positive thresh vals
        # for importance scores call twice with each half
        pass_positive_thresh = tf.cast(tf.greater(features, thresholds), tf.float32)
        #pass_negative_thresh = tf.cast(tf.less(features, -thresholds), tf.float32)
        #pass_thresh = tf.add(pass_positive_thresh, pass_negative_thresh)
        outputs[DataKeys.FEATURES] = pass_positive_thresh

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
        outputs[self.out_hits_key] = MotifScannerWithThresholds.positive_threshold(
            outputs, params)[0][DataKeys.FEATURES]
        outputs[self.out_scores_thresh_key] = tf.multiply(
            outputs[self.out_hits_key],
            outputs[DataKeys.FEATURES])

        # get twosided if required
        # TODO remerge back in? does not influence outcome
        if self.two_sided_thresh:
            # set shuffles to negative scores, and flip features
            outputs[DataKeys.ACTIVE_SHUFFLES] = tf.nn.relu(-shuffle_results[DataKeys.FEATURES])
            outputs[DataKeys.FEATURES] = -outputs[DataKeys.FEATURES]

            # add the results to hits
            # TODO keep as negative?
            outputs[self.out_hits_key] = tf.add(
                outputs[self.out_hits_key],
                MotifScannerWithThresholds.positive_threshold(
                    outputs, params)[0][DataKeys.FEATURES])

            # flip features and multiply again
            outputs[DataKeys.FEATURES] = -outputs[DataKeys.FEATURES]
            outputs[self.out_scores_thresh_key] = tf.multiply(
                outputs[self.out_hits_key],
                outputs[DataKeys.FEATURES])
        
        # also save out raw scores
        outputs[self.out_scores_key] = outputs[DataKeys.FEATURES]
        
        return outputs, params


class DeltaMotifImportanceMapper(MotifScanner):
    """DMIM - given mutational results, scan for motifs and return results"""

    def preprocess(self, inputs, params):
        """assertions, and blank the motif site
        """
        # check for correct input {N, task, pos, 4}

        # blank the motif site

        # clip edges?
        
        return

    
    def scan(self, inputs, params):
        """scan motifs
        """
        
        return


    def postprocess(self, inputs, params):
        """subtract orig from mut
        """

        return
    

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
    # with the hits from raw sequence
    # TODO can try max instead of sum?
    outputs[DataKeys.ORIG_SEQ_PWM_SCORES_SUM] = tf.reduce_sum(
        outputs[DataKeys.ORIG_SEQ_PWM_SCORES_THRESH], axis=2) # {N, 1, M}

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


# TODO is this necessary?
def get_bp_overlap(inputs, params):
    """Re-weight by num of importance-weighted base pairs that are nonzero
    """
    features = inputs["features"]
    
    features_present = tf.cast(tf.not_equal(features, [0]), tf.float32)
    max_size = params.get("filter_width")
    assert max_size is not None
    nonzero_bp_fraction_per_window = tf.reduce_sum(
        tf.layers.average_pooling2d(
            features_present, [1, max_size], [1,1], padding="VALID"),
        axis=3, keepdims=True)
    #features = tf.multiply(
    #    features,
    #    nonzero_bp_fraction_per_window)
    
    return nonzero_bp_fraction_per_window, params





# TODO more correctly, need to set up a check so that a motif across time is only kept if seen at least
# twice across time too

# TODO is this used?
def pwm_consistency_check(features, labels, config, is_training=False):
    """Try to keep most consistent motifs across tasks. max scores are accounted for later
    """

    # TODO is this useful? check multitask global importance to see thresholding
    
    # split by example
    features_by_example = [tf.expand_dims(tensor, axis=0) for tensor in tf.unstack(features)] # {1, task, pos, M}

    # for each example, get sum across tasks
    masked_features_list = []
    for example_features in features_by_example:
        motif_present = tf.cast(tf.not_equal(example_features, 0), tf.float32) # {1, task, pos, M}

        # TODO: probably just want max sum across the real scores, not counts.
        motif_counts = tf.reduce_sum(
            motif_present, axis=1, keep_dims=True) # sum across tasks {1, 1, pos, M}
        #motif_counts = tf.reduce_sum(
        #    tf.abs(example_features), axis=1, keep_dims=True) # sum across tasks {1, 1, pos, M}
        motif_max = tf.reduce_max(motif_counts, axis=3, keep_dims=True) # max across motifs {1, 1, pos, 1}
        # then mask based on max position
        motif_mask = tf.cast(tf.greater_equal(motif_counts, motif_max), tf.float32) # {1, 1, pos, M}
        # and mask
        masked_features = tf.multiply(motif_mask, example_features)
        masked_features_list.append(masked_features)

    # stack
    features = tf.concat(masked_features_list, axis=0) # {N, task, pos, M}

    # sometimes keep (for grammars)
    if config.get("keep_pwm_scores_full") is not None:
        # attach to config
        config["outputs"][config["keep_pwm_scores_full"]] = features # {N, task, pos, M}

    return features, labels, config


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
