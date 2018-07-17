"""Contains nets that perform PWM convolutions
"""

import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim # TODO try to factor out slim

from tronn.nets.threshold_nets import build_null_distribution_threshold_fn

from tronn.util.initializers import pwm_simple_initializer

from tronn.util.tf_utils import get_fan_in

from tronn.util.utils import DataKeys


class MotifScanner(object):
    """base class, scans sequence"""
    
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

        # fill in accordingly
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


        # relu for just positive scores?
        outputs, params = pwm_relu(inputs, params)
        print outputs[DataKeys.FEATURES]

        # TODO - do this outside
        # squeeze to get summed score across positions
        #params.update({"squeeze_type": "sum"})
        #outputs, params = pwm_position_squeeze(outputs, params)
        #print outputs[DataKeys.FEATURES]

        return outputs, params


    def scan(self, inputs, params):
        """put all the pieces together
        """
        # run preprocess
        inputs, params = self.preprocess(inputs, params)
        # convolve
        outputs, params = self.convolve_motifs(inputs, params)
        # postprocess
        outputs, params = self.postprocess(outputs, params)

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
            **kwargs):
        """init
        """
        self.shuffles_key = shuffles_key
        self.out_scores_key = out_scores_key
        self.out_hits_key = out_hits_key
        self.out_scores_thresh_key = out_scores_thresh_key
        self.pval = pval
        super(MotifScannerWithThresholds, self).__init__(
            features_key=features_key, **kwargs)
    
    
    @staticmethod
    def threshold(inputs, params):
        """threshold scores using motif scores from 
        shuffled sequence (ie null distribution)
        """
        # requires you have scores on the examples AND the shuffles
        assert inputs.get(DataKeys.ACTIVE_SHUFFLES) is not None
        assert inputs.get(DataKeys.FEATURES) is not None

        # adjust the shuffle key so that when running map_fn
        # you get a threshold for each example for each task
        shuffles = tf.transpose(inputs[DataKeys.ACTIVE_SHUFFLES], perm=[0,2,4,1,3]) # becomes {N, task, M, shuf, pos}
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

        pass_positive_thresh = tf.cast(tf.greater(features, thresholds), tf.float32)
        pass_negative_thresh = tf.cast(tf.less(features, -thresholds), tf.float32)
        pass_thresh = tf.add(pass_positive_thresh, pass_negative_thresh)
        outputs[DataKeys.FEATURES] = pass_thresh

        return outputs, params

    
    def scan(self, inputs, params):
        """run the shuffle scanner and get the results
        """
        # get sequence results
        outputs = super(MotifScannerWithThresholds, self).scan(inputs, params)[0]
        
        # get shuffle results
        shuffle_scanner = ShufflesMotifScanner(features_key=self.shuffles_key)
        shuffle_results = shuffle_scanner.scan(inputs, params)[0]
        outputs[DataKeys.ACTIVE_SHUFFLES] = shuffle_results[DataKeys.FEATURES]

        # get thresholds
        params.update({"pval": self.pval})
        outputs[self.out_hits_key] = MotifScannerWithThresholds.threshold(
            outputs, params)[0][DataKeys.FEATURES]
        outputs[self.out_scores_thresh_key] = tf.multiply(
            outputs[self.out_hits_key],
            outputs[DataKeys.FEATURES])

        # also save out raw scores
        outputs[self.out_scores_key] = outputs[DataKeys.FEATURES]
        
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

    # features coming out: weighted pwm scores thresholded
    outputs[DataKeys.FEATURES] = outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH]

    # SEPARATE: squeeze
    # SEPARATE: motif density, max motif density
    
    return outputs, params













# OLD CODE BELOW


        

def pwm_convolve_old(inputs, params):
    """Convolve with PWMs and normalize with vector projection:

      projection = a dot b / | b |
    """
    # TODO options for running (1) FWD, (2) REV, (3) COMB
    assert inputs.get(DataKeys.FEATURES) is not None
    assert params.get("pwms") is not None

    # features
    features = inputs[DataKeys.FEATURES]
    outputs = dict(inputs)

    # params
    pwm_list = params["pwms"]
    num_filters = len(pwm_list)
    reuse = params.get("reuse_pwm_layer", False)
    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]
    params["filter_width"] = max_size
            
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


def threshold_pwm_scores_old(inputs, params):
    # requires you have scores on the examples AND the shuffles
    assert inputs.get(DataKeys.ACTIVE_SHUFFLES) is not None
    assert inputs.get(DataKeys.FEATURES) is not None

    # adjust the shuffle key so that when running map_fn
    # you get a threshold for each example for each task
    shuffles = tf.transpose(inputs[DataKeys.ACTIVE_SHUFFLES], perm=[0,2,4,1,3]) # becomes {N, task, M, shuf, pos}
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

    pass_positive_thresh = tf.cast(tf.greater(features, thresholds), tf.float32)
    pass_negative_thresh = tf.cast(tf.less(features, -thresholds), tf.float32)
    pass_thresh = tf.add(pass_positive_thresh, pass_negative_thresh)
    outputs[DataKeys.FEATURES] = pass_thresh

    return outputs, params


def pwm_convolve_twotailed_old(inputs, params):
    """Convolve both pos and negative scores with PWMs. Prevents getting scores
    when the negative correlation is stronger.
    """
    assert inputs.get(DataKeys.FEATURES) is not None

    # features
    features = inputs[DataKeys.FEATURES]
    outputs = dict(inputs)
    
    # positive weighted sequence
    pos_scores = pwm_convolve(
        {DataKeys.FEATURES: tf.nn.relu(features)},
        params)[0][DataKeys.FEATURES]
    pos_scores = tf.nn.relu(pos_scores)

    # negative weighted sequence
    params.update({"reuse_pwm_layer": True})
    neg_scores = pwm_convolve(
        {DataKeys.FEATURES: tf.nn.relu(-features)},
        params)[0][DataKeys.FEATURES]
    neg_scores = tf.nn.relu(neg_scores)

    # condition: is positive score higher than neg score
    is_positive_max = tf.greater_equal(pos_scores, neg_scores)

    # fill in accordingly
    outputs[DataKeys.FEATURES] = tf.where(is_positive_max, pos_scores, -neg_scores)
    
    return outputs, params


def get_pwm_scores_old(inputs, params):
    """main function to run all
    """
    # move inputs to outputs
    outputs = dict(inputs)
    
    # run raw sequence through pwm layer
    outputs[DataKeys.ORIG_SEQ_PWM_SCORES] = pwm_convolve_twotailed(
        {DataKeys.FEATURES: inputs[DataKeys.ORIG_SEQ_ACTIVE]},
        params)[0][DataKeys.FEATURES]

    # run raw sequence shuffles through pwm layer
    shuffle_shape = inputs[DataKeys.ORIG_SEQ_ACTIVE_SHUF].get_shape().as_list()
    shuffles_reshaped = tf.reshape(
        inputs[DataKeys.ORIG_SEQ_ACTIVE_SHUF],
        [shuffle_shape[0], shuffle_shape[1]*shuffle_shape[2], shuffle_shape[3], shuffle_shape[4]])
    shuffle_pwm_scores = pwm_convolve_twotailed(
        {DataKeys.FEATURES: shuffles_reshaped},
        params)[0][DataKeys.FEATURES]
    shuffle_shape[3:] = shuffle_pwm_scores.get_shape().as_list()[2:]
    outputs[DataKeys.ORIG_SEQ_SHUF_PWM_SCORES] = tf.reshape(shuffle_pwm_scores, shuffle_shape)

    # theshold, return the threshold mask (ie, motif hits) and scores that pass threshold
    outputs[DataKeys.ORIG_SEQ_PWM_HITS] = threshold_pwm_scores(
        {DataKeys.FEATURES: outputs[DataKeys.ORIG_SEQ_PWM_SCORES],
         DataKeys.ACTIVE_SHUFFLES: outputs[DataKeys.ORIG_SEQ_SHUF_PWM_SCORES]},
        {})[0][DataKeys.FEATURES]
    outputs[DataKeys.ORIG_SEQ_PWM_SCORES_THRESH] = tf.multiply(
        outputs[DataKeys.ORIG_SEQ_PWM_SCORES],
        outputs[DataKeys.ORIG_SEQ_PWM_HITS])
    
    # run weighted sequence + shuffles through pwm layer
    # TODO consider weighting by base pair overlap for the importance score set?
    outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES] = pwm_convolve_twotailed(
        {DataKeys.FEATURES: inputs[DataKeys.WEIGHTED_SEQ_ACTIVE]},
        params)[0][DataKeys.FEATURES]

    shuffle_shape = inputs[DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF].get_shape().as_list()
    shuffles_reshaped = tf.reshape(
        inputs[DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF],
        [shuffle_shape[0], shuffle_shape[1]*shuffle_shape[2], shuffle_shape[3], shuffle_shape[4]])
    shuffle_pwm_scores = pwm_convolve_twotailed(
        {DataKeys.FEATURES: shuffles_reshaped},
        params)[0][DataKeys.FEATURES]
    shuffle_shape[3:] = shuffle_pwm_scores.get_shape().as_list()[2:]
    outputs[DataKeys.WEIGHTED_SEQ_SHUF_PWM_SCORES] = tf.reshape(shuffle_pwm_scores, shuffle_shape)

    # theshold, return the threshold mask (ie, motif hits) and scores that pass threshold
    outputs[DataKeys.WEIGHTED_SEQ_PWM_HITS] = threshold_pwm_scores(
        {DataKeys.FEATURES: outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES],
         DataKeys.ACTIVE_SHUFFLES: outputs[DataKeys.WEIGHTED_SEQ_SHUF_PWM_SCORES]},
        {})[0][DataKeys.FEATURES]
    outputs[DataKeys.WEIGHTED_SEQ_PWM_HITS] = tf.multiply(
        outputs[DataKeys.WEIGHTED_SEQ_PWM_HITS],
        outputs[DataKeys.ORIG_SEQ_PWM_HITS])
    outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH] = tf.multiply(
        outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES],
        outputs[DataKeys.WEIGHTED_SEQ_PWM_HITS])

    # delete the pwm shuffles
    del outputs[DataKeys.ORIG_SEQ_SHUF_PWM_SCORES]
    del outputs[DataKeys.WEIGHTED_SEQ_SHUF_PWM_SCORES]

    # and set up main features
    outputs[DataKeys.FEATURES] = outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH]
    
    return outputs, params





# OLD


def pwm_match_filtered_convolve(inputs, params):
    """Run pwm convolve twice, with importance scores and without.
    Choose max for motif across positions using raw sequence
    """
    assert params.get("raw-sequence-clipped-key") is not None
    assert inputs.get(params["raw-sequence-clipped-key"]) is not None
    assert params.get("pwms") is not None
    
    # features
    features = inputs["features"]
    raw_sequence = inputs[params["raw-sequence-clipped-key"]]
    is_training = params.get("is_training", False)
    outputs = dict(inputs)
    
    # run on raw sequence
    #if raw_sequence is not None:
    binarized_features = raw_sequence

    
    pwm_binarized_feature_scores, _ = pwm_convolve_inputxgrad(
        {"features": binarized_features}, params) # {N, 1, pos, M}

    # adjust the raw scores and save out
    if params.get("raw-pwm-scores-key") is not None:
        raw_bp_overlap, _ = get_bp_overlap({"features": binarized_features}, params) # this does nothing on raw sequence
        raw_scores = tf.multiply(
            pwm_binarized_feature_scores,
            raw_bp_overlap)
        raw_scores = tf.squeeze(tf.reduce_max(raw_scores, axis=2), axis=1) # {N, M}
        outputs[params["raw-pwm-scores-key"]] = raw_scores

    # multiply by raw sequence matches
    pwm_binarized_feature_maxfilt_mask = tf.cast(
        tf.greater(pwm_binarized_feature_scores, [0]), tf.float32)

    # testing using the actual raw scores?
    #pwm_binarized_feature_maxfilt_mask = tf.nn.relu(
    #    pwm_binarized_feature_scores)
    
    # run on impt weighted features
    #with tf.variable_scope("impt_weighted"):
    pwm_impt_weighted_scores, _ = pwm_convolve_inputxgrad(
        {"features": features}, params)
    
    # and filter through mask
    filt_features = tf.multiply(
        pwm_binarized_feature_maxfilt_mask,
        pwm_impt_weighted_scores)

    # at this stage also need to perform the weighting by bp presence
    impt_bp_overlap, _ = get_bp_overlap({"features": features}, params)
    features = tf.multiply(
        filt_features,
        impt_bp_overlap)

    outputs["features"] = features
    
    # keep for grammars
    if params.get("positional-pwm-scores-key") is not None:
        outputs[params["positional-pwm-scores-key"]] = features
        
    return outputs, params












def pwm_convolve_old(inputs, params):
    """Convolve with PWMs and normalize with vector projection:

      projection = a dot b / | b |
    """
    assert inputs.get(DataKeys.FEATURES) is not None
    assert params.get("pwms") is not None
    
    features = inputs["features"]
    pwm_list = params.get("pwms")
    reuse = params.get("reuse_pwm_layer", False)
    # TODO options for running (1) FWD, (2) REV, (3) COMB
    
    assert pwm_list is not None
        
    # get various sizes needed to instantiate motif matrix
    num_filters = len(pwm_list)
    #logging.info("Total PWMs: {}".format(num_filters))
    
    max_size = 0
    for pwm in pwm_list:
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]
    #logging.info("Filter size: {}".format(max_size))
    params["filter_width"] = max_size
            
    # make the convolution net for dot product, normal filters
    # here, the pwms are already normalized to unit vectors for vector projection
    conv1_filter_size = [1, max_size]
    with tf.variable_scope("pwm_layer", reuse=reuse):
        with slim.arg_scope(
                [slim.conv2d],
                padding='VALID',
                activation_fn=None,
                weights_initializer=pwm_simple_initializer(
                    conv1_filter_size,
                    pwm_list,
                    get_fan_in(features),
                    unit_vector=True,
                    length_norm=True), # check this
                biases_initializer=None,
                trainable=False):
            # pwm cross correlation
            features = slim.conv2d(
                features, num_filters, conv1_filter_size)
        
    return features, params


def pwm_convolve_inputxgrad(inputs, params):
    """Convolve both pos and negative scores with PWMs. Prevents getting scores
    when the negative correlation is stronger.
    """
    # do positive sequence. ignore negative scores. only keep positive results
    #with tf.variable_scope("pos_seq_pwm"):
    features = inputs["features"]
    
    pos_seq_scores, _ = pwm_convolve({"features": tf.nn.relu(features)}, params)
    pos_seq_scores = tf.nn.relu(pos_seq_scores["features"])
        
    # do negative sequence. ignore positive scores. only keep positive (ie negative) results
    #with tf.variable_scope("neg_seq_pwm"):
    params.update({"reuse_pwm_layer": True})
    neg_seq_scores, _, = pwm_convolve({"features": tf.nn.relu(-features)}, params)
    neg_seq_scores = tf.nn.relu(neg_seq_scores["features"])
        
    # and then take max (best score, whether pos or neg) do not use abs; and keep sign
    max_seq_scores = tf.reduce_max(tf.stack([pos_seq_scores, neg_seq_scores], axis=0), axis=0) # {N, task, seq_len/pool_width, M}
    
    # and now get the sign and mask
    pos_scores_masked = tf.multiply(
        pos_seq_scores,
        tf.cast(tf.equal(pos_seq_scores, max_seq_scores), tf.float32))
    neg_scores_masked = tf.multiply(
        -neg_seq_scores,
        tf.cast(tf.equal(neg_seq_scores, max_seq_scores), tf.float32))

    features = tf.add(pos_scores_masked, neg_scores_masked)
    
    return features, params


def pwm_motif_max(features, labels, config, is_training=False):
    """Get max for each motif? this likely reduces noise, but is it necessary?
    # is this sorta subsumed by position squeeze later?
    """
    features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)] # list of {N, 1, pos, M}

    # TODO build a function to filter for two sided max?
    features_pos_max = []
    for i in xrange(len(features)):
        task_features = features[i]
        # fix this? is wrong?
        features_max_vals = tf.reduce_max(tf.abs(task_features), axis=2, keep_dims=True) # {N, 1, 1, M}
        features_max_mask = tf.multiply(
            tf.cast(tf.equal(tf.abs(task_features), features_max_vals), tf.float32),
            tf.cast(tf.not_equal(task_features, 0), tf.float32))
        task_features = tf.multiply(task_features, features_max_mask)
        features_pos_max.append(task_features)
        
    # restack
    features = tf.concat(features_pos_max, axis=1) # {N, task, pos, M}

    return features, labels, config


def get_bp_overlap(inputs, params):
    """Re-weight by num of importance-weighted base pairs that are nonzero
    """
    features = inputs["features"]
    
    features_present = tf.cast(tf.not_equal(features, [0]), tf.float32)
    max_size = params.get("filter_width")
    assert max_size is not None
    nonzero_bp_fraction_per_window = tf.reduce_sum(
        slim.avg_pool2d(
            features_present, [1, max_size], stride=[1,1], padding="VALID"),
        axis=3, keepdims=True)
    #features = tf.multiply(
    #    features,
    #    nonzero_bp_fraction_per_window)
    
    return nonzero_bp_fraction_per_window, params


def pwm_match_filtered_convolve(inputs, params):
    """Run pwm convolve twice, with importance scores and without.
    Choose max for motif across positions using raw sequence
    """
    assert params.get("raw-sequence-clipped-key") is not None
    assert inputs.get(params["raw-sequence-clipped-key"]) is not None
    assert params.get("pwms") is not None
    
    # features
    features = inputs["features"]
    raw_sequence = inputs[params["raw-sequence-clipped-key"]]
    is_training = params.get("is_training", False)
    outputs = dict(inputs)
    
    # run on raw sequence
    #if raw_sequence is not None:
    binarized_features = raw_sequence
    print "WARNING DID NOT DELETE RAW SEQUENCE"
        #if params["keep_ism_scores"] is None:
        #    del outputs["onehot_sequence"] # remove this from outputs now
        #else:
        #    print "WARNING DID NOT DELETE RAW SEQUENCE"
    
    pwm_binarized_feature_scores, _ = pwm_convolve_inputxgrad(
        {"features": binarized_features}, params) # {N, 1, pos, M}

    # adjust the raw scores and save out
    if params.get("raw-pwm-scores-key") is not None:
        raw_bp_overlap, _ = get_bp_overlap({"features": binarized_features}, params)
        raw_scores = tf.multiply(
            pwm_binarized_feature_scores,
            raw_bp_overlap)
        raw_scores = tf.squeeze(tf.reduce_max(raw_scores, axis=2), axis=1) # {N, M}
        outputs[params["raw-pwm-scores-key"]] = raw_scores

    # multiply by raw sequence matches
    pwm_binarized_feature_maxfilt_mask = tf.cast(
        tf.greater(pwm_binarized_feature_scores, [0]), tf.float32)

    # testing using the actual raw scores?
    #pwm_binarized_feature_maxfilt_mask = tf.nn.relu(
    #    pwm_binarized_feature_scores)
    
    # run on impt weighted features
    #with tf.variable_scope("impt_weighted"):
    pwm_impt_weighted_scores, _ = pwm_convolve_inputxgrad(
        {"features": features}, params)
    
    # and filter through mask
    filt_features = tf.multiply(
        pwm_binarized_feature_maxfilt_mask,
        pwm_impt_weighted_scores)

    # at this stage also need to perform the weighting by bp presence
    impt_bp_overlap, _ = get_bp_overlap({"features": features}, params)
    features = tf.multiply(
        filt_features,
        impt_bp_overlap)

    outputs["features"] = features
    
    # keep for grammars
    if params.get("positional-pwm-scores-key") is not None:
        outputs[params["positional-pwm-scores-key"]] = features
        
    return outputs, params



def pwm_maxpool(features, labels, config, is_training=False):
    """Two tailed pooling operation when have both pos/neg scores
    """
    pool_width = config.get("pool_width", None)
    assert pool_width is not None

    # testing
    # big maxpool across 15 pwm width, but per position

    # end testing

    
    # get the max vals, both pos and neg, VALID padding
    maxpool_pos = slim.max_pool2d(features, [1, pool_width], stride=[1, pool_width]) # {N, task, seq_len/pool_width, M}
    maxpool_neg = slim.max_pool2d(-features, [1, pool_width], stride=[1, pool_width]) # {N, task, seq_len/pool_width, M}
    maxpool_abs = tf.reduce_max(tf.stack([maxpool_pos, maxpool_neg], axis=0), axis=0) # {N, task, seq_len/pool_width, M}
    
    # get the right values
    maxpool_pos_masked = tf.multiply(
        maxpool_pos,
        tf.cast(tf.equal(maxpool_pos, maxpool_abs), tf.float32))
    maxpool_neg_masked = tf.multiply(
        -maxpool_neg,
        tf.cast(tf.equal(maxpool_neg, maxpool_abs), tf.float32))
    features = tf.add(maxpool_pos_masked, maxpool_neg_masked)

    return features, labels, config


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


# TODO is this used?
def pwm_positional_max(features, labels, config, is_training=False):
    """Get max at a position
    """
    features = [tf.expand_dims(tensor, axis=1) for tensor in tf.unstack(features, axis=1)] # list of {N, 1, pos, M}

    # TODO build a function to filter for two sided max?
    features_pos_max = []
    for i in xrange(len(features)):
        task_features = features[i]
        # fix this? is wrong?
        features_max_vals = tf.reduce_max(tf.abs(task_features), axis=3, keep_dims=True) # {N, 1, pos, 1}
        features_max_mask = tf.multiply(
            tf.cast(tf.equal(tf.abs(task_features), features_max_vals), tf.float32),
            tf.cast(tf.not_equal(task_features, 0), tf.float32))
        task_features = tf.multiply(task_features, features_max_mask)
        features_pos_max.append(task_features)
        
    # restack
    features = tf.concat(features_pos_max, axis=1) # {N, task, pos, M}

    return features, labels, config


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
