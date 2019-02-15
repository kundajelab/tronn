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
        inputs, params = self.preprocess(inputs, params)
        outputs, params = self.convolve_motifs(inputs, params)
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
            shuffles,
            back_prop=False,
            parallel_iterations=1) # {N*M*task}

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
        if False:
            outputs[DataKeys.ACTIVE_SHUFFLES] = tf.nn.relu(shuffle_results[DataKeys.FEATURES]) # TODO remove this?
        else:
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

        # delete active shuffles
        del outputs[DataKeys.ACTIVE_SHUFFLES]
        
        return outputs, params


class DeltaMotifImportanceMapper(MotifScanner):
    """DMIM - given mutational results, scan for motifs and return results"""

    def preprocess(self, inputs, params):
        """assertions, and blank the motif site
        """
        assert inputs.get(DataKeys.FEATURES) is not None # {N, mutM, task, pos, 4}
        assert inputs.get(DataKeys.MUT_MOTIF_MASK) is not None # {N, mut_M, 1, pos, 1}

        # features - these are dfim scores
        features = inputs[DataKeys.FEATURES]
        mask = tf.cast(tf.equal(inputs[DataKeys.MUT_MOTIF_MASK], 0), tf.float32)
        outputs = dict(inputs)
        
        # blank out motif sites using pre-prepped mask
        features = tf.multiply(features, mask) # {N, mutM, task, pos, 4}
        
        # reshape for scanning
        features_shape = features.get_shape().as_list()
        params["batch_size"] = features_shape[0]
        features = tf.reshape(features, [-1]+features_shape[2:]) # {N*mutM, task, pos, 4}
        outputs[DataKeys.FEATURES] = features

        # preprocess
        outputs, params = super(DeltaMotifImportanceMapper, self).preprocess(
            outputs, params)

        return outputs, params

    
    def postprocess(self, inputs, params):
        """make sure you only keep the hits and reduce sum?
        """
        # utilizing existing pwm hits, mask the scores
        features = tf.multiply(
            inputs[DataKeys.FEATURES],
            inputs[DataKeys.ORIG_SEQ_PWM_HITS]) # {N*mutM, task, pos, M}

        # reshape back
        original_batch_size = params["batch_size"]
        features_shape = features.get_shape().as_list()
        features = tf.reshape(
            features, [original_batch_size, -1] + features_shape[1:])

        # sum across positions
        features = tf.reduce_sum(features, axis=3) # {N, mutM, task, M}

        # save out to appropriate tensors
        inputs[DataKeys.FEATURES] = features
        inputs[DataKeys.DMIM_SCORES] = features

        return inputs, params

    
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
    
    
    # squeeze the raw and weighted to get summed motif score for the example
    # this happens here because you need to filter the scores
    # with the hits from raw sequence first
    # TODO can try max instead of sum?
    outputs[DataKeys.ORIG_SEQ_PWM_SCORES_SUM] = tf.reduce_sum(
        outputs[DataKeys.ORIG_SEQ_PWM_SCORES_THRESH], axis=2) # {N, 1, M}

    # relu on the weighted scores for now
    # TODO remove this?
    if False:
        outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH] = tf.nn.relu(
            outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH]) # {N, task, pos, M}

    outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM] = tf.reduce_sum(
        outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH], axis=2) # {N, task, M}
    
    # features coming out: the summed weighted scores
    outputs[DataKeys.FEATURES] = outputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_SUM] # {N, task, M}

    # get max positions and values (for downstream analyses), also null positions
    outputs, _ = get_pwm_max_vals_and_positions(outputs, params)
    outputs, _ = get_pwm_null_positions(outputs, params)
    
    # filter for just those with a motif present
    if params.get("use_filtering", True):
        outputs, _ = filter_by_any_motif_present(outputs, params)
    
    return outputs, params


def get_pwm_max_vals_and_positions(inputs, params):
    """read off the max positions (and vals, comes for free)
    for downstream analyses
    """
    assert inputs.get(DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH) is not None
    
    # get features
    features = inputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH] # {N, task, pos, M}    
    outputs = dict(inputs)

    # reduce and transpose so position is last
    features = tf.reduce_max(features, axis=1) # {N, pos, M}
    features = tf.transpose(features, perm=[0,2,1]) # {N, M, pos}
    
    # get the max positions ON THE ACTIVE REGION
    # NOTE that these are indices for the short region!!
    vals, indices = tf.nn.top_k(features, k=1, sorted=True)
    indices = tf.add(indices, int(params["filter_width"] / 2.))
    
    # adjust for clipping
    if params.get("left_clip") is not None:
        indices = tf.add(indices, params["left_clip"])
    
    outputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL] = vals # {N, M, 1}
    outputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX] = indices # {N, M, 1}

    return outputs, params


def get_pwm_null_positions(inputs, params):
    """extract positions that are null (exactly zero) across the PWMs tested

    might be better to do this on importance scores?
    actually though in practice i am selecting positions based on pwms
    """
    assert inputs.get(DataKeys.WEIGHTED_SEQ_PWM_SCORES) is not None
    
    # get features
    features = inputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES] # {N, task, pos, M}    
    outputs = dict(inputs)
    min_pos = params.get("null_k", 10)
    
    # get real null positions
    real_null_features = tf.equal(features, 0) # {N, task, pos, M}
    real_null_features = tf.reduce_all(real_null_features, axis=[1,3]) # {N, pos}
    num_null = tf.reduce_sum(tf.cast(real_null_features, tf.float32), axis=1) # {N}
    #outputs["real_null"] = real_null_features
    #outputs["num_null"] = num_null

    # get pseudo null positions (smallest total pwm score)
    pseudo_null_features = tf.reduce_min(tf.abs(features), axis=1) # {N, pos, M}
    pseudo_null_features = tf.reduce_sum(pseudo_null_features, axis=2) # {N, pos}
    pseudo_null_features = tf.equal(
        pseudo_null_features,
        tf.reduce_min(pseudo_null_features, axis=1, keepdims=True)) # {N, pos}
    #outputs["pseudo_null"] = pseudo_null_features

    # select depending on num null
    features = tf.where(
        tf.greater(num_null, 0),
        x=real_null_features,
        y=pseudo_null_features)

    def select_null_positions(null_features):
        """use this on single examples (ie, need to combine this with map fn)
        """
        # null features {pos}
        null_indices = tf.where(null_features) # {some positions}
        chosen_null_indices = []
        for i in range(min_pos):
            null_indices = tf.random_shuffle(null_indices)
            chosen_null_indices.append(null_indices[0])
        chosen_null_indices = tf.concat(chosen_null_indices, axis=0)
        return chosen_null_indices

    # use map fn to select null positions
    null_indices = tf.map_fn(
        select_null_positions,
        features,
        dtype=tf.int64) # {N, pos}

    # adjust for clipping
    if params.get("left_clip") is not None:
        null_indices = tf.add(null_indices, params["left_clip"])

    null_indices = tf.expand_dims(null_indices, axis=2)
    outputs[DataKeys.NULL_PWM_POSITION_INDICES] = null_indices # {N, null, 1}
    
    return outputs, params


def attach_null_indices(inputs, params):
    """attach null indices (and vals) to appropriate tensors before
    running the mutations
    """
    assert inputs.get(DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT) is not None
    assert inputs.get(DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT) is not None
    assert inputs.get(DataKeys.MUT_MOTIF_PRESENT) is not None

    # get null indices and save out number of null indices
    null_indices = tf.cast(inputs[DataKeys.NULL_PWM_POSITION_INDICES], tf.int64)
    params["num_null_mut"] = null_indices.get_shape().as_list()[1]
    
    # attach to vals and indices and mut motif present
    inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT] = tf.concat(
        [inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT],
         null_indices], axis=1) # {N, mut_M+null, k}
    inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT] = tf.concat(
        [inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT],
         tf.ones(null_indices.get_shape())], axis=1) # {N, mut_M+null, k}
    inputs[DataKeys.MUT_MOTIF_PRESENT] = tf.reduce_any(
        tf.greater(inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT], 0),
        axis=2) # {N, mut_M+null}
    
    return inputs, params


def extract_null_results(inputs, params):
    """after running a function, extract out the null muts
    """
    # use motif_mut as key for which tensors to separate?
    num_null = params.get("num_null_mut")

    for key in inputs.keys():
        if "motif_mut" in key:
            null_key = key.replace("motif_mut", "null_mut")
            null_tensors = inputs[key][:,-num_null:]
            motif_mut_tensors = inputs[key][:,:-num_null]
            
            inputs[key] = motif_mut_tensors
            inputs[null_key] = null_tensors

    # also need to adjust features
    inputs[DataKeys.FEATURES] = inputs[DataKeys.FEATURES][:,:-num_null]

    return inputs, params


def get_sig_mut_motifs(inputs, params):
    """make CIs on the null mutants and use to filter the actual scores
    """
    z_thresh = 1.645 # 90% CI
    dmim_mut_scores = inputs[DataKeys.DMIM_SCORES]
    dmim_null_scores = inputs[DataKeys.DMIM_SCORES.replace("motif_mut", "null_mut")]
    outputs = dict(inputs)
    
    # get mean, var
    mean, var = tf.nn.moments(dmim_null_scores, axes=[1], keep_dims=True) # {N, 1, task, M}
    std = tf.sqrt(var)
    
    # cutoff
    passed_threshold = tf.subtract(dmim_mut_scores, mean)
    passed_threshold = tf.greater_equal(tf.abs(passed_threshold), z_thresh*std)

    # it's only real if the motif was present
    passed_threshold = tf.multiply(
        tf.cast(passed_threshold, tf.float32),
        dmim_mut_scores)
    
    outputs[DataKeys.DMIM_SCORES_SIG] = passed_threshold

    return outputs, params


def get_sig_mut_logits(inputs, params):
    """given null mutation logits, build a distribution and
    get cutoff for significant change
    """
    # TODO need to think about how this shifts if have ensemble of results.
    z_thresh = 0.645 # 90% CI
    null_mut_logits = inputs[DataKeys.MUT_MOTIF_LOGITS.replace("motif_mut", "null_mut")]
    motif_mut_logits = inputs[DataKeys.MUT_MOTIF_LOGITS]
    outputs = dict(inputs)

    # get mean and std
    mean, std = tf.nn.moments(null_mut_logits, axes=[1], keep_dims=True) # {N, 1, logit}
    
    # cutoff
    passed_threshold = tf.subtract(motif_mut_logits, mean)
    passed_threshold = tf.greater_equal(tf.abs(passed_threshold), z_thresh*std)

    # it's only real if the motif was present
    passed_threshold = tf.multiply(
        tf.cast(passed_threshold, tf.int64),
        tf.expand_dims(tf.cast(inputs[DataKeys.MUT_MOTIF_PRESENT], tf.int64), axis=-1))
    
    outputs[DataKeys.MUT_MOTIF_LOGITS_SIG] = passed_threshold
    
    return outputs, params


def cleanup_null_muts(inputs, params):
    """ delete a bunch of tensors
    """
    

    

    return


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




def filter_for_any_sig_pwms(inputs, params):
    """require that there is a param with a sig pwm vector, check if ANY motif exists
    """
    pwm_hits_key = DataKeys.ORIG_SEQ_PWM_HITS

    # assertions
    assert inputs.get(pwm_hits_key) is not None
    assert params.get("sig_pwms") is not None

    # get features and reduce
    features = inputs[pwm_hits_key] # {N, task, pos, M}
    features = tf.reduce_any(
        tf.not_equal(features, 0),
        axis=(1,2))
    features = tf.cast(features, tf.float32)
    sig_pwms = tf.expand_dims(
        params["sig_pwms"].astype(np.float32), axis=0) # {1, M}
    
    # multiply to mask
    features = tf.multiply(features, sig_pwms) # {N, M}
    
    # reduce
    has_sig_pwm = tf.reduce_any(features > 0, axis=1) # {N}
    inputs["condition_mask"] = has_sig_pwm
    
    # filter            
    params.update({"name": "any_sig_pwm_filter"})
    outputs, params = filter_and_rebatch(inputs, params)

    return outputs, params



def filter_for_significant_pwms_OLD(inputs, params):
    """this is specifically if you have clusters
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


# TODO just throw this in inference nets?
def run_dmim(inputs, params):
    """run dmim
    """
    scanner = DeltaMotifImportanceMapper(features_key=DataKeys.FEATURES)
    outputs, params = scanner.scan(inputs, params)
    
    return outputs, params



def filter_by_any_motif_present(inputs, params):
    """check if any motif was marked present to filter out zeros
    NOTE: requires motif be POSITIVE
    
    """
    # features
    features = inputs[DataKeys.FEATURES] # {N, task, M}
    
    # adjust
    batch_size = features.get_shape().as_list()[0]

    # see if any positive hits
    any_motif_present = tf.reduce_any(
        tf.greater(features, 0),
        axis=range(len(features.get_shape()))[1:])
    
    # condition mask
    inputs["condition_mask"] = any_motif_present
    params.update({"name": "motif_presence_filter", "use_queue": True})
    outputs, _ = filter_and_rebatch(inputs, params)
    
    return outputs, params
