"""description: statistics for nets
"""

import tensorflow as tf



def get_gaussian_confidence_intervals(inputs, params):
    """build intervals
    """
    assert params.get("ci_in_key") is not None
    assert params.get("ci_out_key") is not None

    # get inputs
    input_key = params["ci_in_key"]
    out_key = params["ci_out_key"]
    std_thresh = params.get("std_thresh", 2.576)
    axis = params.get("axis", 1)
    features = inputs[input_key]
    outputs = dict(inputs)
    num_samples = features.get_shape().as_list()[axis]
    
    # get mean and standard error
    mean, var = tf.nn.moments(features, [axis])
    std = tf.sqrt(var)
    se = tf.divide(std, tf.sqrt(tf.cast(num_samples, tf.float32)))
    se_limit = tf.multiply(se, std_thresh)
    
    # get CI
    upper_bound = tf.add(mean, se_limit)
    lower_bound = tf.subtract(mean, se_limit) # {N, 10, 160}
    confidence_intervals = tf.stack(
        [upper_bound, lower_bound], axis=1) # {N, 2, ...}
    outputs[out_key] = confidence_intervals
    
    return outputs, params


def check_confidence_intervals(inputs, params):
    """check whether confidence interval overlaps desired val
    """
    # get inputs
    vals_key = params.get("true_val_key_for_ci", "NULL")
    ci_key = params["ci_out_key"]
    out_key = params["ci_pass_key"]
    
    ci = inputs[ci_key]
    vals = inputs.get(vals_key, 0.0)
    
    # check if inside the confidence interval
    pass_upper_bound = tf.less_equal(vals, ci[:,0])
    pass_lower_bound = tf.greater_equal(vals, ci[:,1])
    passes_both = tf.logical_and(pass_upper_bound, pass_lower_bound)
    inputs[out_key] = passes_both
    
    return inputs, params

