"""Description: nets for normalizing results
"""

import h5py
import logging

import numpy as np
import tensorflow as tf

from tronn.util.utils import DataKeys


def _build_multitask_interpolation_fn(x_full, y_full):
    """assumes 2d matrices, adjusts in place
    """
    # build an interp for each task
    sorted_x_full = np.sort(x_full, axis=0)
    sorted_y_full = np.sort(y_full, axis=0)

    # build numpy based fn
    def interp_fn(x_vals):
        new_x_vals = np.zeros_like(x_vals)
        for i in range(x_vals.shape[1]):
            new_x_vals[:,i] = np.interp(
                x_vals[:,i],
                sorted_x_full[:,i],
                sorted_y_full[:,i])
        return new_x_vals
    
    return interp_fn


def interpolate_logits_to_labels(inputs, params):
    """quantile norm between the logit and the label
    to get them to match as well as possible
    note that this does NOT change the spearman cor
    (since rank order does not change) but may improve
    the pearson cor.
    """
    assert params.get("prediction_sample") is not None
    is_ensemble = params.get("is_ensemble", False)
    
    
    # set up keys
    if is_ensemble:
        logit_key = DataKeys.LOGITS_MULTIMODEL
        new_logit_key = DataKeys.LOGITS_MULTIMODEL_NORM
    else:
        logit_key = DataKeys.LOGITS
        new_logit_key = DataKeys.LOGITS_NORM
    
    # first, build the comparison vectors
    with h5py.File(params["prediction_sample"], "r") as hf:
        labels = hf[DataKeys.LABELS][:] # {N, ...}
        logits = hf[logit_key][:] # {N, ...}

    # build interp functions that go into py_func
    if is_ensemble:
        num_models = params["num_models"]
        model_norm_fns = []
        for model_idx in range(num_models):
            model_norm_fn = _build_multitask_interpolation_fn(
                logits[:,model_idx], labels)
            model_norm_fns.append(model_norm_fn)

        def interp(x_vals):
            norm_logits = []
            for model_idx in range(num_models):
                norm_logits.append(model_norm_fns[model_idx](x_vals[:,model_idx]))
            norm_logits = np.stack(norm_logits, axis=1) # {N, model, logit}
            return norm_logits

    else:
        model_norm_fn = _build_multitask_interpolation_fn(
            logits, labels)
        
        def interp(x_vals):
            norm_logits = model_norm_fn(x_vals) # {N, logit}
            return norm_logits
        
    # build py_func
    old_logits = inputs[logit_key]
    inputs[new_logit_key] = tf.py_func(
        func=interp,
        inp=[inputs[logit_key]],
        Tout=tf.float32,
        stateful=False,
        name="normalize_logits")
    
    # and have to reset the shape
    inputs[new_logit_key].set_shape(old_logits.get_shape())

    # and then if ensemble, adjust the logits
    if is_ensemble:
        inputs[DataKeys.LOGITS_NORM] = tf.reduce_mean(
            inputs[DataKeys.LOGITS_MULTIMODEL_NORM], axis=1)
    
    return inputs, params


def normalize_to_importance_logits(inputs, params):
    """normalize to the logits
    """
    # assertions
    assert params.get("weight_key") is not None
    assert params.get("importance_task_indices") is not None
    assert inputs.get(params["weight_key"]) is not None
    assert inputs.get("features") is not None

    logging.info("LAYER: normalize to the logits")
    
    # features
    features = inputs[DataKeys.FEATURES]
    features_shape = features.get_shape().as_list()
    outputs = dict(inputs)

    # params
    task_axis = params.get("task_axis", -1)
    
    # weights
    weights = tf.gather(
        inputs[params["weight_key"]],
        params["importance_task_indices"],
        axis=task_axis) # {N,.., task}
    weights_shape = weights.get_shape().as_list() # {N, shuf, logit} or {N, logit}
    diff_dims = len(features_shape) - len(weights_shape)
    weights = tf.reshape(weights, weights_shape + [1 for i in xrange(diff_dims)])
    logging.debug("...weights: {}".format(weights.get_shape()))
    
    axes = range(len(features_shape))[len(weights_shape):]
    logging.debug("...reduction over axes: {}".format(axes))

    # summed importances
    importance_sums = tf.reduce_sum(tf.abs(features), axis=axes, keepdims=True)
    logging.debug("...importance_sums: {}".format(importance_sums.get_shape()))
    
    # normalize
    features = tf.divide(features, importance_sums)
    features = tf.multiply(weights, features)

    # guard against inf
    features = tf.multiply(
        features,
        tf.cast(tf.greater(importance_sums, 1e-7), tf.float32))

    # guard against nan
    features = tf.where(
        tf.is_nan(features),
        tf.zeros_like(features),
        features)
    
    outputs[DataKeys.FEATURES] = features

    logging.debug("RESULTS: {}".format(outputs[DataKeys.FEATURES].get_shape()))
    
    return outputs, params
