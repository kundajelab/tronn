
import tensorflow as tf

from tronn.nets.sequence_nets import pad_examples
from tronn.nets.sequence_nets import unpad_examples


def get_variant_importance_scores(inputs, params):
    """
    """
    assert inputs.get("features") is not None
    assert inputs.get("variant_relative_pos") is not None
    
    features = tf.reduce_sum(inputs["features"], axis=3) # {N, task, seqlen}
    indices = tf.cast(inputs["variant_relative_pos"], tf.int64)
    outputs = dict(inputs)
    
    # set up a mask
    mask = tf.one_hot(
        indices,
        depth=features.get_shape().as_list()[2],
        on_value=1.0,
        off_value=0.0,
        dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=1) # {N, task, seqlen}

    # filter with mask and get score
    masked_features = tf.multiply(mask, features)
    variant_importances = tf.reduce_sum(masked_features, axis=2) # {N, task}
    outputs["variant_importances"] = variant_importances
    
    return outputs, params


def _blank_sequence(tensor_dict):
    """to be used with map_fn
    """
    assert tensor_dict.get("features") is not None
    assert tensor_dict.get("variant_relative_pos") is not None

    # get features, params
    features = tensor_dict["features"] # {task, seqlen, 4}
    variant_pos = tensor_dict["variant_relative_pos"] # {}
    extend_len = tensor_dict.get("extend_len", 10) # {}
    block_nonvariant = tensor_dict.get("inverse", False)
    
    # set up start stop
    total_length = features.get_shape().as_list()[1]
    indices = tf.range(total_length)
    start = tf.cast(tf.maximum(variant_pos - extend_len, 0), tf.int32)
    stop = tf.cast(tf.minimum(variant_pos + extend_len, total_length), tf.int32)
    
    # set up mask
    mask = tf.zeros(total_length)

    if block_nonvariant:
        mask = tf.add(mask, tf.cast(tf.less(indices, start), tf.float32))
        mask = tf.add(mask, tf.cast(tf.greater(indices, stop), tf.float32))
    else:
        mask = tf.add(mask, tf.cast(tf.greater(indices, start), tf.float32))
        mask = tf.multiply(mask, tf.cast(tf.less(indices, stop), tf.float32))
    mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
        
    # multiply features
    features = tf.multiply(features, mask)
    
    return features


def blank_variant_sequence(inputs, params):
    """blank any sequence that is not variant (to scan site for motifs etc)
    """
    assert inputs.get("variant_relative_pos") is not None

    # set up inputs to map fn
    map_features = {
        "features": inputs["features"],
        "variant_relative_pos": inputs["variant_relative_pos"]}
    outputs = dict(inputs)

    batch_size = inputs["features"].get_shape().as_list()[0]
    # map fn to blank everywhere else (get response at site)
    print map_features
    
    
    variant_sequence = tf.map_fn(
        _blank_sequence,
        map_features,
        dtype=tf.float32) # {N, task, seqlen, 4}
    variant_sequence = tf.reshape(
        variant_sequence,
        [-1, 2]+variant_sequence.get_shape().as_list()[1:]) # {N/2, 2, task, seqlen, 4}
    
    # map fn to blank the site (get responders)
    map_features = {
        "features": inputs["features"],
        "variant_relative_pos": inputs["variant_relative_pos"]}
    
    other_sequence = tf.map_fn(
        _blank_sequence,
        map_features,
        dtype=tf.float32) # {N, task, seqlen, 4}
    other_sequence = tf.reshape(
        other_sequence,
        [-1, 2]+other_sequence.get_shape().as_list()[1:]) # {N/2, 2, task, seqlen, 4}

    tmp_features = tf.stack([variant_sequence, other_sequence], axis=1) # {N/2, 2, 2, task, seqlen, 4}
    tmp_features = tf.reshape(
        tmp_features,
        [-1]+inputs["features"].get_shape().as_list()[1:]) # {N*2, task, seqlen, 4}

    # pad the outputs
    outputs["features"] = tmp_features
    outputs, _ = pad_examples(outputs, {"ignore": ["features"], "batch_size": batch_size})
    
    return outputs, params



def reduce_alleles(inputs, params):
    """given specific keys in the input, adjust them
    to reduce.
    """
    # given the inputs, adjust them for output
    # specifically need to adjust: variant_importances, features

    outputs = dict(inputs)
    print inputs.keys()
    print inputs

    # first adjust the motif scores and then unpad
    # TODO fix the reshape above!
    delta_motifs = tf.reshape(
        inputs["features"],
        [-1, 2]+inputs["features"].get_shape().as_list()[1:]) # {N, 2, task, motif} where 2 is in/out

    outputs["pwm-scores.allelic"] = delta_motifs

    batch_size = inputs["features"].get_shape().as_list()[0]
    outputs, _ = unpad_examples(outputs, {"ignore": ["pwm-scores.allelic"], "batch_size": batch_size})

    # then adjust the variant importance scores and the logits/probs
    allelic_keys = ["variant_importances", "logits", "probs"]
    for key in allelic_keys:
        outputs[key] = tf.reshape(
            outputs[key],
            [-1, 2]+outputs[key].get_shape().as_list()[1:])

    batch_size = outputs["features"].get_shape().as_list()[0]
    outputs, _ = unpad_examples(outputs, {"ignore": allelic_keys, "batch_size": batch_size})
    
    return outputs, params


# TODO maybe move this to mutate code
def get_position_of_delta_motifs(inputs, params):
    """given the position map of deltas (N, task, pos, M),
    give back the best position for each responding motif (for distance dependency)
    """
    

    return
