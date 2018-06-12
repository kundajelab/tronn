
import tensorflow as tf


def get_variant_importance_scores(inputs, params):
    """
    """
    assert inputs.get("features") is not None

    features = inputs["features"]
    indices = inputs["variant_relative_pos"]
    
    outputs = dict(inputs)
    
    features = tf.gather(inputs, indices, axis=2) # {N, task, 4}
    outputs["variant_importances"] = tf.reduce_sum(features, axis=2) # {N, task}
    
    return outputs


def _blank_sequence(tensor_dict):
    """to be used with map_fn
    """
    assert tensor_dict.get("features") is not None
    assert tensor_dict.get("variant_relative_pos") is not None

    # get features, params
    features = tensor_dict["features"] # {task, seqlen, 4}
    variant_pos = tensor_dict["variant_relative_pos"] # {}
    extend_len = tensor_dict["extend_len"] # {}
    block_nonvariant = tensor_dict.get("inverse", False)
    
    # set up start stop
    total_length = features.get_shape()[1]
    indices = tf.range(total_length)
    start = tf.max(variant_pos - extend_len, 0)
    stop = tf.min(variant_pos + extend_len, total_length)

    # set up mask
    mask = tf.zeros(total_length)

    if block_nonvariant:
        mask = tf.add(mask, tf.cast(tf.less(indices, start), tf.float32))
        mask = tf.add(mask, tf.cast(tf.greater(indices, stop), tf.float32))
    else:
        mask = tf.add(mask, tf.cast(tf.greater(indices, start), tf.float32))
        mask = tf.multiply(mask, tf.cast(tf.less(indices, stop), tf.float32))

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
        "variant_relative_pos": inputs["variant_relative_pos"],
        "extend_len": params.get("snp_extend_len", 10),
        "inverse": params.get("inverse", False)}
    outputs = dict(inputs)

    # params
    output_key = params.get("blanked_name", "features.nonvariant-blanked")
    
    # map fn
    blanked_sequence = tf.map_fn(
        _blank_sequence,
        map_features,
        dtype=tf.float32)

    print blanked_sequence

    outputs[output_key] = blanked_sequence
    
    return outputs



def reduce_alleles(inputs, params):
    """given specific keys in the input, adjust them
    to reduce.
    """
    
    

    
    return


# TODO maybe move this to mutate code
def get_position_of_delta_motifs(inputs, params):
    """given the position map of deltas (N, task, pos, M),
    give back the best position for each responding motif (for distance dependency)
    """
    

    return
