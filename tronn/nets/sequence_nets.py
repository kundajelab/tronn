"""description: layers that operate on onehot raw sequences,
as in input preprocessing
"""

import tensorflow as tf

from tronn.nets.filter_nets import rebatch


def pad_examples(data, params):
    """pad when examples are offset from other data
    """
    ignore_keys = params.get("ignore", [])
    assert ignore_keys is not None
    assert params.get("batch_size") is not None
    
    # params
    batch_size = params.get("batch_size")
    
    # calculate the offsets
    full_size = data[ignore_keys[0]].get_shape().as_list()[0]
    offset_factor = full_size / float(batch_size)
    assert offset_factor.is_integer()
    offset_factor = int(offset_factor)

    # now pad all
    for key in data.keys():
        if key in ignore_keys:
            continue

        # now with the key, split, pad and concat
        transformed_features = []
        features = [tf.expand_dims(tensor, axis=0)
                    for tensor in tf.unstack(data[key], axis=0)]
        for example_idx in xrange(batch_size):
            transformed_features.append(
                tf.concat(
                    [features[example_idx]
                     for i in xrange(offset_factor)], axis=0))
        data[key] = tf.concat(transformed_features, axis=0)

    # backcheck work
    for key in data.keys():
        assert data[key].get_shape().as_list()[0] == full_size
        
    return data, params


def unpad_examples(inputs, params):
    """unpad the examples to remove the offset
    """
    assert params.get("ignore") is not None
    assert len(params["ignore"]) > 0

    # params
    batch_size = params["batch_size"]
    num_scaled_inputs = params["num_scaled_inputs"]
    ignore_keys = params["ignore"]

    # for all keys, grab the first one and keep
    keep_indices = tf.range(0, batch_size, num_scaled_inputs)
    outputs = {}
    for key in inputs.keys():
        if key in ignore_keys:
            outputs[key] = inputs[key]
            continue
        outputs[key] = tf.gather(inputs[key], keep_indices)

    return outputs, params


def dinucleotide_shuffle(data, params):
    """shuffle input by dinucleotides
    """
    features_key = params.get("features_key", "features")
    
    # assertions
    assert data.get(features_key) is not None
    assert len(data.get(features_key).get_shape()) == 4
    assert data.get(features_key).get_shape().as_list()[3] == 4

    # pull features: {1, 1, seq_len, 4}
    features = inputs.get(features_key)

    # adjust to {seq_len, 4}
    features = tf.squeeze(features)
    num_bp = features.get_shape().as_list()[1]

    # set up indices (range) and shuffle
    positions = tf.range(num_bp, delta=2)
    shuffled_first_positions = tf.random_shuffle(positions)
    shuffled_second_positions = tf.add(shuffled_first_positions, [1])

    # gather the dinucleotides
    first_bps = tf.gather(features, shuffled_first_positions)
    second_bps = tf.gather(features, shuffled_second_positions)

    # interleave by concatenating on second axis, and then reshaping
    pairs = tf.concat([first_bps, second_bps], axis=1)
    features = tf.reshape(pairs, [num_bp, -1])

    # and readjust back
    data[features_key] = tf.expand_dims(
        tf.expand_dims(features, axis=0), axis=0)
    
    return data, params


def generate_dinucleotide_shuffles(data, params):
    """generate multiple shuffle sequences and rebatch
    """
    features_key = params.get("features_key", "features")

    # assertions
    assert data.get(features_key) is not None
    assert len(data.get(features_key).get_shape()) == 4
    assert data.get(features_key).get_shape().as_list()[1] == 1
    assert data.get(features_key).get_shape().as_list()[3] == 4
    assert params.get("batch_size") is not None

    # params
    batch_size = params.get("batch_size")
    num_shuffles = params.get("num_shuffles", 7)
    assert batch_size % (num_shuffles + 1) == 0
    
    # separate out features, shuffle, and append all together
    features = data.get(features_key)
    features = [tf.expand_dims(tensor, axis=0)
                for tensor in tf.unstack(features, axis=0)]
    transformed_features = []
    for example_idx in xrange(batch_size):
        features_w_shuffles = [features[example_idx]]
        for shuffle_idx in xrange(num_shuffles):
            shuffled_features, _ = dinucleotide_shuffle(
                {"features": features[example_idx]}, {"feature_key": "features"})
            features_w_shuffles.append(shuffled_features)
        features_w_shuffles = tf.concat(features_w_shuffles, axis=0)
        transformed_features.append(features_w_shuffles)
    data[features_key] = tf.concat(transformed_features, axis=0)
    
    # do the same for all others in inputs
    data, _ = pad_examples(data, {"ignore": features_key})

    # rebatch
    data, params = rebatch(data, {"name": "rebatch_dinuc"})

    return data, params


def generate_scaled_inputs(data, params):
    """generate scaled inputs (mostly for integrated gradients)
    """
    features_key = params.get("features_key", "features")

    # assertions
    assert data.get(features_key) is not None
    assert len(data.get(features_key).get_shape()) == 4
    assert data.get(features_key).get_shape().as_list()[1] == 1
    assert data.get(features_key).get_shape().as_list()[3] == 4
    assert params.get("batch_size") is not None

    # params
    batch_size = params.get("batch_size")
    steps = params.get("num_scaled_inputs", 8)
    assert batch_size % num_scaled_inputs == 0
    
    # separate out features, scale, and append all together
    features = data.get(features_key)
    features = [tf.expand_dims(tensor, axis=0)
                for tensor in tf.unstack(features, axis=0)]
    transformed_features = []
    for example_idx in xrange(batch_size):
        scaled_features = tf.concat(
            [(float(i)/steps) * features[example_idx]
             for i in xrange(1, steps+1)],
            axis=0)
        transformed_features.append(scaled_features)
    data[features_key] = tf.concat(transformed_features, axis=0)
    
    # do the same for all others in inputs
    data, _ = pad_examples(data, {"ignore": features_key})

    # rebatch
    data, params = rebatch(data, {"name": "rebatch_dinuc"})
    
    return data, params



def _make_basepair_array(features, basepair_string="N"):
    """generate a tf.string array of basepairs, to be used with masks
    """
    # get shape
    num_examples = features.get_shape().as_list()[0]
    num_basepairs = features.get_shape().as_list()[1]

    # get basepair array
    basepair_array = tf.stack([tf.convert_to_tensor(basepair_string, dtype=tf.string)
                               for i in xrange(num_basepairs)])
    basepair_array = tf.stack([basepair_array for i in xrange(num_examples)], axis=0)
    
    return basepair_array


def onehot_to_string(inputs, params):
    """convert onehot back into a string
    """
    features = tf.squeeze(inputs["features"]) # {N, 1000, 4}
    outputs = dict(inputs)

    max_indices = tf.argmax(features, axis=2) # {N, 1000}
    is_n = tf.reduce_sum(features, axis=2) # {N, 1000}

    # set up the null string and basepair strings
    basepair_strings = ["N", "A", "C", "G", "T"]
    basepair_arrays = []
    for basepair_string in basepair_strings:
        basepair_arrays.append(_make_basepair_array(max_indices, basepair_string))

    # set up the basepairs
    string_array = basepair_arrays[0] # null
    for i in xrange(1, len(basepair_strings)):
        string_array = tf.where(
            tf.equal(max_indices, i-1),
            basepair_arrays[i],
            string_array)
        
    # make sure that null basepairs stay null
    string_array = tf.where(
        tf.equal(is_n, 0),
        basepair_arrays[0],
        string_array)

    # and reduce
    features = tf.reduce_join(string_array, axis=1)
    key = params.get("string_key", "features.string")
    outputs[key] = features
    
    return outputs, params
