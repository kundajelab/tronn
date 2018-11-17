"""description: layers that operate on onehot raw sequences,
as in input preprocessing
"""

import numpy as np
import tensorflow as tf

from tronn.nets.util_nets import rebatch

from tronn.util.utils import DataKeys


def _check_is_onehot_sequence(tensor):
    """assertion whether tensor is shaped like 
    onehot sequence
    """
    tensor_shape = tensor.get_shape().as_list()
    assert len(tensor_shape) == 4
    assert tensor_shape[3] == 4
    
    return None


def dinucleotide_shuffle(sequence):
    """shuffle input by dinucleotides
    """
    # get features and seq_len
    features = tf.squeeze(sequence, axis=0) # {seqlen, 4}
    seq_len = features.get_shape().as_list()[0]

    # set up indices (range) and shuffle
    positions = tf.range(seq_len, delta=2)
    shuffled_first_positions = tf.random_shuffle(positions)
    shuffled_second_positions = tf.add(shuffled_first_positions, [1])

    # gather the dinucleotides
    first_bps = tf.gather(features, shuffled_first_positions)
    second_bps = tf.gather(features, shuffled_second_positions)

    # interleave by concatenating on second axis, and then reshaping
    pairs = tf.concat([first_bps, second_bps], axis=1)
    features = tf.reshape(pairs, [seq_len, -1])
    
    # and readjust back
    shuffled_sequence = tf.expand_dims(features, axis=0)
    
    return shuffled_sequence


def _build_dinucleotide_shuffle_fn_NEW(num_shuffles):
    """internal build fn to give to map_fn
    """
    def shuffle_fn(sequence):
        def dinuc_fn(val):
            return dinucleotide_shuffle(sequence)
        sequences = tf.map_fn(
            dinuc_fn,
            tf.range(num_shuffles),
            dtype=tf.float32,
            parallel_iterations=1)
        return sequences

    return shuffle_fn

    
def _build_dinucleotide_shuffle_fn(num_shuffles):
    def shuffle_fn(sequence):
        sequences = []
        for shuffle_idx in xrange(num_shuffles):
            sequences.append(dinucleotide_shuffle(sequence))
        sequences = tf.stack(sequences, axis=0)
        return sequences
    
    return shuffle_fn


def generate_dinucleotide_shuffles(inputs, params):
    """generate multiple shuffle sequences and rebatch
    """
    # assertions
    assert inputs.get(DataKeys.FEATURES) is not None
    _check_is_onehot_sequence(inputs[DataKeys.FEATURES])

    # data
    features = inputs[DataKeys.FEATURES]
    features_shape = features.get_shape().as_list()
    outputs = dict(inputs)
    
    # params
    aux_key = params.get("aux_key")
    num_shuffles = params.get("num_shuffles", 7)
    batch_size = features_shape[0]
    assert batch_size % (num_shuffles + 1) == 0

    # build the dinuc shuffle fn
    shuffle_fn = _build_dinucleotide_shuffle_fn(num_shuffles)

    # call map_fn
    dinuc_shuffles = tf.map_fn(
        shuffle_fn,
        features,
        back_prop=False,
        parallel_iterations=4) # TODO see if this is important
    dinuc_shuffles = tf.reshape(
        dinuc_shuffles,
        [batch_size, -1]+features_shape[1:])
    
    # save this to an auxiliary tensor and save out number used
    outputs[aux_key] = dinuc_shuffles # {N, aux, 1, 1000, 4}
    params["num_aux_examples"] = num_shuffles

    return outputs, params


# TODO clean this up
def generate_scaled_inputs(inputs, params):
    """generate scaled inputs (mostly for integrated gradients)
    """
    # assertions
    assert inputs.get(DataKeys.FEATURES) is not None
    _check_is_onehot_sequence(inputs[DataKeys.FEATURES])

    # features
    features = inputs[DataKeys.FEATURES]
    outputs = dict(inputs)
    
    # params
    batch_size = features.get_shape().as_list()[0]
    steps = params.get("num_scaled_inputs", 8)
    assert batch_size % num_scaled_inputs == 0
    
    # separate out features, scale, and append all together
    features = [tf.expand_dims(tensor, axis=0)
                for tensor in tf.unstack(features, axis=0)]
    transformed_features = []
    for example_idx in xrange(batch_size):
        scaled_features = tf.concat(
            [(float(i)/steps) * features[example_idx]
             for i in xrange(1, steps+1)],
            axis=0)
        transformed_features.append(scaled_features)
        
    outputs[DataKeys.FEATURES] = tf.concat(transformed_features, axis=0)
    
    # do the same for all others in inputs
    outputs, _ = pad_data(outputs, {"ignore": [DataKeys.FEATURES]})

    # rebatch
    outputs, _ = rebatch(data, {"name": "rebatch_scaled_inputs"})
    
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

# TODO keep to convert onehot mutants to string
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
    features = tf.reduce_join(string_array, axis=1, keep_dims=True)
    key = params.get("string_key", "features.string")
    outputs[key] = features
    
    return outputs, params


def decode_onehot_sequence(inputs, params):
    """go from onehot back to basepairs
    """
    assert params.get("decode_key") is not None
    assert inputs.get(params["decode_key"]) is not None

    # convert back to indices, making sure to adjust for N (put as index 4)
    features = inputs[params["decode_key"]] # {N, 1, seqlen, 4}
    features_n = tf.cast(tf.not_equal(tf.reduce_sum(features, axis=-1), 0), tf.int64) # {N, 1, seqlen}
    features = tf.argmax(features, axis=-1) # {N, 1, seqlen}
    features = tf.multiply(
        tf.add(features, 1),
        features_n) # now ranges 0-4, where N is 0
    #features = tf.mod(tf.subtract(features, 1), 5)
    #features = tf.as_string(features) # {N, seqlen}
    features = tf.reduce_join(features, axis=-1) # {N, 1}

    # and now replace all indices
    features = tf.regex_replace(features, "0", "N")
    features = tf.regex_replace(features, "1", "A")
    features = tf.regex_replace(features, "2", "C")
    features = tf.regex_replace(features, "3", "G")
    features = tf.regex_replace(features, "4", "T")
    
    # save out
    new_key = "{}.string".format(params["decode_key"])
    inputs[new_key] = features
    
    return inputs, params





# TODO move this?
def get_variance_importance(inputs, params):
    """read off the importance score at the specific variant position? is this important to read off?
    """
    assert inputs.get("features") is not None
    assert inputs.get("snp_relative_pos") is not None
    
    outputs = dict(inputs)

    # need to do it a task level
    importance_at_all_positions = tf.reduce_sum(inputs["features"], axis=3) # {N, task, seq_len}
    importance_at_snp_pos = importance_at_snp_pos[:,:,inputs["snp_relative_pos"]] # {N, task, 1}

    # save to outputs
    outputs["snp_importance_scores"] = importance_at_snp_pos
    
    return None
