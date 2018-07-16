"""description: layers that operate on onehot raw sequences,
as in input preprocessing
"""

import numpy as np
import tensorflow as tf

from tronn.nets.filter_nets import rebatch

from tronn.util.utils import DataKeys



def _check_is_onehot_sequence(tensor):
    """assertion whether tensor is shaped like 
    onehot sequence
    """
    tensor_shape = tensor.get_shape().as_list()
    assert len(tensor_shape) == 4
    assert tensor_shape[3] == 4
    
    return None


# TODO move this out of sequence nets to filter nets?
def _build_pad_fn(num_shuffles):
    """internal build fn to build a padding fn 
    """
    def pad_fn(tensor):
        tensors = [tensor]
        for shuffle_idx in xrange(num_shuffles):
            tensors.append(tensor)
        tensors = tf.stack(tensors, axis=0)
        return tensors

    return pad_fn


def pad_data(inputs, params):
    """pad examples when a data key was adjusted
    """
    # assertions
    assert params.get("num_shuffles") is not None
    
    # params
    num_shuffles = params["num_shuffles"]
    ignore = params["ignore"]
    
    # build the pad fn
    pad_fn = _build_pad_fn(num_shuffles)

    # run map fn on all inputs except ignore set
    outputs = {}
    for key in inputs.keys():
        if key in ignore:
            outputs[key] = inputs[key]
            continue

        output = tf.map_fn(
            pad_fn,
            inputs[key])
        outputs[key] = tf.reshape(
            output,
            [-1]+inputs[key].get_shape().as_list()[1:])

    # backcheck work
    final_batch_size = inputs[params["ignore"][0]].get_shape().as_list()[0]
    for key in outputs.keys():
        assert outputs[key].get_shape().as_list()[0] == final_batch_size
    
    return outputs, params


# TODO deprecate this
def unpad_examples(inputs, params):
    """unpad the examples to remove the offset
    """
    assert params.get("ignore") is not None
    assert len(params["ignore"]) > 0

    # params
    batch_size = params["batch_size"]
    ignore_keys = params["ignore"]

    # calculate the offsets
    reduced_size = inputs[ignore_keys[0]].get_shape().as_list()[0]
    num_scaled_inputs = batch_size / float(reduced_size)
    assert num_scaled_inputs.is_integer()
    num_scaled_inputs = int(num_scaled_inputs)
    
    # for all keys, grab the first one and keep
    keep_indices = tf.range(0, batch_size, num_scaled_inputs)
    outputs = {}
    for key in inputs.keys():
        if key in ignore_keys:
            outputs[key] = inputs[key]
            continue
        outputs[key] = tf.gather(inputs[key], keep_indices)

    return outputs, params


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


def _build_dinucleotide_shuffle_fn(num_shuffles):
    """internal build fn to give to map_fn
    """
    def shuffle_fn(sequence):
        sequences = [sequence]
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
    outputs = dict(inputs)
    
    # params
    num_shuffles = params.get("num_shuffles", 7)
    params["num_shuffles"] = num_shuffles
    batch_size = features.get_shape().as_list()[0]
    assert batch_size % (num_shuffles + 1) == 0

    # build the dinuc shuffle fn
    shuffle_fn = _build_dinucleotide_shuffle_fn(num_shuffles)

    # call map_fn
    features_w_shuffles = tf.map_fn(
        shuffle_fn,
        features)
    outputs[DataKeys.FEATURES] = tf.reshape(
        features_w_shuffles,
        [-1]+features.get_shape().as_list()[1:])

    # and then pad everything else
    outputs, _ = pad_data(
        outputs,
        {"num_shuffles": num_shuffles,
         "ignore": [DataKeys.FEATURES]})

    # rebatch
    outputs, _ = rebatch(outputs, {"name": "rebatch_dinuc", "batch_size": batch_size})

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
        
    outputs[DataKeys.FEATURESx] = tf.concat(transformed_features, axis=0)
    
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

# TODO keep for backwards compatibility, but generally push through string
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



def remove_shuffles(inputs, params):
    """calculate offsets, gather the indices, and separate out
    """
    assert params.get("num_shuffles") is not None

    # params
    num_shuffles = params["num_shuffles"]
    keep_shuffles = params.get("keep_shuffles", True)
    keep_shuffle_keys = params.get("keep_shuffle_keys", [])
    full_batch_size = inputs[inputs.keys()[0]].get_shape().as_list()[0]
    
    # get indices
    indices = range(full_batch_size)
    example_indices = np.where(np.mod(indices, [num_shuffles+1]) == 0)[0]
    shuffle_indices = np.where(np.mod(indices, [num_shuffles+1]) != 0)[0]
    batch_size = len(example_indices)
    
    # go through keys
    outputs = {}
    for key in inputs.keys():

        # gather examples and save out
        example_batch = tf.gather(inputs[key], example_indices)
        outputs[key] = example_batch

        # gather shuffles and keep if desired
        if key in keep_shuffle_keys:
            shuffle_batch = tf.gather(inputs[key], shuffle_indices)
            shuffle_batch = tf.reshape(
                shuffle_batch,
                [batch_size, -1] + inputs[key].get_shape().as_list()[1:])
            outputs["{}.{}".format(key, DataKeys.SHUFFLE_SUFFIX)] = shuffle_batch

    # rebatch back up
    outputs, _ = rebatch(outputs, {"name": "remove_shuffles_rebatch", "batch_size": full_batch_size})
    
    return outputs, params



def clear_shuffles(inputs, params):
    """clear out anything from shuffles
    """
    outputs = {}
    for key in inputs.keys():

        if DataKeys.SHUFFLE_SUFFIX in key:
            continue

        outputs[key] = inputs[key]

    return outputs, params
