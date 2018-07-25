"""description: helpful util functions for tensor management
"""

import tensorflow as tf

from tronn.util.utils import DataKeys


def attach_auxiliary_tensors(inputs, params):
    """take auxiliary tensors and attach to main key
    this is useful for adjusted features like
    shuffles, scaled step inputs, mutated sequences
    """
    assert inputs.get(DataKeys.FEATURES) is not None
    assert params.get("aux_key") is not None
    assert inputs.get(params["aux_key"]) is not None
    
    main_features = inputs[DataKeys.FEATURES]
    main_features_shape = main_features.get_shape().as_list()
    aux_features = inputs[params["aux_key"]]
    num_aux_examples = aux_features.get_shape().as_list()[1]
    outputs = dict(inputs)
    
    # interleave in the aux features
    aux_features = tf.reshape(
        aux_features,
        [-1]+main_features_shape[1:])
    features = tf.stack([main_features, aux_features], axis=1)
    features = tf.reshape(
        features,
        [-1]+main_features_shape[1:])

    outputs[DataKeys.FEATURES] = features

    # and pad correctly
    params.update({"num_aux_features": num_aux_examples})
    params.update({"ignore_keys": [DataKeys.FEATURES]})
    outputs, _ = pad_inputs(outputs, params)

    # rebatch? depends
    
    return outputs, params


def _build_pad_fn(num_aux_examples):
    """internal build fn to build a padding fn 
    """
    def pad_fn(tensor):
        tensors = [tensor]
        for aux_idx in xrange(num_aux_examples):
            tensors.append(tensor)
        tensors = tf.stack(tensors, axis=0)
        return tensors

    return pad_fn


def pad_inputs(inputs, params):
    """when you attach the auxiliary tensors, you need to pad
    the others to maintain consistency across batches
    """
    assert params.get("num_aux_examples") is not None

    # params
    num_aux_examples = params["num_aux_examples"]
    ignore_keys = params["ignore_keys"]
    
    # build the pad fn
    pad_fn = _build_pad_fn(num_aux_examples)

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


def detach_auxiliary_tensors(inputs, params):
    """remove auxiliary tensors from the main key
    this is useful for pulling out features like
    shuffles, scaled step inputs, mutated sequences
    """
    assert inputs.get(DataKeys.FEATURES) is not None
    assert params.get("num_aux_examples") is not None
    assert params.get("rebatch_name") is not None

    # features
    features = inputs[DataKeys.FEATURES]
    
    # params
    num_aux_examples = params["num_aux_examples"]
    save_aux = params.get("save_aux", {}) # this should really come with orig key and final key tuple
    current_batch_size = features.get_shape().as_list()[0]
    new_batch_size = params.get("batch_size", current_batch_size)
    filter_name = params["rebatch_name"]
    
    # get indices
    indices = range(current_batch_size)
    main_indices = np.where(np.mod(indices, [num_aux_examples+1]) == 0)[0]
    aux_indices = np.where(np.mod(indices, [num_aux_examples+1]) != 0)[0]

    outputs = {}
    for key in inputs.keys():

        # gather examples
        outputs[key] = tf.gather(inputs[key], example_indices)

        # gather auxiliary examples if desired
        if key in save_aux.keys():
            aux_batch = tf.gather(inputs[key], aux_indices)
            aux_batch = tf.reshape(
                aux_batch,
                [current_batch_size, -1] + inputs[key].get_shape().as_list()[1:])
            outputs[save_aux[key]] = aux_batch

    # finally rebatch
    outputs, _ = rebatch(outputs, {"name": filter_name, "batch_size": new_batch_size})
            
    return outputs, params


def rebatch(inputs, params):
    """Re-batch after "breaking" a batch
    """
    # assertions
    assert params.get("name") is not None
    assert params.get("batch_size") is not None

    # params
    name = params["name"]
    batch_size = params["batch_size"]
    num_threads = params.get("num_queue_threads", 1)
    
    # set up the train batch
    with tf.variable_scope(name):
        outputs = tf.train.batch(
            inputs,
            batch_size,
            capacity=batch_size*3,
            num_threads=num_threads,
            enqueue_many=True,
            name="rebatch_queue")

    # delete name to make sure queues stay
    # in separate scopes
    del params["name"]

    return outputs, params
