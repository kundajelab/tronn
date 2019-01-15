"""description: helpful util functions for tensor management
"""

import logging

import numpy as np
import tensorflow as tf

from tronn.util.utils import DataKeys


def build_stack(inputs, params, stack):
    """take a stack of layers (functional, not object oriented)
    and stack them
    """
    outputs = dict(inputs)
    for layer_fn, layer_params in stack:
        print layer_fn
        params.update(layer_params)
        outputs, out_params = layer_fn(outputs, params)
        params.update(out_params)
        print outputs[DataKeys.FEATURES].get_shape()
        
    return outputs, params


# define auxiliary tensors as tensors that are like the main features
# but different in some way - such as dinuc shuffles, mutations, etc
# want to have an extra dimension on the aux tensors in case there
# are multiple aux for 1 example

def transform_auxiliary_tensor(layer_fn, aux_key, inputs, params, aux_axis=1):
    """sometimes just want to adjust one tensor that is not main,
    wrap away the rest of the extra functional implementation
    """
    logging.info("LAYER: transforming aux tensors")
    # adjust aux tensors (assumes that there's an extra aux dimension)
    features = inputs[aux_key]
    perm = range(len(features.get_shape()))
    perm.insert(0, perm.pop(aux_axis))
    logging.debug("...permutation: {}".format(perm))
    unperm = np.argsort(perm)
    features = tf.transpose(features, perm=perm) # {aux, N,...}
    
    # set up map fn
    def map_layer_fn(tensors):
        result = layer_fn(
            {DataKeys.FEATURES: tensors},
            params)[0][DataKeys.FEATURES]
        return result

    # run
    features = tf.map_fn(map_layer_fn, features)
    
    # adjust the results to save back out
    features = tf.transpose(features, perm=unperm)

    logging.debug("RESULTS: {}".format(features.get_shape()))
    
    return features


def attach_auxiliary_tensors(inputs, params):
    """take auxiliary tensors and attach to main key
    this is useful for adjusted features like
    shuffles, scaled step inputs, mutated sequences
    """
    assert inputs.get(DataKeys.FEATURES) is not None
    assert params.get("aux_key") is not None
    assert inputs.get(params["aux_key"]) is not None
    logging.info("LAYER: attaching auxiliary tensors {}".format(params["aux_key"]))
    
    main_features = inputs[DataKeys.FEATURES]
    main_features_shape = main_features.get_shape().as_list()
    old_batch_size = main_features_shape[0]
    params["batch_size"] = old_batch_size
    
    aux_features = inputs[params["aux_key"]]
    num_aux_examples = aux_features.get_shape().as_list()[1]
    aux_axis = 1
    outputs = dict(inputs)
    batch_size = np.ceil(old_batch_size / float(num_aux_examples+1)) * (num_aux_examples+1)
    
    # interleave in the aux features
    main_features = tf.expand_dims(main_features, axis=aux_axis)
    features = tf.concat([main_features, aux_features], axis=1)
    features = tf.reshape(
        features,
        [-1]+main_features_shape[1:])
    
    outputs[DataKeys.FEATURES] = features
    
    # and pad correctly
    params.update({"num_aux_examples": num_aux_examples})
    params.update({"ignore_keys": [DataKeys.FEATURES]})
    outputs, _ = pad_inputs(outputs, params)
    del params["ignore_keys"]
    
    # rebatch? depends
    # NOTE this rebatch needs to be a factor of num_aux_features + 1
    if False:
        filter_name = params["name"]
        outputs, _ = rebatch(outputs, {"name": filter_name, "batch_size": batch_size})

    logging.debug("RESULT: {}".format(outputs[DataKeys.FEATURES].get_shape()))

    
    return outputs, params


def _build_pad_fn(num_aux_examples, axis=1):
    """internal build fn to build a padding fn 
    """
    def pad_fn(tensor):
        tensors = [tensor for i in xrange(num_aux_examples + 1)]
        tensors = tf.stack(tensors, axis=0)
        tensors = tf.reshape(
            tensors,
            [-1]+tensors.get_shape().as_list()[1:])
            
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
        if key in ignore_keys:
            outputs[key] = inputs[key]
            continue

        output = tf.map_fn(
            pad_fn,
            inputs[key])
        outputs[key] = tf.reshape(
            output,
            [-1]+inputs[key].get_shape().as_list()[1:])

    # backcheck work
    final_batch_size = inputs[params["ignore_keys"][0]].get_shape().as_list()[0]
    for key in outputs.keys():
        assert outputs[key].get_shape().as_list()[0] == final_batch_size, key

    return outputs, params


def detach_auxiliary_tensors(inputs, params):
    """remove auxiliary tensors from the main key
    this is useful for pulling out features like
    shuffles, scaled step inputs, mutated sequences
    """
    assert inputs.get(DataKeys.FEATURES) is not None
    assert params.get("num_aux_examples") is not None
    assert params.get("rebatch_name") is not None

    logging.info("LAYER: detach auxiliary tensors")
    
    # features
    features = inputs[DataKeys.FEATURES]
    
    # params
    num_aux_examples = params["num_aux_examples"]
    save_aux = params.get("save_aux", {})
    current_batch_size = features.get_shape().as_list()[0]
    final_batch_size = params.get("batch_size", current_batch_size)
    filter_name = params["rebatch_name"]
    
    # get indices
    indices = range(current_batch_size)
    main_indices = np.where(np.mod(indices, [num_aux_examples+1]) == 0)[0]
    aux_indices = np.where(np.mod(indices, [num_aux_examples+1]) != 0)[0]
    new_batch_size = len(main_indices)
    
    outputs = {}
    seen_keys = []
    for key in inputs.keys():
        if key in seen_keys:
            continue

        # gather examples
        outputs[key] = tf.gather(inputs[key], main_indices)

        # gather auxiliary examples if desired
        if key in save_aux.keys():
            aux_batch = tf.gather(inputs[key], aux_indices)
            aux_batch = tf.reshape(
                aux_batch,
                [new_batch_size, -1] + inputs[key].get_shape().as_list()[1:])
            outputs[save_aux[key]] = aux_batch
            seen_keys.append(save_aux[key])
            
    # finally rebatch
    if False:
        outputs, _ = rebatch(outputs, {"name": filter_name, "batch_size": final_batch_size})

    logging.debug("RESULTS: {}".format(outputs[DataKeys.FEATURES].get_shape()))
    
    return outputs, params


def clear_auxiliary_tensors(inputs, params):
    """clear certain tensors from tensor dict
    """ 

    return


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
            capacity=batch_size*2,
            num_threads=num_threads,
            enqueue_many=True,
            name="rebatch_queue")

    # delete name to make sure queues stay
    # in separate scopes
    del params["name"]

    return outputs, params


def rebatch_TEST(inputs, params):
    """Re-batch after "breaking" a batch
    """
    # assertions
    assert params.get("name") is not None
    assert params.get("batch_size") is not None

    # params
    name = params["name"]
    batch_size = params["batch_size"]

    # sorted params
    sorted_keys = sorted(inputs.keys())
    sorted_dtypes = [inputs[key].dtype for key in sorted_keys]
    sorted_shapes = [inputs[key].get_shape().as_list()[1:]
                    for key in sorted_keys]

    # dtypes 
    dtype_dict = {}
    for key in inputs.keys():
        dtype_dict[key] = inputs[key].dtype

    # put into dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # get 1 element at a time?
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(1))

    # iterator
    iterator = dataset.make_initializable_iterator()
    outputs = iterator.get_next()
    tf.get_collection("DATASETUP")
    tf.add_to_collection("DATASETUP", iterator.initializer)
    print tf.get_collection("DATASETUP")

    # get rid of first dim
    for key in outputs.keys():
        outputs[key] = tf.squeeze(outputs[key], axis=0)
    
    # staging area
    with tf.variable_scope(name):
        # set up a staging area
        area = tf.contrib.staging.StagingArea(
            dtypes=sorted_dtypes,
            names=sorted_keys,
            shapes=sorted_shapes) # shapes are crucial

        tf.get_collection("STAGING_OPS")
        tf.add_to_collection("STAGING_OPS", area.put(outputs))
        
        area.put(outputs)

        # make get_fn to get a batch back
        def get_fn(val):
            return area.get()
                
        # get results
        outputs = tf.map_fn(
            get_fn,
            tf.range(batch_size),
            dtype=dtype_dict)

    # delete name to make sure staging areas stay
    # in separate scopes
    del params["name"]
    
    return outputs, params
    
        

def fake2():
    #print inputs
    
    # dtypes 
    dtype_dict = {}
    for key in inputs.keys():
        dtype_dict[key] = inputs[key].dtype
    
    # trying dataset
    #with tf.device('/gpu:0'):
    if True:
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        #print dataset

        dataset = dataset.prefetch(batch_size*8)
        
        #dataset = dataset.batch(batch_size)
        dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(batch_size))
        #dataset = dataset.prefetch(batch_size)
        #outputs =  tf.contrib.data.get_single_element(dataset)
        #quit()

        # use this to block while waiting
        null_dataset = tf.data.Dataset.from_tensor_slices([1])
        dataset = null_dataset.apply(
            tf.contrib.data.parallel_interleave(
                lambda x: dataset,
                cycle_length=1,
                sloppy=True)) # sloppy=True crucial for blocking!

        print dataset
        
        if True:
            #print dataset
            
            #quit()
            iterator = dataset.make_initializable_iterator()
            #iterator = dataset.make_one_shot_iterator()
            outputs = iterator.get_next()
            tf.get_collection("DATASETUP")
            tf.add_to_collection("DATASETUP", iterator.initializer)
            print tf.get_collection("DATASETUP")
        
    # set batch size
    #for key in outputs.keys():
    #    outputs[key].set_shape(
    #        [batch_size] + outputs[key].get_shape().as_list()[1:])
    
    return outputs, params


def fake():
        
    with tf.variable_scope(name):
        # set up a staging area
        area = tf.contrib.staging.StagingArea(
            dtypes=sorted_dtypes,
            names=sorted_keys,
            shapes=sorted_shapes) # shapes are crucial

        for i in xrange(inputs[sorted_keys[0]].get_shape().as_list()[0]):
            print i

            def extract_and_put(i):
                # extract an example
                example = {}
                for key in inputs.keys():
                    example[key] = inputs[key][i]
                return area.put(example)
                    
            put_op = tf.cond(
                tf.equal(condition_mask[i], 1),
                lambda: extract_and_put(i),
                tf.no_op)
            tf.get_collection("STAGING_OPS")
            tf.add_to_collection("STAGING_OPS", put_op)

        quit()

        def split_fn(inputs):
            return dict(inputs)

        # use map fn to split and put
        inputs = tf.map_fn(
            split_fn,
            inputs)
        print inputs
        quit()
        
        
        # now place examples individually
        put_op = tf.where(
            tf.equal(condition_mask, 1),
            area.put(inputs),
            tf.no_op())
        tf.get_collection("STAGING_OPS")
        tf.add_to_collection("STAGING_OPS", put_op)

        quit()

        

        if False:
            # extract an example
            example = {}
            for key in inputs.keys():
                example[key] = inputs[key][i]

            # set up conditional and save out staging ops
            put_op = tf.cond(
                tf.equal(condition_mask[i], 1),
                lambda: area.put(example),
                tf.no_op)
            tf.get_collection("STAGING_OPS")
            tf.add_to_collection("STAGING_OPS", put_op)
            
        quit()
        # make put_fn to load individual examples to area
        def put_fn(inputs):
            put_op = area.put(inputs)
            tf.get_collection("STAGING_OPS")
            tf.add_to_collection("STAGING_OPS", put_op)
            return inputs
        
        # use map fn to split and put
        tf.map_fn(
            put_fn,
            inputs)

        # make get_fn to get a batch back
        def get_fn(val):
            return area.get()
                
        # get results
        outputs = tf.map_fn(
            get_fn,
            tf.range(batch_size),
            dtype=dtype_dict)

    # delete name to make sure staging areas stay
    # in separate scopes
    del params["name"]

    return outputs, params
