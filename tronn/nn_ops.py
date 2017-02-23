"""Contains user defined operations

Functions here are non-gradient training updates 
that need to occur during training (such as torch7
style maxnorm)

"""


import tensorflow as tf


def maxnorm(norm_val=7):
    '''
    Torch7 style maxnorm. After gradients are applied in a training step,
    the weights of each layer are clipped. This maxnorm applies to 
    convolutional layers as well as fully connected layers and does 
    not apply to biases.
    '''

    weights = [v for v in tf.global_variables()
               if ('weights' in v.name)]

    for weight in weights:
        op_name = '{}/maxnorm'.format(weight.name.split('/weights')[0])
        maxnorm_update = weight.assign(
            tf.clip_by_norm(weight, norm_val, axes=[0], name=op_name))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             maxnorm_update)
        
    return None

def order_preserving_k_max(input, k):
    '''
    Finds values k largest entries for the last dimension and returns them in the order they originally appeared.
    If the input is a vector (rank-1), finds the k largest entries in the vector and outputs their values and indices as vectors. Thus values[j] is the j-th largest entry in input, and its index is indices[j].
    For matrices (resp. higher rank input), computes the top k entries in each row (resp. vector along the last dimension).
    return value shape: input.shape[:-1] + [k]
    Example:
    input: input=[1, 3, 2, 4], k=3
    output: [3,2,4]
    '''

    # input_shape = tf.shape(input)
    # output_shape = tf.concat([input_shape[:-1],[k]], 0)
    # dim = input_shape[-1]
    # input = tf.reshape(input, (-1, dim))
    # indices = tf.nn.top_k(input, k, sorted=False).indices
    # indices = tf.nn.top_k(indices, k, sorted=True).values
    # my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)
    # my_range_repeated = tf.tile(my_range, [1, k])
    # full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)], 2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    # output = tf.gather_nd(input, full_indices)
    # output = tf.reshape(output, output_shape)
    # return output

    input_shape = input.get_shape().as_list()
    output_shape = tf.concat([input_shape[:-1],[k]], 0)
    dim = input_shape[-1]
    input = tf.reshape(input, (-1, dim))
    indices = tf.nn.top_k(input, k, sorted=False).indices
    indices = tf.nn.top_k(indices, k, sorted=True).values
    my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)
    my_range_repeated = tf.tile(my_range, [1, k])
    full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)], 2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    output = tf.gather_nd(input, full_indices)
    output = tf.reshape(output, output_shape)
    return output