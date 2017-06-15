"""Contains user defined operations

Functions here are non-gradient training updates 
that need to occur during training (such as torch7
style maxnorm)

"""

import tensorflow as tf


def maxnorm(norm_val=7):
    """Torch7 style maxnorm. After gradients are applied in a training step,
    the weights of each layer are clipped. This maxnorm applies to 
    convolutional layers as well as fully connected layers and does 
    not apply to biases.

    Args:
      norm_val: the clipping value

    Returns:
      None
    """
    weights = [v for v in tf.global_variables()
               if ('weights' in v.name)]
    for weight in weights:
        op_name = '{}/maxnorm'.format(weight.name.split('/weights')[0])
        maxnorm_update = weight.assign(
            tf.clip_by_norm(weight, norm_val, axes=[0], name=op_name))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             maxnorm_update)
        
    return None


def order_preserving_k_max(input_tensor, k):
    """Finds values k largest entries for the last dimension and 
    returns them in the order they originally appeared.

    If the input is a vector (rank-1), finds the k largest entries 
    in the vector and outputs their values and indices as vectors. 
    Thus values[j] is the j-th largest entry in input, and its 
    index is indices[j].

    For matrices (resp. higher rank input), computes the top k 
    entries in each row (resp. vector along the last dimension).
    
    return value shape: input.shape[:-1] + [k]
    
    Example:
    input: input=[1, 3, 2, 4], k=3
    output: [3,2,4]
    """
    ndims = input_tensor.shape.ndims
    
    # get indices of topk elements
    indices = tf.nn.top_k(input_tensor, k, sorted=False).indices#shape [d1,d2..,dn-1,k]
    # sort indices of topk elements
    indices = tf.nn.top_k(indices, k, sorted=True).values#shape [d1,d2..,dn-1,k]
    indices = tf.expand_dims(indices, axis=ndims)#shape [d1,d2..,dn-1,1,k]

    # build supporting indices for first n-1 dims
    support = tf.meshgrid(*[tf.range(tf.shape(input_tensor)[d])
                            for d in xrange(ndims-1)], indexing='ij')#see numpy.meshgrid
    support = tf.stack(support, axis=ndims-1)#shape [d1,d2..,dn-1,ndims-1]
    support = tf.expand_dims(support, axis=ndims-1)#shape [d1,d2..,dn-1,1,ndims-1]
    support = tf.tile(support, [1]*(ndims-1)+[k, 1])#shape [d1,d2..,dn-1,k,ndims-1]

    full_indices = tf.concat([support, indices], axis=ndims)#shape [d1,d2..,dn-1,k,ndims]
    output = tf.gather_nd(input_tensor, full_indices)
    
    return output
