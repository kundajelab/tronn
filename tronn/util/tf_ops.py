"""Contains user defined operations

Functions here are non-gradient training updates 
that need to occur during training (such as torch7
style maxnorm)

"""

import logging

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from tronn.datalayer import get_positive_weights_per_task


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


def restore_variables_op(checkpoint_dir, skip=[]):
    """Builds a function that can be run to restore from a checkpoint
    """
    # get the checkpoint file and variables
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    variables_to_restore = slim.get_model_variables()
    variables_to_restore.append(slim.get_global_step())

    # remove variables as needed
    for skip_string in skip:
        variables_to_restore_tmp = [var for var in variables_to_restore
                                    if (skip_string not in var.name)]
        variables_to_restore = variables_to_restore_tmp

    logging.info(str(variables_to_restore))

    # create assign op
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        checkpoint_path,
        variables_to_restore)

    return init_assign_op, init_feed_dict


def task_weighted_loss_fn(data_files, loss_fn, labels, logits):
    """Use task pos example imbalances to reweight the tasks
    """
    print logging.info("NOTE: using weighted loss!")
    pos_weights = get_positive_weights_per_task(data_files)
    task_losses = []
    for task_num in range(labels.get_shape()[1]):
        # somehow need to calculate task imbalance...
        task_losses.append(loss_fn(logits[:,task_num],
                                   labels[:,task_num],
                                   pos_weights[task_num],
                                   loss_collection=None))
        task_loss_tensor = tf.stack(task_losses, axis=1)
        loss = tf.reduce_sum(task_loss_tensor)
        # to later get total_loss
        tf.add_to_collection(ops.GraphKeys.LOSSES, loss)

    return loss
