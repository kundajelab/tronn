"""Contains user defined operations

Functions here are non-gradient training updates 
that need to occur during training (such as torch7
style maxnorm)
"""

import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import ops


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


def threshold(input_tensor, th, val, name):
    """Sets up a threshold layer
    
    Use for torch7 models (like deepSEA) that use
    the threshold function instead of the ReLU function

    Args:
      input_tensor: input tensor
      th: threshold value
      val: the value to output if below threshold

    Returns:
      out_tensor: tensor after threshold
    """
    with tf.variable_scope(name) as scope:

        # First figure out where values are less than
        lessthan_tensor = tf.cast(tf.lesser(input_tensor, th), tf.int32)
        greaterthan_tensor = tf.cast(tf.greater(input_tensor, th), tf.int32)

        # Then make thresholded values equal to val and the rest are the same
        out_tensor = input_tensor * greaterthan_tensor + val * lessthan_tensor

    return out_tensor
    

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


def restore_variables_op(checkpoint, skip=[], scope_change=None):
    """Builds a function that can be run to restore from a checkpoint
    """
    variables_to_restore = slim.get_model_variables()
    variables_to_restore.append(tf.train.get_or_create_global_step())
    if None in variables_to_restore:
        variables_to_restore.remove(None)
    
    # remove variables as needed
    for skip_string in skip:
        variables_to_restore_tmp = [var for var in variables_to_restore
                                    if (skip_string not in var.name)]
        variables_to_restore = variables_to_restore_tmp

    logging.info(str(variables_to_restore))

    # TODO adjust variable names as needed (if ensembling, etc etc)
    scope_change = ["", "basset/"]
    #scope_change = None
    
    if scope_change is not None:
        start_scope, end_scope = scope_change
        checkpoint_name_to_var = {}
        for v in variables_to_restore:
            checkpoint_var_name = "{}".format(
                v.name.split(end_scope)[-1].split(":")[0])
            checkpoint_name_to_var[checkpoint_var_name] = v
        variables_to_restore = checkpoint_name_to_var

    # tool for debug as needed
    #from tensorflow.python.tools import inspect_checkpoint as chkp
    #chkp.print_tensors_in_checkpoint_file(checkpoint, tensor_name='', all_tensors=True)
    
    # create assign op
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        checkpoint,
        variables_to_restore)

    return init_assign_op, init_feed_dict


def class_weighted_loss_fn(loss_fn, labels, logits, pos_weights):
    """Use task pos example imbalances to reweight the tasks
    """
    logging.info("NOTE: using weighted loss!")
    task_losses = []
    for task_num in range(labels.get_shape()[1]):
        task_losses.append(
            loss_fn(labels[:,task_num],
                    logits[:,task_num],
                    pos_weights[task_num]))
    task_loss_tensor = tf.stack(task_losses, axis=1)
    loss = tf.reduce_sum(task_loss_tensor)
    # to later get total_loss
    tf.add_to_collection(ops.GraphKeys.LOSSES, loss)

    return loss


def positives_focused_loss_fn_old(loss_fn, labels, logits, task_weights, class_weights):
    """Reweight both tasks and classes such that a positive is basically
    equal weight across all tasks

    To do so, for each task use weighted cross entropy
    Then, weight the tasks when you sum them up

    """
    logging.info("NOTE: using positives focused loss!")
    task_losses = []
    for task_num in range(labels.get_shape()[1]):
        task_losses.append(
            tf.multiply(
                task_weights[task_num].astype("float32"),
                loss_fn(
                    labels[:,task_num],
                    logits[:,task_num],
                    class_weights[task_num])))
    task_loss_tensor = tf.stack(task_losses, axis=1)
    loss = tf.reduce_sum(task_loss_tensor)
    # later put this in training op using get_total_loss
    tf.add_to_collection(ops.GraphKeys.LOSSES, loss)

    return loss

def positives_focused_loss_fn(data_files, loss_fn, labels, logits):
    """Reweight both tasks and classes such that a positive is basically
    equal weight across all tasks

    To do so, for each task use weighted cross entropy
    Then, weight the tasks when you sum them up

    """
    logging.info("NOTE: using positives focused loss!")
    task_weights, class_weights = get_task_and_class_weights(data_files)
    task_losses = []
    for task_num in range(labels.get_shape()[1]):
        task_losses.append(
            loss_fn(
                labels[:,task_num],
                logits[:,task_num],
                class_weights[task_num]))
    task_loss_tensor = tf.stack(task_losses, axis=1)
    
    task_weights_list = []
    for i in range(labels.get_shape()[0]):
        task_weights_list.append(task_weights)
    task_weights_tensor = tf.stack(task_weights_list, axis=0)
    
    loss = tf.losses.compute_weighted_loss(
        task_loss_tensor,
        weights=task_weights_tensor,
        scope="loss")

    return loss

