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

    weights = [v for v in tf.all_variables()
               if ('weights' in v.name)]

    for weight in weights:
        op_name = '{}/maxnorm'.format(weight.name.split('/weights')[0])
        maxnorm_update = weight.assign(
            tf.clip_by_norm(weight, norm_val, axes=[0], name=op_name))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             maxnorm_update)
        
    return None
