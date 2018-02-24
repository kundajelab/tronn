# description: random utility nets

import tensorflow as tf



def remove_global_task(features, labels, config, is_training=False):
    """Remove the global task (last result in axis 1)
    """
    features = [tf.expand_dims(tensor, axis=1)
                for tensor in tf.unstack(features, axis=1)]

    features = tf.concat(
        [features[i] for i in xrange(len(features)-1)], axis=1)
    
    return features, labels, config
