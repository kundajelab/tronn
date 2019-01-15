"""description: operations that do QC on the fly (to tensors to get summary stats)
"""

import tensorflow as tf

from tronn.util.utils import DataKeys


def get_multimodel_score_relationships(inputs, params):
    """given the consensus after using all, get correlation compared to final clean results
    """
    multimodel_importances = inputs[DataKeys.WEIGHTED_SEQ_MULTIMODEL]
    importances = tf.reduce_sum(inputs[DataKeys.WEIGHTED_SEQ_ACTIVE], axis=-1) # {N, task, seqlen}
    num_models = multimodel_importances.get_shape().as_list()[1]
    outputs = dict(inputs)

    if False:
        # cosine similarity
        # problem - this ignores the zero values so artificially inflates correspondence
        tiled_consensus_importances = []
        for model_i in range(num_models):
            tiled_consensus_importances.append(importances)
        consensus_importances = tf.stack(tiled_consensus_importances, axis=1) # {N, model, task, seqlen}

        # normalize
        multimodel_importances_norm = tf.nn.l2_normalize(multimodel_importances, axis=(3))
        consensus_importances_norm = tf.nn.l2_normalize(consensus_importances, axis=(3))

        # get similarity
        similarities = tf.reduce_sum(
            tf.multiply(multimodel_importances_norm, consensus_importances_norm),
            axis=(3))
    else:
        # try jaccard, split up neg and pos
        importances = tf.expand_dims(importances, axis=1)

        # positive side
        min_pos_vals = tf.minimum(tf.nn.relu(importances), tf.nn.relu(multimodel_importances))
        max_pos_vals = tf.maximum(tf.nn.relu(importances), tf.nn.relu(multimodel_importances))

        # negative side
        min_neg_vals = tf.minimum(tf.nn.relu(-importances), tf.nn.relu(-multimodel_importances))
        max_neg_vals = tf.maximum(tf.nn.relu(-importances), tf.nn.relu(-multimodel_importances))

        # sum all min vals
        min_val_sum = tf.add(
            tf.reduce_sum(min_pos_vals, axis=-1),
            tf.reduce_sum(min_neg_vals, axis=-1))

        # sum all max vals
        max_val_sum = tf.add(
            tf.reduce_sum(max_pos_vals, axis=-1),
            tf.reduce_sum(max_neg_vals, axis=-1))

        # and divide
        similarities = tf.divide(min_val_sum, max_val_sum)

    outputs["multimodel.importances.similarities"] = similarities

    return outputs, params
