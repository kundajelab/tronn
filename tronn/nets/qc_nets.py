"""description: operations that do QC on the fly (to tensors to get summary stats)
"""

import tensorflow as tf

from tronn.util.utils import DataKeys


def get_multimodel_score_relationships(inputs, params):
    """given the consensus after using all, get correlation compared to final clean results
    """
    multimodel_importances = tf.reduce_sum(inputs["importances.multimodel.tmp"], axis=-1) # {N, model, task, seqlen}
    importances = tf.reduce_sum(inputs[DataKeys.WEIGHTED_SEQ_ACTIVE], axis=-1) # {N, task, seqlen}
    outputs = dict(inputs)

    num_models = multimodel_importances.get_shape().as_list()[1]

    if False:
        # try cosine similarity?
        # problem - this ignores the zero values so artificially inflates correspondence
        tiled_consensus_importances = []
        for model_i in range(num_models):
            tiled_consensus_importances.append(importances)
        consensus_importances = tf.stack(tiled_consensus_importances, axis=1) # {N, model, task, seqlen}

        # normalize
        #multimodel_importances_norm = tf.nn.l2_normalize(multimodel_importances, axis=(0,1,2))
        #consensus_importances_norm = tf.nn.l2_normalize(consensus_importances, axis=(0,1,2))
        multimodel_importances_norm = tf.nn.l2_normalize(multimodel_importances, axis=(3))
        consensus_importances_norm = tf.nn.l2_normalize(consensus_importances, axis=(3))

        # get similarity
        cosine_similarities = tf.reduce_sum(
            tf.multiply(multimodel_importances_norm, consensus_importances_norm),
            axis=(3))
    else:
        # try jaccard, split up neg and pos
        importances = tf.expand_dims(importances, axis=1)

                
        min_vals = tf.nn.relu(tf.minimum(importances, multimodel_importances))
        #min_neg_vals = tf.nn.relu(tf.minimum(-importances, -multimodel_importances))
        #min_vals = tf.add(min_pos_vals, min_neg_vals)
        
        max_vals = tf.maximum(importances, multimodel_importances)
        #max_neg_vals = tf.maximum(-importances, -multimodel_importances)
        #max_vals = tf.add(max_pos_vals, max_neg_vals)

        similarities = tf.divide(
            tf.reduce_sum(min_vals, axis=-1),
            tf.reduce_sum(max_vals, axis=-1))
        

    outputs["multimodel.importances.similarities"] = similarities

    return outputs, params
