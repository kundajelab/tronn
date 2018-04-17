"""contains code for creating batches of mutations and then analyzing them
"""

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.initializers import pwm_simple_initializer
from tronn.util.tf_utils import get_fan_in

from tronn.nets.filter_nets import rebatch
#from tronn.nets.filter_nets import filter_through_mask


def mutate_motif(features, labels, config, is_training=False):
    """Find max motif positions and mutate
    """
    positions = config.get("pos")
    filter_width = config.get("filter_width") / 2 # 14 ish
    filter_width = 5 # just mess up the very middle piece
    assert positions is not None
    assert filter_width is not None

    # use gather, with an index tensor that part of it gets shuffled
    features = features[0,0] # {1000, 4}
    
    for pwm_idx, position in positions:
        # set up start and end positions
        start_pos = tf.maximum(position - filter_width, 0)
        end_pos = tf.minimum(position + filter_width, features.get_shape()[0])
        
        # get the indices that are LEFT of the shuffle region
        # these do NOT get shuffled
        left_indices = tf.range(start_pos)

        # get the indices to shuffle
        indices = tf.range(start_pos, end_pos)
        shuffled_indices = tf.random_shuffle(indices)

        # get the indices that are RIGHT of the shuffle region
        # these do NOT get shuffled
        right_indices = tf.range(end_pos, features.get_shape()[0])

        # concat the indices, and set shape
        all_indices = tf.concat([left_indices, shuffled_indices, right_indices], axis=0)
        all_indices = tf.reshape(all_indices, [features.get_shape()[0]])
        
        # and gather
        features = tf.gather(features, all_indices)

    # readjust dims before returning
    features = tf.expand_dims(tf.expand_dims(features, axis=0), axis=0)
        
    return features, labels, config


def repeat_config(config, repeat_num):
    """Helper function to adjust config as needed
    """
    for key in config["outputs"].keys():
        # adjust output
        outputs = [tf.expand_dims(tensor, axis=0)
                  for tensor in tf.unstack(config["outputs"][key], axis=0)]
        new_outputs = []
        for output in outputs:
            new_outputs += [output for i in xrange(repeat_num)]
        config["outputs"][key] = tf.concat(new_outputs, axis=0)

    return config


def generate_mutation_batch(features, labels, config, is_training=False):
    """Given motif map (x position), mutate at strongest hit position
    and pass back a mutation batch, also track how big the batch is
    """
    # get the motifs desired for the grammar, for each task.
    grammar_sets = config.get("grammars")
    position_maps = config["outputs"].get(config["keep_pwm_scores_full"]) # {N, task, pos, M}
    raw_sequence = config["outputs"].get("onehot_sequence") # {N, 1, 1000, 4}
    pairwise_mutate = config.get("pairwise_mutate", False)
    start_shift = config.get("left_clip", 0) + int(config.get("filter_width", 0) / 2.)
    assert grammar_sets is not None
    assert position_maps is not None
    assert raw_sequence is not None
    
    # do it at the global level, on the importance score positions
    # adjust for start position shifts
    motif_max_positions = tf.argmax(
        tf.reduce_max(position_maps, axis=1), axis=1) # {N, M}
    motif_max_positions = tf.add(motif_max_positions, start_shift)
    
    print motif_max_positions

    # get indices at global level
    pwm_vector = np.zeros((grammar_sets[0][0].pwm_vector.shape[0]))
    for i in xrange(len(grammar_sets[0])):
        pwm_vector += grammar_sets[0][i].pwm_thresholds > 0

    # set up the mutation batch
    pwm_indices = np.where(pwm_vector > 0)
    total_pwms = pwm_indices[0].shape[0]
    if pairwise_mutate:
        mutation_batch_size = (total_pwms**2 - total_pwms) / 2 + total_pwms + 1 # +1 for original sequence
    else:
        mutation_batch_size = total_pwms + 1 # +1 for original sequence
    config["mutation_batch_size"] = mutation_batch_size
    print pwm_indices
    print mutation_batch_size

    # now do the following for each example
    features = [tf.expand_dims(tensor, axis=0)
                for tensor in tf.unstack(raw_sequence, axis=0)]
    labels = [tf.expand_dims(tensor, axis=0)
                for tensor in tf.unstack(labels, axis=0)]
    
    features_w_mutated = []
    labels_w_mutated = []
    for i in xrange(len(features)):
        features_w_mutated.append(features[i]) # first is always the original
        labels_w_mutated.append(labels[i])
        # single mutation
        for pwm1_idx in pwm_indices[0]:
            new_config = {
                "pos": [
                    (pwm1_idx, motif_max_positions[i,pwm1_idx])]}
            config.update(new_config)
            mutated_features, _, _ = mutate_motif(features[i], labels, config)
            features_w_mutated.append(mutated_features) # append mutated sequences
            labels_w_mutated.append(labels[i])
        if pairwise_mutate:
            # double mutated
            for pwm1_idx in pwm_indices[0]:
                for pwm2_idx in pwm_indices[0]:
                    if pwm1_idx < pwm2_idx:
                        new_config = {
                            "pos": [
                                (pwm1_idx, motif_max_positions[i,pwm1_idx]),
                                (pwm2_idx, motif_max_positions[i,pwm2_idx])]}
                        config.update(new_config)
                        mutated_features, _, _ = mutate_motif(features[i], labels, config)
                        features_w_mutated.append(mutated_features) # append mutated sequences
                        labels_w_mutated.append(labels[i])

    # concat all. this is very big, so rebatch
    features = tf.concat(features_w_mutated, axis=0) # {N, 1, 1000, 4}
    labels = tf.concat(labels_w_mutated, axis=0) 
    config = repeat_config(config, mutation_batch_size)

    # note - if rebatching, keep the queue size SMALL for speed
    config["old_batch_size"] = config.get("batch_size")
    with tf.variable_scope("ism_mutation_rebatch"): # for the rebatch queue
        new_config = {"batch_size": mutation_batch_size}
        config.update(new_config)
        features, labels, _ = rebatch(features, labels, config)

    return features, labels, config


def run_model_on_mutation_batch(features, labels, config, is_training=False):
    """Run the model on the mutation batch
    """
    with tf.variable_scope("", reuse=True):
        model = config.get("model")
        logits = model(features, labels, config, is_training=False) # {N_mutations, task}
        # save logits
        config["outputs"]["importance_logits"] = logits
        # TODO - note that here you can then also get the mutation effects out
        
    return features, labels, config


def delta_logits(features, labels, config, is_training=False):
    """Extract the delta logits and save out
    """
    importance_task_indices = config.get("importance_task_indices")
    logits = config["outputs"].get("importance_logits")
    logits_to_features = config.get("logits_to_features", True)
    assert importance_task_indices is not None
    assert logits is not None

    # just get the importance related logits
    logits = [tf.expand_dims(tensor, axis=1)
              for tensor in tf.unstack(logits, axis=1)]
    importance_logits = []
    for task_idx in importance_task_indices:
        importance_logits.append(logits[task_idx])
    logits = tf.concat(importance_logits, axis=1)

    # set up delta from normal
    logits = tf.subtract(logits[1:], tf.expand_dims(logits[0,:], axis=0))
    
    # get delta from normal and
    # reduce the outputs {1, task, N_mutations}
    # TODO adjust this here so there is the option of saving it to config or setting as features
    logits_adj = tf.expand_dims(
        tf.transpose(logits, perm=[1, 0]),
        axis=0) # {1, task, mutations}
    
    if logits_to_features:
        features = logits_adj

        # readjust the labels and config
        labels = tf.expand_dims(tf.unstack(labels, axis=0)[0], axis=0)
        for key in config["outputs"].keys():
            config["outputs"][key] = tf.expand_dims(
                tf.unstack(config["outputs"][key], axis=0)[0], axis=0)
    
        # put through filter to rebatch
        config.update({"batch_size": config["old_batch_size"]})
        with tf.variable_scope("ism_final_rebatch"): # for the rebatch queue
            features, labels, config = rebatch(features, labels, config)
        
    else:
        # duplicate the delta logits info to make equal to others
        config["outputs"]["delta_logits"] = tf.concat(
            [logits_adj for i in xrange(config["batch_size"])], axis=0)
        
    return features, labels, config


def dfim(features, labels, config, is_training=False):
    """Given motif scores on mutated sequences, get the differences and return
    """
    # subtract from reference
    features = tf.subtract(features, tf.expand_dims(features[0], axis=0))
    
    return features, labels, config


def motif_dfim(features, labels, config, is_training=False):
    """Readjust the dimensions to get correct batch out
    """
    # remove the original sequence and permute
    features = tf.expand_dims(
        tf.transpose(features[1:], perm=[1, 0, 2]),
        axis=0) # {1, task, M, M}

    # readjust the labels and config
    labels = tf.expand_dims(tf.unstack(labels, axis=0)[0], axis=0)
    for key in config["outputs"].keys():
        config["outputs"][key] = tf.expand_dims(
            tf.unstack(config["outputs"][key], axis=0)[0], axis=0)
    
    # put through filter to rebatch
    config.update({"batch_size": config["old_batch_size"]})
    with tf.variable_scope("ism_final_rebatch"): # for the rebatch queue
        features, labels, config = rebatch(features, labels, config)
    
    return features, labels, config


def filter_mutation_directionality(features, labels, config, is_training=False):
    """When you mutate the max motif, the motif score should
    drop (even with compensation from another hit) - make sure
    directionality is preserved
    """
    # features {N, task, mutation, motifset}
    grammar_sets = config.get("grammars")
    pwm_vector = np.zeros((grammar_sets[0][0].pwm_vector.shape[0]))
    for i in xrange(len(grammar_sets[0])):
        pwm_vector += grammar_sets[0][i].pwm_thresholds > 0
    pwm_indices = np.where(pwm_vector > 0)
    print pwm_indices

    # number of pwms should match the mutation axis
    assert pwm_indices[0].shape[0] == features.get_shape().as_list()[2]

    # set up mask
    directionality_mask = tf.ones((features.get_shape()[0]))

    # for each motif, filter for directionality
    # TODO check this, might be too rigorous, since needs to pass for ALL tasks
    for i in xrange(pwm_indices[0].shape[0]):
        per_task_motif_mask = tf.cast(
            tf.less_equal(features[:,:,i,pwm_indices[0][i]], [0]),
            tf.float32) # {N, task}
        global_motif_mask = tf.reduce_min(per_task_motif_mask, axis=1) # {N}
        directionality_mask = tf.multiply(
            directionality_mask,
            global_motif_mask)
        
    # then run the mask through the filter queue
    condition_mask = tf.greater(directionality_mask, [0])

    # filter
    with tf.variable_scope("mutation_directionality_filter"):
        features, labels, config = filter_through_mask(
            features, labels, config, condition_mask, num_threads=1)
        
    return features, labels, config


