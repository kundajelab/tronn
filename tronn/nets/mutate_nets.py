"""contains code for ism nets
"""

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.initializers import pwm_simple_initializer
from tronn.util.tf_utils import get_fan_in


from tronn.nets.filter_nets import rebatch


def mutate_motif(features, labels, config, is_training=False):
    """Find max motif positions and mutate
    """
    positions = config.get("pos")
    filter_width = config.get("filter_width") / 2 # 14 ish
    filter_width = 5
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


def motif_ism(features, labels, config, is_training=False):
    """run motif level in silico mutagenesis (ISM) to extract
    dependencies

    Note that you should return all sequences WITH the grammar scores,
    and then only use the top scoring regions to merge the deltas
    """
    # get the motifs desired for the grammar, for each task.
    grammar_sets = config.get("grammars")
    position_maps = config["outputs"].get(config["keep_pwm_scores_full"]) # {N, task, pos, M}
    raw_sequence = config["outputs"].get("onehot_sequence") # {N, 1, 1000, 4}
    assert grammar_sets is not None
    assert position_maps is not None
    assert raw_sequence is not None

    print position_maps
    
    # do it at the global level
    motif_max_positions = tf.argmax(
        tf.reduce_max(position_maps, axis=1), axis=1) # {N, M}
    print motif_max_positions

    # get indices at global level
    pwm_vector = np.zeros((grammar_sets[0][0].pwm_vector.shape[0]))
    for i in xrange(len(grammar_sets[0])):
        pwm_vector += grammar_sets[0][i].pwm_vector
        
    pwm_indices = np.where(pwm_vector)
    total_pwms = pwm_indices[0].shape[0]
    mutation_batch_size = (total_pwms**2 - total_pwms) / 2 + total_pwms + 1 # +1 for original sequence
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
    old_batch_size = config.get("batch_size")
    with tf.variable_scope("ism_mutation_rebatch"): # for the rebatch queue
        new_config = {"batch_size": mutation_batch_size}
        config.update(new_config)
        features, labels, _ = rebatch(features, labels, config)
    
    # adjust batch size here to be whatever the total mutations are
    # push this batch through the model
    # {N_mutations, tasks}
    # note that model needs to SPECIFICALLY name layers, dont rely on auto naming
    # since it wont work with reuse.
    with tf.variable_scope("", reuse=True):
        model = config.get("model")
        logits = model(features, labels, config, is_training=False) # {N_mutations, task}

    # TODO need to adjust the logits for the ones we care about
    # ie, the timepoint tasks
    importance_task_indices = config.get("importance_task_indices")
    assert importance_task_indices is not None
    logits = [tf.expand_dims(tensor, axis=1)
              for tensor in tf.unstack(logits, axis=1)]
    importance_logits = []
    for task_idx in importance_task_indices:
        importance_logits.append(logits[task_idx])
    logits = tf.concat(importance_logits, axis=1)

    # set up delta from normal
    logits = tf.subtract(logits, tf.expand_dims(logits[0,:], axis=0))
    
    # get delta from normal and
    # reduce the outputs {1, task, N_mutations}
    features = tf.expand_dims(
        tf.transpose(logits, perm=[1, 0]),
        axis=0) # {1, task, mutations}

    # readjust the labels and config
    labels = tf.expand_dims(tf.unstack(labels, axis=0)[0], axis=0)
    for key in config["outputs"].keys():
        config["outputs"][key] = tf.expand_dims(
            tf.unstack(config["outputs"][key], axis=0)[0], axis=0)
    
    # put through filter to rebatch
    config.update({"batch_size": old_batch_size})
    with tf.variable_scope("ism_final_rebatch"): # for the rebatch queue
        features, labels, config = rebatch(features, labels, config)
    
    return features, labels, config










# OLD


def _mutate(feature_tensor, position_tensor, zero_out_ref=False):
    """Small module to perform 3 mutations from reference base pair
    Input goes from
    """
    mutated_examples = []
    # build the index position
    positions_tiled = tf.reshape(tf.stack([position_tensor for i in range(4)]), (4, 1))
    indices_list = [tf.cast(tf.zeros((4, 1)), 'int64'),
                    tf.cast(tf.zeros((4, 1)), 'int64'),
                    tf.reshape(tf.cast(positions_tiled, 'int64'), (4, 1)),
                    tf.reshape(tf.cast(tf.range(4), 'int64'), (4, 1))]
    indices_tensor = tf.squeeze(tf.stack(indices_list, axis=1))
    
    # go through all 4 positions, and ignore (zero out) the one that is reference
    for mut_idx in range(4):
        updates_array = np.ones((4))
        updates_array[mut_idx] = 0
        updates_tensor = tf.constant(updates_array)
        mutated_mask_inv = tf.cast(
            tf.scatter_nd(indices_tensor, updates_tensor, feature_tensor.get_shape()),
            'bool')
        mutated_mask = tf.cast(
            tf.logical_not(mutated_mask_inv),
            'float32')
        mutated_example = tf.squeeze(tf.multiply(feature_tensor, mutated_mask), axis=0)
        mutated_examples.append(mutated_example)

    mutated_batch = tf.stack(mutated_examples, axis=0)
    #print mutated_batch.get_shape()
    
    if zero_out_ref:
        # zero out the one that is reference
        
        slice_start_position = tf.stack([tf.constant(0),
                                         tf.constant(0),
                                         tf.cast(position_tensor, 'int32'),
                                         tf.constant(0)])

        #print slice_start_position.get_shape()
        #print feature_tensor.get_shape()
        mut_position_slice = tf.slice(feature_tensor,
                                      slice_start_position,
                                      [1, 1, 1, 4])

        # and flip
        mut_position_slice_bool = tf.cast(mut_position_slice, 'bool')
        mut_position_mask = tf.cast(tf.logical_not(mut_position_slice_bool), 'float32')
    
    
        #print mut_position_slice.get_shape()
        
        mask_per_bp = [tf.expand_dims(tf.squeeze(mut_position_mask), axis=1) for i in range(feature_tensor.get_shape()[2])]
        mask_per_bp_tensor = tf.stack(mask_per_bp, axis=2)
        #print mask_per_bp_tensor.get_shape()
        
        mask_list = [mask_per_bp_tensor for i in range(feature_tensor.get_shape()[3])]
        mask_batch = tf.stack(mask_list, axis=3)
    
        #print mask_batch.get_shape()
        mutated_filtered_examples = tf.multiply(mutated_batch, mask_batch)
        #print mutated_filtered_examples.get_shape()
    else:
        mutated_filtered_examples = mutated_batch
    
        
    return tf.unstack(mutated_filtered_examples)


def generate_point_mutant_batch(features, max_idx, num_mutations):
    """Given a position in a sequence (one hot encoded) generate mutants
    """
    half_num_mutations = num_mutations / 2
    sequence_batch_list = []    
    for mutation_idx in range(-half_num_mutations, half_num_mutations):
        offset_tensor = tf.cast(mutation_idx, 'int64')
        final_position_tensor = tf.add(max_idx, offset_tensor)
        nonnegative_mask = tf.squeeze(tf.greater(final_position_tensor,
                                                 tf.cast(tf.constant([0]), 'int64')))
        final_position_tensor_filt = tf.multiply(final_position_tensor, tf.cast(nonnegative_mask, 'int64'))
        sequence_batch_list = sequence_batch_list + _mutate(features, final_position_tensor_filt)
    # then stack
    return tf.stack(sequence_batch_list, axis=0)


def generate_paired_mutant_batch(features, max_idx1, max_idx2, num_mutations):
    """Given two positions, generate mutants
    """
    single_mutant_batch = generate_point_mutant_batch(features, max_idx1, num_mutations)
    single_mutants = tf.unstack(single_mutant_batch)
    paired_mutant_batch = []
    for single_mutant in single_mutants:
        single_mutant_extended = tf.expand_dims(single_mutant, axis=0)
        paired_mutant_batch = paired_mutant_batch + tf.unstack(
            generate_point_mutant_batch(single_mutant_extended, max_idx2, num_mutations))
    return tf.stack(paired_mutant_batch)
    

def ism_for_grammar_dependencies(
        features,
        labels,
        model_params,
        is_training=False, # not used but necessary for model pattern
        num_mutations=6, # TODO (fix all these)
        num_tasks=43,
        current_task=0):
    """Run a form of in silico mutagenesis to get dependencies between motifs
    Remember that the input to this should be onehot encoded sequence
    NOT importance scores, and only those for the subtask set you care about
    """
    model = model_params["trained_net"]
    pwm_a = model_params["pwm_a"]
    pwm_b = model_params["pwm_b"]
    
    # first layer - instantiate motifs and scan for best match in sequence
    max_size = max(pwm_a.weights.shape[1], pwm_b.weights.shape[1])
    conv1_filter_size = [1, max_size]
    with slim.arg_scope([slim.conv2d],
                        padding='VALID',
                        activation_fn=None,
                        weights_initializer=pwm_simple_initializer(
                            conv1_filter_size, [pwm_a, pwm_b], get_fan_in(features)),
                        biases_initializer=None,
                        scope='mutate'):
        net = slim.conv2d(features, 2, conv1_filter_size)

    # get max positions for each motif
    max_indices = tf.argmax(net, axis=2) # TODO need to adjust to midpoint
    max_idx1_tensor = tf.squeeze(tf.slice(max_indices, [0, 0, 0], [1, 1, 1]))
    max_idx2_tensor = tf.squeeze(tf.slice(max_indices, [0, 0, 1], [1, 1, 1]))

    # generate mutant sequences for motif 1
    motif1_mutants_batch = generate_point_mutant_batch(
        features, max_idx1_tensor, num_mutations)
    motif1_batch_size = motif1_mutants_batch.get_shape().as_list()[0]
    print "motif 1 total mutants:", motif1_mutants_batch.get_shape()

    # for motif 2: do the same as motif 1
    motif2_mutants_batch = generate_point_mutant_batch(
        features, max_idx2_tensor, num_mutations)
    motif2_batch_size = motif2_mutants_batch.get_shape().as_list()[0]
    print "motif 2 total mutants:", motif2_mutants_batch.get_shape()

    # for joint
    joint_mutants_batch = generate_paired_mutant_batch(
        features, max_idx1_tensor, max_idx2_tensor, num_mutations)
    joint_batch_size = joint_mutants_batch.get_shape().as_list()[0]
    print "joint motif total mutants:", joint_mutants_batch.get_shape()

    # stack original sequence, motif 1 mutations, motif 2 mutations
    all_mutants_batch = tf.stack(
        tf.unstack(features) +
        tf.unstack(motif1_mutants_batch) +
        tf.unstack(motif2_mutants_batch) +
        tf.unstack(joint_mutants_batch)
    )

    # check batch size
    print "full total batch size:", all_mutants_batch.get_shape()
    batch_size = all_mutants_batch.get_shape()[0]

    # use labels to set output size THEN need to select the correct output logit node
    multilabel = tf.stack([labels for i in range(num_tasks)], axis=1)
    labels_list = []
    for i in range(batch_size):
        labels_list = labels_list + tf.unstack(multilabel)
    labels_extended = tf.squeeze(tf.stack(labels_list))

    # pass through model
    logits_alltasks = model(all_mutants_batch,
                        labels_extended,
                        model_params,
                        is_training=False)

    # Now need to select the correct logit position
    logits = tf.slice(logits_alltasks, [0, current_task], [tf.cast(batch_size, 'int32'), 1])
    print logits.get_shape()
    
    # might want to do it on logits actually (see larger synergy score?)
    # like: logit(orig) / (logit(a)/2 + logit (b)/2)
    # this tells you how much the score is versus only having 1 of each
    # actually you'll get two scores back - synergy dependent on one vs other motif
    single_mutant_total = num_mutations*4
    if False:
        example_logits_list = tf.unstack(logits)
        logit_orig = example_logits_list[0]
            
        logits_mutant_motif1 = tf.stack(example_logits_list[1:motif1_batch_size])
        logit_mutant_motif1_min = tf.reduce_min(logits_mutant_motif1, axis=0)
        logits_mutant_motif2 = tf.stack(example_logits_list[(1 + single_mutant_total):])
        logit_mutant_motif2_min = tf.reduce_min(logits_mutant_motif2, axis=0)

        synergy_score = tf.divide(logit_orig, tf.divide(tf.add(logit_mutant_motif1_min, logit_mutant_motif2_min), 2))

    # =============================
    # probabilities
    probabilities = tf.nn.sigmoid(logits)

    example_probs_list = tf.unstack(probabilities)
    prob_orig = example_probs_list[0]
    probs_mutant_motif1 = tf.stack(example_probs_list[1:motif1_batch_size])
    prob_mutant_motif1_min = tf.reduce_min(probs_mutant_motif1, axis=0)
    
    probs_mutant_motif2 = tf.stack(example_probs_list[(1 + motif1_batch_size):(1 + motif1_batch_size + motif2_batch_size)])
    print probs_mutant_motif2.get_shape()
    prob_mutant_motif2_min = tf.reduce_min(probs_mutant_motif2, axis=0)
    
    probs_mutant_joint = tf.stack(example_probs_list[(1 + motif1_batch_size + motif2_batch_size):])
    print probs_mutant_joint.get_shape()
    prob_mutant_joint_min = tf.reduce_min(probs_mutant_joint, axis=0)
    
    synergy_score = tf.divide(prob_orig, tf.divide(tf.add(prob_mutant_motif1_min, prob_mutant_motif2_min), 2))

    synergy_score2 = tf.divide(prob_orig - prob_mutant_joint_min,
                               tf.add(tf.subtract(prob_mutant_motif1_min, prob_mutant_joint_min),
                                      tf.subtract(prob_mutant_motif2_min, prob_mutant_joint_min)))


    # TODO(dk) individual motif scores
    # this gives you the multiplier (coefficient) for each motif, relative to each other
    motif1_score = tf.divide(prob_mutant_motif2_min, prob_mutant_joint_min)
    motif2_score = tf.divide(prob_mutant_motif1_min, prob_mutant_joint_min)
    
    # debug tool
    #interesting_outputs = [synergy_score, prob_orig, logit_orig, logit_mutant_motif1_min, logit_mutant_motif2_min, logits, all_mutants_batch, max_indices]
    
    # output the ratio
    return [synergy_score2, motif1_score, motif2_score]
