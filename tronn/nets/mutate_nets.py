"""contains code for creating batches of mutations and then analyzing them
"""

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.initializers import pwm_simple_initializer
from tronn.util.tf_utils import get_fan_in

from tronn.nets.sequence_nets import pad_examples
from tronn.nets.sequence_nets import unpad_examples

from tronn.nets.filter_nets import rebatch
from tronn.nets.filter_nets import filter_and_rebatch



def mutate_motif(inputs, params):
    """Find max motif positions and mutate
    """
    # assertions
    assert params.get("pos") is not None

    # features
    features = inputs["features"]
    outputs = dict(inputs)
    
    # params
    positions = params["pos"]
    filter_width = params.get("filter_width", 10) / 2

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
    outputs["features"] = tf.expand_dims(tf.expand_dims(features, axis=0), axis=0)
    
    return outputs, params



def blank_motif_sites(inputs, params):
    """given the shuffle positions, zero out this range
    """

    # features
    positions = inputs["mutation_ranges"]

    # params
    
    
    
    # something like tf assign

    

    
    return


#def repeat_config(config, repeat_num):
#    """Helper function to adjust config as needed
#    """
#    for key in config["outputs"].keys():
#        # adjust output
#        outputs = [tf.expand_dims(tensor, axis=0)
#                  for tensor in tf.unstack(config["outputs"][key], axis=0)]
 #       new_outputs = []
#        for output in outputs:
#            new_outputs += [output for i in xrange(repeat_num)]
#        config["outputs"][key] = tf.concat(new_outputs, axis=0)
#
#    return config


def generate_mutation_batch(inputs, params):
    """Given motif map (x position), mutate at strongest hit position
    and pass back a mutation batch, also track how big the batch is
    """
    # assertions
    assert params.get("positional-pwm-scores-key") is not None
    assert inputs.get(params["positional-pwm-scores-key"]) is not None
    assert params.get("raw-sequence-key") is not None
    assert inputs.get(params["raw-sequence-clipped-key"]) is not None
    assert params.get("grammars") is not None

    # features
    position_maps = inputs[params["positional-pwm-scores-key"]] # {N, task, pos, M}
    raw_sequence = inputs[params["raw-sequence-key"]] # {N, 1, seqlen, 4}
    outputs = dict(inputs)
    
    # params
    grammar_sets = params.get("grammars")
    pairwise_mutate = params.get("pairwise_mutate", False)
    start_shift = params.get("left_clip", 0) + int(params.get("filter_width", 0) / 2.)
    
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
    params["mutation_batch_size"] = mutation_batch_size
    print pwm_indices
    print mutation_batch_size

    # now do the following for each example
    features = [tf.expand_dims(tensor, axis=0)
                for tensor in tf.unstack(raw_sequence, axis=0)]
    #labels = [tf.expand_dims(tensor, axis=0)
    #            for tensor in tf.unstack(labels, axis=0)]
    
    features_w_mutated = []
    #labels_w_mutated = []
    for i in xrange(len(features)):
        features_w_mutated.append(features[i]) # first is always the original
        #labels_w_mutated.append(labels[i])
        # single mutation
        for pwm1_idx in pwm_indices[0]:
            mutate_params = {
                "pos": [
                    (pwm1_idx, motif_max_positions[i,pwm1_idx])]}
            #mutate_params.update(params)
            new_features = {"features": features[i]}
            mutated_outputs, _= mutate_motif(new_features, mutate_params)
            features_w_mutated.append(mutated_outputs["features"]) # append mutated sequences
            #labels_w_mutated.append(labels[i])
        if pairwise_mutate:
            # double mutated
            for pwm1_idx in pwm_indices[0]:
                for pwm2_idx in pwm_indices[0]:
                    if pwm1_idx < pwm2_idx:
                        mutate_params = {
                            "pos": [
                                (pwm1_idx, motif_max_positions[i,pwm1_idx]),
                                (pwm2_idx, motif_max_positions[i,pwm2_idx])]}
                        #config.update(new_config)
                        new_features = {"features": features[i]}
                        mutated_outputs, _= mutate_motif(features[i], mutate_params)
                        features_w_mutated.append(mutated_outputs["features"]) # append mutated sequences
                        #labels_w_mutated.append(labels[i])

    # use the pad examples function here
    outputs["features"] = tf.concat(features_w_mutated, axis=0)
    params["ignore"] = ["features"]
    outputs, params = pad_examples(outputs, params)

    # and rebatch
    params["name"] = "mutation_rebatch"
    params["batch_size_orig"] = params["batch_size"]
    params["batch_size"] = mutation_batch_size
    outputs, params = rebatch(outputs, params)
    
    return outputs, params


def run_model_on_mutation_batch(inputs, params):
    """Run the model on the mutation batch
    """
    # assertions
    assert params.get("model_fn") is not None

    # features
    outputs = dict(inputs)

    # set up model
    with tf.variable_scope("", reuse=True):
        model_fn = params["model"]
        model_outputs, params = model_fn(inputs, params) # {N_mutations, task}
        # save logits
        outputs["importance_logits"] = model_outputs["logits"]
        # TODO - note that here you can then also get the mutation effects out
        
    return outputs, params


def delta_logits(inputs, params):
    """Extract the delta logits and save out
    """
    # assertions
    assert params.get("importance_task_indices") is not None
    assert inputs.get("importance_logits") is not None

    # features
    logits = inputs["importance_logits"]
    outputs = dict(inputs)
    
    # params
    importance_task_indices = params["importance_task_indices"]
    logits_to_features = params.get("logits_to_features", True)
    
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
        outputs["features"] = logits_adj

        params["ignore"] = ["features"]
        outputs, params = pad_examples(outputs, params)

        params["name"] = "extract_delta_logits"
        outputs, params = rebatch(outputs, params)
        
    else:
        # duplicate the delta logits info to make equal to others
        outputs["delta_logits"] = tf.concat(
            [logits_adj for i in xrange(params["batch_size"])], axis=0)
        
    return outputs, params


def dfim(inputs, params):
    """Given motif scores on mutated sequences, get the differences and return
    currently assumes that 1 batch is 1 reference sequence, with the rest
    being mutations of that reference.
    """
    # assertions
    assert inputs.get("features") is not None

    # features
    features = inputs["features"]
    outputs = dict(inputs)
    
    # subtract from reference
    outputs["features"] = tf.subtract(
        features, tf.expand_dims(features[0], axis=0))
    
    return outputs, params


def motif_dfim(inputs, params):
    """Readjust the dimensions to get correct batch out
    """
    # assertions
    assert inputs.get("features") is not None

    # features
    features = inputs["features"]
    outputs = dict(inputs)
    
    # remove the original sequence and permute
    outputs["features"] = tf.expand_dims(
        tf.transpose(features[1:], perm=[1, 0, 2]),
        axis=0) # {1, task, M, M}

    # unpad
    params["num_scaled_inputs"] = params["batch_size"]
    params["ignore"] = ["features"]
    outputs, params = unpad_examples(outputs, params) 
    
    # rebatch
    params["name"] = "remove_mutations"
    params["batch_size"] = params["batch_size_orig"]
    outputs, params = rebatch(outputs, params)
    
    return outputs, params


def filter_mutation_directionality(inputs, params):
    """When you mutate the max motif, the motif score should
    drop (even with compensation from another hit) - make sure
    directionality is preserved
    """
    # assertions
    assert inputs.get("features") is not None
    assert params.get("grammars") is not None

    # features
    features = inputs["features"] # {N, task, mutation, motifset}
    outputs = dict(inputs)

    # params
    grammar_sets = params["grammars"]
    
    # set up pwm vector
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
    outputs["condition_mask"] = tf.greater(directionality_mask, [0])
    params["name"] = "motif_dfim_directionality"
    outputs, params = filter_and_rebatch(outputs, params)
        
    return outputs, params


