"""contains code for creating batches of mutations and then analyzing them
"""

import h5py

import numpy as np

import tensorflow as tf
#import tensorflow.contrib.slim as slim

#from tronn.util.initializers import pwm_simple_initializer
#from tronn.util.tf_utils import get_fan_in

from tronn.nets.sequence_nets import pad_data
from tronn.nets.sequence_nets import unpad_examples

from tronn.nets.filter_nets import rebatch
from tronn.nets.filter_nets import filter_and_rebatch


# Mutagenizer
# preprocess - generate mutation batch
# this requires finding the strongest motif hit
# also would like to track bp level gradients - which single bp change would most disrupt the motif/prediction output
# run model
# postprocess - feature extraction, get delta logits, dfim, motifscanner



def mutate_motif(inputs, params):
    """Find max motif positions and mutate
    """
    # assertions
    assert params.get("pos") is not None

    # features
    features = inputs["features"]
    # TODO grab the conditional here
    
    
    outputs = dict(inputs)
    
    # params
    positions = params["pos"]
    filter_width = params.get("filter_width", 10) / 2

    # use gather, with an index tensor that part of it gets shuffled
    features = features[0,0] # {1000, 4}
    
    for pwm_idx, position, motif_present in positions:
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
        mut_features = tf.gather(features, all_indices)

        # conditional on whether the motif actually had a positive score
        features = tf.cond(motif_present, lambda: mut_features, lambda: features)
        
    # readjust dims before returning
    outputs["features"] = tf.expand_dims(tf.expand_dims(features, axis=0), axis=0)
    
    return outputs, params


def blank_motif_site_old(sequence, pwm_indices, pwm_positions, filter_width):
    """given the position and the filter width, mutate sequence
    """
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
    sequence = tf.expand_dims(tf.expand_dims(features, axis=0), axis=0)
    
    return sequence


def blank_motif_sites(inputs, params):
    """given the shuffle positions, zero out this range
    NOTE: only built to block one site per example at the moment
    """
    # assertions
    assert inputs.get("pos") is not None
    
    # features
    features = tf.unstack(inputs["features"], axis=0)
    positions = tf.unstack(inputs["pos"], axis=0)
    outputs = dict(inputs)
    
    # params
    filter_width = params.get("filter_width", 10) #/2

    # use gather, with an index tensor that part of it gets shuffled
    #features = features[0,0] # {1000, 4}

    # make a tuple
    #  ( {N, 1, 1000, 4}, {N, M} )
    #blank_motif_sites_fn = build_motif_ablator() # here need to give it the params (pwm indices)

    # use map to parallel mutate
    #blanked_sequences = tf.map_fn(
    #    blank_motif_sites_fn,
    #    features,
    #    parallel_iterations=params["batch_size"])
    
    #for pwm_idx, position in positions:
    masked_features = []
    for example_idx in xrange(len(features)):

        example = features[example_idx]
        position = positions[example_idx]
        
        # set up start and end positions
        start_pos = tf.maximum(position - filter_width, 0)
        end_pos = tf.minimum(position + filter_width, example.get_shape()[1])
        
        # set up indices
        mask = tf.zeros(example.get_shape()[1])
        indices = tf.range(example.get_shape()[1], dtype=tf.int64)
        mask = tf.add(mask, tf.cast(tf.less(indices, start_pos), tf.float32))
        mask = tf.add(mask, tf.cast(tf.greater(indices, end_pos), tf.float32))
        mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=2)
        
        # multiply features by mask
        masked_features.append(tf.multiply(example, mask))

    # attach to outputs
    outputs["features"] = tf.stack(masked_features, axis=0)
    
    return outputs, params


def generate_mutation_batch(inputs, params):
    """Given motif map (x position), mutate at strongest hit position
    and pass back a mutation batch, also track how big the batch is
    """
    # assertions
    assert params.get("positional-pwm-scores-key") is not None
    assert inputs.get(params["positional-pwm-scores-key"]) is not None
    assert params.get("raw-sequence-key") is not None
    assert inputs.get(params["raw-sequence-clipped-key"]) is not None
    #assert params.get("grammars") is not None
    assert params.get("manifold") is not None

    # features
    position_maps = inputs[params["positional-pwm-scores-key"]] # {N, task, pos, M}
    raw_sequence = inputs[params["raw-sequence-key"]] # {N, 1, seqlen, 4}
    outputs = dict(inputs)
    
    # params
    #grammar_sets = params.get("grammars")
    manifold_h5_file = params["manifold"]
    pairwise_mutate = params.get("pairwise_mutate", False)
    start_shift = params.get("left_clip", 0) + int(params.get("filter_width", 0) / 2.)
    
    # do it at the global level, on the importance score positions
    # adjust for start position shifts
    global_position_map = tf.reduce_max(position_maps, axis=1) # {N, pos, M}
    motif_max_positions = tf.argmax(global_position_map, axis=1) # {N, M}
    motif_max_positions = tf.add(motif_max_positions, start_shift)
    outputs["pwm-max-positions"] = motif_max_positions # {N, M}
    print motif_max_positions

    # only generate a mutation if the score is positive (not negative)
    # build a conditional for it
    motif_max_vals = tf.reduce_max(global_position_map, axis=1) # {N, M}
    outputs["pwm-max-vals"] = motif_max_vals
    
    # TODO - only generate the mutation if there is a positive log likelihood hit

    # get indices at global level
    with h5py.File(manifold_h5_file, "r") as hf:
        pwm_vector = hf["master_pwm_vector"][:]
    #pwm_vector = np.zeros((grammar_sets[0][0].pwm_vector.shape[0]))
    #for i in xrange(len(grammar_sets[0])):
    #    pwm_vector += grammar_sets[0][i].pwm_thresholds > 0

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
    positions = []
    #labels_w_mutated = []
    for i in xrange(len(features)):
        features_w_mutated.append(features[i]) # first is always the original
        positions.append(0)
        #labels_w_mutated.append(labels[i])
        # single mutation
        for pwm1_idx in pwm_indices[0]:
            mutate_params = {
                "pos": [
                    (pwm1_idx,
                     motif_max_positions[i,pwm1_idx],
                     tf.greater(motif_max_vals[i,pwm1_idx], 0))]}
            #mutate_params.update(params)
            new_features = {"features": features[i]}
            mutated_outputs, _ = mutate_motif(new_features, mutate_params)
            features_w_mutated.append(mutated_outputs["features"]) # append mutated sequences
            positions.append(motif_max_positions[i,pwm1_idx])
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
    outputs["pos"] = tf.stack(positions, axis=0)
    params["ignore"] = ["features", "pos"]
    outputs, params = pad_data(outputs, params)
    quit()
    
    # and delete the used features
    del outputs[params["positional-pwm-scores-key"]]
    
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
        model_fn = params["model_fn"]
        model_outputs, params = model_fn(inputs, params) # {N_mutations, task}
        # save logits
        outputs["importance_logits"] = model_outputs["logits"]
        outputs["mut_probs"] = tf.nn.sigmoid(model_outputs["logits"])
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
    #importance_task_indices = params["importance_task_indices"]
    logits_to_features = params.get("logits_to_features", True)
    
    # just get the importance related logits
    #logits = [tf.expand_dims(tensor, axis=1)
    #          for tensor in tf.unstack(logits, axis=1)]
    #importance_logits = []
    #for task_idx in importance_task_indices:
    #    importance_logits.append(logits[task_idx])
    #logits = tf.concat(importance_logits, axis=1)

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
        outputs, params = pad_data(outputs, params)
        quit()

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

    # keep the sequence strings
    outputs["mut_features.string"] = tf.expand_dims(
        outputs["mut_features.string"], axis=0) # {1, M}

    # unpad
    params["num_scaled_inputs"] = params["batch_size"]
    params["ignore"] = ["features", "mut_features.string"]
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
    assert params.get("manifold") is not None
    #assert params.get("grammars") is not None

    # features
    features = inputs["features"] # {N, task, mutation, motifset}
    outputs = dict(inputs)

    # params
    #grammar_sets = params["grammars"]
    manifold_h5_file = params["manifold"]
    with h5py.File(manifold_h5_file, "r") as hf:
        pwm_vector = hf["master_pwm_vector"][:]
    
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


