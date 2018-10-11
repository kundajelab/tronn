"""contains code for creating batches of mutations and then analyzing them
"""

import h5py
import itertools

import numpy as np
import tensorflow as tf

from tronn.nets.util_nets import rebatch
from tronn.nets.filter_nets import filter_and_rebatch

from tronn.util.utils import DataKeys


class Mutagenizer(object):
    """basic in silico mutagenesis"""

    def __init__(self, mutation_type="point"):
        self.mutation_type = mutation_type

        
    def preprocess(self, inputs, params):
        # determine where to mutagenize
        # base level - search for motif
        # requires motif score thresh map {N, task, pos, M} and filter_width
        # base level, returns the pos of greatest M (top k)
        assert inputs.get(DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH) is not None
        assert params.get("manifold") is not None # {M}
        outputs = dict(inputs)
        
        # TODO should this be a separate function in motifs?
        # features - look at global top score positions
        features = inputs[DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH] # {N, task, pos, M}
        features = tf.reduce_max(features, axis=1) # {N, pos, M}
        features = tf.transpose(features, perm=[0,2,1]) # {N, M, pos}
        
        # params
        num_mutations = params.get("num_mutations", 1)
        
        # get the max positions ON THE ACTIVE REGION
        # NOTE that these are indices for the short region!!
        vals, indices = tf.nn.top_k(features, k=num_mutations, sorted=True)
        
        # and then only keep the ones that filter through pwm vector
        with h5py.File(params["manifold"], "r") as hf:
            pwm_mask = hf[DataKeys.MANIFOLD_PWM_SIG_CLUST_ALL][:]
        params["num_aux_examples"] = np.sum(pwm_mask > 0)
        sig_pwm_indices = np.where(pwm_mask > 0)[0]
        vals = tf.gather(vals, sig_pwm_indices, axis=1)
        indices = tf.gather(indices, sig_pwm_indices, axis=1)
        
        # save
        outputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL] = vals # {N, mut_M, k}
        outputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX] = indices # {N, mut_M, k}
        outputs[DataKeys.MUT_MOTIF_PRESENT] = tf.reduce_any(tf.greater(vals, 0), axis=2) # {N, mut_M}

        if self.mutation_type == "point":
            # set up gradients
            grad = inputs[DataKeys.IMPORTANCE_GRADIENTS]
            grad_mins = tf.reduce_min(grad, axis=1, keepdims=True) # {N, 1, 1000, 4}
            grad_mins = tf.argmin(grad_mins, axis=3) # {N, 1, 1000}
            grad_mins = tf.one_hot(grad_mins, 4) # {N, 1, 1000, 4}
            outputs["grad_mins"] = grad_mins
        
        return outputs, params

    
    @staticmethod
    def point_mutagenize_example(tensors):
        """generate a point mutant
        """
        # extract tensors
        example = tensors[0] # {1, 1000, 4}
        motif_present = tensors[1]
        pos = tensors[2] # {1, 1000, 1}
        grad_mins = tensors[3] # {tasks, 1000, 4}

        # adjust tensor
        example_mut = tf.where(positions, example, grad_mins) # {1, 1000, 4}

        # but only actually adjust if the motif actually there
        example = tf.cond(motif_present, lambda: example_mut, lambda: example)
        
        return example, pos


    @staticmethod
    def shuffle_mutagenize_example(tensors):
        """generate shuffle mutants
        """
        # extract tensors
        example = tensors[0] # {1, 1000, 4}
        motif_present = tensors[1]
        positions = tensors[2] # {1, 1000, 1}
        k = 10
        
        # adjustments
        example = tf.squeeze(example, axis=0) # {1000, 4}
        positions = tf.expand_dims(positions, axis=0) # {1, 1, 1000, 1}
        
        # use pooling to get a spread
        positions = tf.nn.max_pool(positions, [1,1,5,1], [1,1,1,1], padding="SAME") # {1, 1, 1000, 1}
        positions = tf.squeeze(positions) # {1000}
        num_positions = positions.get_shape().as_list()[0]

        # set up true indices, after splitting out positions to shuffle
        positions_to_shuffle = tf.where(tf.greater(positions, 0)) # {?}
        positions_to_keep = tf.where(tf.equal(positions, 0)) # {?}
        true_indices = tf.concat([positions_to_shuffle, positions_to_keep], axis=0) # {1000, 1}
        true_indices = tf.squeeze(true_indices, axis=1)
        true_indices.set_shape(positions.get_shape())

        # shuffle, and save out to shuffled positions
        positions_to_shuffle = tf.random_shuffle(positions_to_shuffle) # {?}
        shuffled_indices = tf.concat([positions_to_shuffle, positions_to_keep], axis=0) # {1000, 1}
        shuffled_indices = tf.squeeze(shuffled_indices, axis=1)
        shuffled_indices.set_shape(positions.get_shape())
        
        # now rank the TRUE indices to get the indices ordered back to normal
        vals, indices = tf.nn.top_k(true_indices, k=num_positions, sorted=True) # indices
        indices = tf.reverse(indices, [-1])
        
        # gather on the SHUFFLED to get back indices shuffled only where requested
        final_indices = tf.gather(shuffled_indices, indices)

        # gather from example
        example_mut = tf.gather(example, final_indices)
        
        # conditional on whether the motif actually there
        example = tf.cond(motif_present, lambda: example_mut, lambda: example)
        example = tf.expand_dims(example, axis=0)
        
        # adjust positions
        no_positions = tf.zeros(positions.get_shape())
        positions = tf.cond(motif_present, lambda: positions, lambda: no_positions)
        positions = tf.expand_dims(positions, axis=1)
        positions = tf.expand_dims(positions, axis=0) # {1, 1000, 1}
        
        return example, positions

    
    def mutagenize_multiple_motifs(self, inputs, params):
        """comes in from map fn?
        """
        assert inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX] is not None # {N, mut_M, k}
        outputs = dict(inputs)
        
        # for each motif, run map fn to mutagenize
        features = inputs[DataKeys.ORIG_SEQ] # {N, 1, 1000, 4}
        orig_seq_len = features.get_shape().as_list()[2]
        LEFT_CLIP = 420
        position_indices = tf.add(inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX], LEFT_CLIP) # {N, mut_M, k}
        inputs["test.position_indices"] = position_indices
        motif_present = inputs[DataKeys.MUT_MOTIF_PRESENT] # {N, mut_M}
        
        # choose what kind of mutagenesis
        if self.mutation_type == "point":
            mutagenize_fn = self.point_mutagenize_example
        else:
            mutagenize_fn = self.shuffle_mutagenize_example

        all_mut_sequences = []
        all_mut_positions = []
        for mut_i in xrange(position_indices.get_shape().as_list()[1]):
            
            mut_positions = position_indices[:,mut_i,:] # {N, k}
            mut_positions = tf.one_hot(mut_positions, orig_seq_len) # {N, k, 1000}
            mut_positions = tf.reduce_max(mut_positions, axis=1) # {N, 1000}
            mut_positions = tf.expand_dims(mut_positions, axis=2) # {N, 1000, 1}
            mut_positions = tf.expand_dims(mut_positions, axis=1) # {N, 1, 1000, 1}
            mut_motif_present = motif_present[:,mut_i] # {N}
            mut_sequences, mut_positions = tf.map_fn(
                mutagenize_fn,
                [features, mut_motif_present, mut_positions],
                dtype=(tf.float32, tf.float32)) # {N, 1, 1000, 4}
            all_mut_sequences.append(mut_sequences)
            all_mut_positions.append(mut_positions)
            
        outputs[DataKeys.MUT_MOTIF_ORIG_SEQ] = tf.stack(all_mut_sequences, axis=1) # {N, mut_M, 1, 1000, 4}
        outputs[DataKeys.MUT_MOTIF_POS] = tf.stack(all_mut_positions, axis=1) # {N, mutM, 1, 1000, 1}
        
        return outputs, params


    def mutagenize(self, inputs, params):
        """wrapper function to call everything
        """
        inputs, params = self.preprocess(inputs, params)
        outputs, params = self.mutagenize_multiple_motifs(inputs, params)
        outputs, params = self.postprocess(outputs, params)

        return outputs, params
    

    def postprocess(self, inputs, params):
        # depends on the analysis
        
        return inputs, params



class SynergyMutagenizer(Mutagenizer):
    """Generates sequences for synergy scores"""

    
    def mutagenize_multiple_motifs(self, inputs, params):
        """change this from the Mutagenizer class
        """
        assert inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX] is not None # {N, mut_M, k}
        outputs = dict(inputs)
        
        # for each motif, run map fn to mutagenize
        features = inputs[DataKeys.ORIG_SEQ] # {N, 1, 1000, 4}
        orig_seq_len = features.get_shape().as_list()[2]
        LEFT_CLIP = 420 # TODO fix this
        position_indices = tf.add(inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX], LEFT_CLIP) # {N, mut_M, k}
        inputs["test.position_indices"] = position_indices
        motif_present = inputs[DataKeys.MUT_MOTIF_PRESENT] # {N, mut_M}
        all_motifs_present = tf.reduce_all(motif_present, axis=1) # {N}
        
        # choose what kind of mutagenesis
        if self.mutation_type == "point":
            mutagenize_fn = self.point_mutagenize_example
        else:
            mutagenize_fn = self.shuffle_mutagenize_example

        all_mut_sequences = []
        all_mut_positions = []

        assert position_indices.get_shape().as_list()[2] == 1
        
        # now generate all sorts of mutation combinations (mutM, 2**mutM)
        num_mut_motifs = motif_present.get_shape().as_list()[1]
        mut_combinations = np.zeros((num_mut_motifs, 2**num_mut_motifs)) # {mutM, 2**mutM}
        mut_switch = [[0,1]]*num_mut_motifs
        i = 0
        for combo in itertools.product(*mut_switch):
            mut_combinations[:,i] = combo
            i += 1 
        mut_combinations = np.expand_dims(mut_combinations, axis=0) # {1, mutM, 2**mutM}
        
        # and then elementwise multiply
        position_combinations = tf.multiply(position_indices, mut_combinations) # {N, mutM, 2**mutM}

        # and flip
        position_combinations = tf.transpose(position_combinations, perm=[0,2,1]) # {N, 2**mutM, mutM}

        # set up a vector to zero out the 0th spot
        position_shape = position_combinations.get_shape().as_list()
        zero_positions = tf.zeros([position_shape[0], 1], dtype=tf.int32)
        zero_positions = tf.one_hot(zero_positions, orig_seq_len) # {N, k, 1000}
        zero_positions = tf.reduce_max(zero_positions, axis=1) # {N, 1000}
        zero_mask = tf.cast(tf.equal(zero_positions, 0), tf.float32)
        
        # and then generate all
        for mut_i in xrange(position_combinations.get_shape().as_list()[1]):
            
            mut_positions = position_combinations[:,mut_i,:] # {N, k}
            mut_positions = tf.one_hot(mut_positions, orig_seq_len) # {N, k, 1000}
            mut_positions = tf.reduce_max(mut_positions, axis=1) # {N, 1000}
            mut_positions = tf.multiply(mut_positions, zero_mask) # IMPORTANT TO MASK NON REAL MUTS
            mut_positions = tf.expand_dims(mut_positions, axis=2) # {N, 1000, 1}
            mut_positions = tf.expand_dims(mut_positions, axis=1) # {N, 1, 1000, 1}
            mut_motif_present = all_motifs_present # {N}
            mut_sequences, mut_positions = tf.map_fn(
                mutagenize_fn,
                [features, mut_motif_present, mut_positions],
                dtype=(tf.float32, tf.float32)) # {N, 1, 1000, 4}
            all_mut_sequences.append(mut_sequences)
            all_mut_positions.append(mut_positions)
            
        outputs[DataKeys.MUT_MOTIF_ORIG_SEQ] = tf.stack(all_mut_sequences, axis=1) # {N, mut_M, 1, 1000, 4}
        outputs[DataKeys.MUT_MOTIF_POS] = tf.stack(all_mut_positions, axis=1) # {N, mutM, 1, 1000, 1}
        
        return outputs, params


    
def mutate_weighted_motif_sites(inputs, params):
    """ mutate
    """
    mutagenizer = Mutagenizer(mutation_type="shuffle")
    outputs, params = mutagenizer.mutagenize(inputs, params)

    return outputs, params


def mutate_weighted_motif_sites_combinatorially(inputs, params):
    """mutate combinatorially
    """
    mutagenizer = SynergyMutagenizer(mutation_type="shuffle")
    outputs, params = mutagenizer.mutagenize(inputs, params)

    return outputs, params




def blank_mutation_site(inputs, params):
    """cover up the mutation site when scanning for motifs
    """
    assert inputs.get(DataKeys.FEATURES) is not None # {N, mutM, 1, pos, 4}
    assert inputs.get(DataKeys.MUT_MOTIF_POS) is not None # {N, mutM, 1, pos, 4}
    # given mutation position and mutation style, cover up the mutation site

    # features
    features = inputs[DataKeys.FEATURES]
    positions = inputs[DataKeys.MUT_MOTIF_POS]
    outputs = dict(inputs)
    
    # multiply by the opoosite
    blank_mask = tf.cast(tf.equal(positions, 0), tf.float32)

    features = tf.multiply(blank_mask, features)
    outputs[DataKeys.FEATURES] = features
    
    return outputs, params



def calculate_delta_scores(inputs, params):
    """assumes the the first example is the ref. 
    """
    features = inputs[DataKeys.FEATURES]
    outputs = dict(inputs)
    
    delta_features = tf.subtract(
        features,
        tf.expand_dims(features[0], axis=0))
    
    outputs[DataKeys.FEATURES] = delta_features
    
    return outputs, params









# OLD BELOW

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
    #outputs, params = pad_data(outputs, params)
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
        #outputs, params = pad_data(outputs, params)
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

