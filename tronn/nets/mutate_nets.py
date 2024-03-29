"""contains code for creating batches of mutations and then analyzing them
"""

import h5py
import logging
import itertools

import numpy as np
import tensorflow as tf

from tronn.interpretation.combinatorial import setup_combinations

from tronn.nets.util_nets import rebatch
from tronn.nets.filter_nets import filter_and_rebatch
from tronn.nets.motif_nets import attach_null_indices
from tronn.util.utils import DataKeys


class Mutagenizer(object):
    """basic in silico mutagenesis"""

    def __init__(self, mutation_type="point"):
        self.mutation_type = mutation_type

        
    def preprocess(self, inputs, params):
        """extract the positions for only the desired motifs (sig pwms)
        """
        assert inputs.get(DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX) is not None
        assert inputs.get(DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL) is not None
        assert params.get("sig_pwms") is not None

        # collect features, TODO move others to old?
        indices = tf.cast(inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX], tf.int64)
        vals = inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL]
        outputs = dict(inputs)
        
        # gather sig pwm max positions and vals
        sig_pwms_indices = np.where(params["sig_pwms"])[0] # {mutM}
        params["num_aux_examples"] = sig_pwms_indices.shape[0]
        indices = tf.gather(indices, sig_pwms_indices, axis=1)
        vals = tf.gather(vals, sig_pwms_indices, axis=1)
        
        # save adjusted
        outputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT] = vals # {N, mut_M, k}
        outputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT] = indices # {N, mut_M, k}
        outputs[DataKeys.MUT_MOTIF_PRESENT] = tf.reduce_any(tf.greater(vals, 0), axis=2) # {N, mut_M}

        # and add null indices if present
        if inputs.get(DataKeys.NULL_PWM_POSITION_INDICES) is not None:
            outputs, _ = attach_null_indices(outputs, params)
        
        if self.mutation_type == "point":
            # set up gradients
            grad = inputs[DataKeys.IMPORTANCE_GRADIENTS]
            grad_mins = tf.reduce_min(grad, axis=1, keepdims=True) # {N, 1, 1000, 4}
            grad_mins = tf.argmin(grad_mins, axis=3) # {N, 1, 1000}
            grad_mins = tf.one_hot(grad_mins, 4) # {N, 1, 1000, 4}
            outputs["grad_mins"] = grad_mins # these are the worst case base pairs at each position

            # adjust positions
            outputs, _ = self.select_best_point_mutants_multiple(outputs, params)
            
        return outputs, params

    
    @staticmethod
    def select_best_point_mutants(positions, importances, gradients, orig_seq_len):
        """adjust positions for those with worst gradients (which are also the
        base pairs to change to)
        """
        window_size = 10
        
        # set up mask positions
        mut_masks = []
        for mut_i in xrange(positions.get_shape().as_list()[1]):
            mut_positions = positions[:,mut_i] # {N}
            mut_positions = tf.one_hot(mut_positions, orig_seq_len) # {N, 1000}
            mut_positions = tf.expand_dims(mut_positions, axis=2) # {N, 1000, 1}
            mut_positions = tf.expand_dims(mut_positions, axis=1) # {N, 1, 1000, 1}, nn.max_pool requires 4D tensor
            mut_positions = tf.nn.max_pool(
                mut_positions, [1,1,window_size,1], [1,1,1,1], padding="SAME")
            mut_positions = tf.cast(tf.not_equal(mut_positions, 0), tf.float32)
            mut_masks.append(mut_positions)
        mut_masks = tf.concat(mut_masks, axis=1) # {N, mutM, 1000, 1}
        
        # use HIGHEST importance position to mark best spot to mutate
        importance_max = tf.reduce_max(importances, axis=(1,3), keepdims=True) # {N, 1, 1000, 1}
        masked_importance_max = tf.multiply(importance_max, mut_masks) # {N, mutM, 1000, 1}
        masked_importance_max = tf.reduce_max(masked_importance_max, axis=-1) # {N, mutM, 1000}

        # get best indices
        _, new_positions = tf.nn.top_k(masked_importance_max, k=1, sorted=True) # {N, mutM, 1}
        new_positions = tf.squeeze(new_positions, axis=-1) # {N, mutM}
        
        return new_positions


    @classmethod
    def select_best_point_mutants_multiple(cls, inputs, params):
        """do this across k positions per example
        """
        # get positions and gradients
        k = inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT].get_shape().as_list()[2]

        # for each position, select best point mutant
        positions = []
        for k_idx in range(k):
            new_positions = cls.select_best_point_mutants(
                inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT][:,:,k_idx], # {N, mutM}
                inputs[DataKeys.WEIGHTED_SEQ],
                inputs[DataKeys.IMPORTANCE_GRADIENTS],
                inputs[DataKeys.ORIG_SEQ].get_shape().as_list()[2]) # {N, mutM}
            positions.append(new_positions)

        positions = tf.stack(positions, axis=-1) # {N, mutM, k}
        inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT] = positions
        
        return inputs, params

    
    @staticmethod
    def setup_blanking_mask(positions, blanking_width=10):
        """given positions tensor, set up a blanking mask
        """
        positions = tf.reduce_max(positions, axis=-1, keepdims=True) # {1, 1000, 1}
        positions = tf.expand_dims(positions, axis=0) # {1, 1, 1000, 1}
        positions = tf.nn.max_pool(
            positions,
            [1,1,blanking_width,1],
            [1,1,1,1],
            padding="SAME") # {1, 1, 1000, 1}
        positions = tf.squeeze(positions, axis=0) # {1, 1000, 1}
        
        return positions

    
    @staticmethod
    def point_mutagenize_example(tensors):
        """generate a point mutant
        """
        # extract tensors
        example = tensors[0] # {1, 1000, 4}
        positions = tensors[1] # {1, 1000, 1}
        grad_mins = tensors[2] # {1, 1000, 4}

        # adjust positions for tf.where
        positions = tf.concat([positions]*4, axis=2) # {1, 1000, 4}
        
        # adjust tensor
        example_mut = tf.where(
            tf.equal(positions, 0),
            example,
            grad_mins) # {1, 1000, 4}
        
        return example_mut


    @staticmethod
    def shuffle_mutagenize_example(tensors):
        """generate shuffle mutants
        """
        # extract tensors
        example = tensors[0] # {1, 1000, 4}
        positions = tensors[1] # {1, 1000, 1}
        
        # adjustments
        example = tf.squeeze(example, axis=0) # {1000, 4}
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
        example_mut = tf.expand_dims(example_mut, axis=0)
        
        return example_mut

    
    @staticmethod
    def position_indices_to_mask(position_indices, orig_seq_len, aux_mask=None):
        """convert position indices {N, k} to one hot
        """
        mask = tf.one_hot(position_indices, orig_seq_len) # {N, k, 1000}
        mask = tf.reduce_max(mask, axis=1, keepdims=True) # {N, 1, 1000}
        mask = tf.expand_dims(mask, axis=3) # {N, 1, 1000, 1}
        if aux_mask is not None:
            mask = tf.multiply(mask, aux_mask) # apply any other mask that you might want

        return mask

    
    def mutagenize_multiple_motifs(self, inputs, params):
        """comes in from map fn?
        """
        assert inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT] is not None # {N, mut_M, k}
        outputs = dict(inputs)
        
        # for each motif, run map fn to mutagenize
        features = inputs[DataKeys.ORIG_SEQ] # {N, 1, 1000, 4}
        orig_seq_len = features.get_shape().as_list()[2]
        position_indices = inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT]
        motif_present = inputs[DataKeys.MUT_MOTIF_PRESENT] # {N, mut_M}

        # params
        shuffle_width = params.get("mutate_shuffle_width", 10)
        blank_width = params.get("mutate_mask_width", 14)
        
        # choose what kind of mutagenesis
        if self.mutation_type == "point":
            mutagenize_fn = self.point_mutagenize_example
            replace_seq = inputs["grad_mins"]
        else:
            mutagenize_fn = self.shuffle_mutagenize_example
            replace_seq = tf.identity(features)

        # set up edge mask to mask positions on edges
        edge_mask = tf.cast(tf.greater(tf.range(orig_seq_len), 10), tf.float32) # {1000}
        edge_mask = tf.reshape(edge_mask, (1, 1, orig_seq_len, 1))

        # go through each mut motif
        all_mut_sequences = []
        all_mut_masks = []
        all_blanking_masks = []
        for mut_i in range(position_indices.get_shape().as_list()[1]):
            mut_position_indices = position_indices[:,mut_i,:] # {N, k}
            mut_motif_present = motif_present[:,mut_i] # {N}

            # convert position indices to a position mask and also get a null mask
            mut_mask = Mutagenizer.position_indices_to_mask(
                mut_position_indices, orig_seq_len, aux_mask=edge_mask)
            no_mut_mask = tf.zeros(mut_mask.get_shape())

            # and adjust position mask based on whether the motif was present or not
            mut_mask = tf.where(mut_motif_present, mut_mask, no_mut_mask)

            # create a blanking mask (must do here to know exactly where orig POINT position was)
            blank_mask = tf.nn.max_pool(
                mut_mask, [1,1,blank_width,1],[1,1,1,1], padding="SAME") # {N, 1, 1000, 1}

            # finally adjust if shuffle for larger window
            if self.mutation_type == "shuffle":
                mut_mask = tf.nn.max_pool(
                    mut_mask, [1,1,shuffle_width,1], [1,1,1,1], padding="SAME") # {N, 1, 1000, 1}
            
            # mutagenize
            mut_sequences = tf.map_fn(
                mutagenize_fn,
                [features, mut_mask, replace_seq],
                dtype=tf.float32) # {N, 1, 1000, 4}

            # but only keep the mutant sequence if actually mutated
            mut_sequences = tf.where(mut_motif_present, mut_sequences, replace_seq)
            
            # save out: mut sequence, mut mask, blank mask
            all_mut_sequences.append(mut_sequences)
            all_mut_masks.append(mut_mask)
            all_blanking_masks.append(blank_mask)
        
        # outputs: mutated sequences, position mask of mutated positions, and a larger blanking mask
        outputs[DataKeys.MUT_MOTIF_ORIG_SEQ] = tf.stack(all_mut_sequences, axis=1) # {N, mut_M, 1, 1000, 4}
        outputs[DataKeys.MUT_MOTIF_POS] = tf.stack(all_mut_masks, axis=1) # {N, mutM, 1, 1000, 1}
        outputs[DataKeys.MUT_MOTIF_MASK] = tf.stack(all_blanking_masks, axis=1) # {N, mutM, 1, 1000, 1}
        
        return outputs, params


    def mutagenize(self, inputs, params):
        """wrapper function to call everything
        """
        inputs, params = self.preprocess(inputs, params)
        outputs, params = self.mutagenize_multiple_motifs(inputs, params)
        outputs, params = self.postprocess(outputs, params)

        return outputs, params
    

    def postprocess(self, inputs, params):
        # depends on the anlaysis

        
        return inputs, params


class SynergyMutagenizer(Mutagenizer):
    """Generates sequences for synergy scores"""

        
    def preprocess(self, inputs, params):
        """extract the positions for only the desired motifs (sig pwms)
        """
        assert inputs.get(DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX) is not None
        assert inputs.get(DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL) is not None
        assert params.get("sig_pwms") is not None

        # collect features, TODO move others to old?
        indices = tf.cast(inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX], tf.int64)
        vals = inputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL]
        outputs = dict(inputs)
        
        # gather sig pwm max positions and vals
        sig_pwms_indices = np.where(params["sig_pwms"])[0] # {mutM}
        params["num_aux_examples"] = sig_pwms_indices.shape[0]
        indices = tf.gather(indices, sig_pwms_indices, axis=1) # {N, mutM, 1}
        vals = tf.gather(vals, sig_pwms_indices, axis=1) # {N, mutM, 1}
        assert indices.get_shape().as_list()[2] == 1
        
        # and adjust for combinations
        mut_combinations = params["combinations"]
        mut_combinations = np.expand_dims(mut_combinations, axis=0) # {1, mutM, 2**mutM}        
        combination_indices = tf.multiply(indices, mut_combinations) # {N, mutM, 2**mutM}
        combination_indices = tf.transpose(combination_indices, perm=[0,2,1]) # {N, 2**mutM, mutM}
        combination_vals = tf.multiply(vals, mut_combinations) # {N, mutM, 2**mutM}
        combination_vals = tf.transpose(combination_vals, perm=[0,2,1]) # {N, 2**mutM, mutM}
        num_combinations = combination_indices.get_shape().as_list()[1]
        
        # save adjusted
        outputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT] = combination_vals # {N, 2**mutM, mutM}
        outputs[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT] = combination_indices # {N, 2**mutM, mutM}
        outputs[DataKeys.MUT_MOTIF_PRESENT] = tf.reduce_all(tf.greater(vals, 0), axis=(1,2)) # {N}
        outputs[DataKeys.MUT_MOTIF_PRESENT] = tf.stack(
            [outputs[DataKeys.MUT_MOTIF_PRESENT] for i in range(num_combinations)], axis=1) # {N, 2**mutM, mutM}

        # and add null indices if present
        #if inputs.get(DataKeys.NULL_PWM_POSITION_INDICES) is not None:
        #    outputs, _ = attach_null_indices(outputs, params)
        
        if self.mutation_type == "point":
            # set up gradients
            grad = inputs[DataKeys.IMPORTANCE_GRADIENTS]
            grad_mins = tf.reduce_min(grad, axis=1, keepdims=True) # {N, 1, 1000, 4}
            grad_mins = tf.argmin(grad_mins, axis=3) # {N, 1, 1000}
            grad_mins = tf.one_hot(grad_mins, 4) # {N, 1, 1000, 4}
            outputs["grad_mins"] = grad_mins

            # adjust positions
            outputs, _ = self.select_best_point_mutants_multiple(outputs, params)
            
        return outputs, params


    
def mutate_weighted_motif_sites(inputs, params):
    """ mutate
    """
    mutagenizer = Mutagenizer(mutation_type=params["mutate_type"])
    outputs, params = mutagenizer.mutagenize(inputs, params)

    return outputs, params


def mutate_weighted_motif_sites_combinatorially(inputs, params):
    """mutate combinatorially
    """
    mutagenizer = SynergyMutagenizer(mutation_type=params["mutate_type"])
    outputs, params = mutagenizer.mutagenize(inputs, params)

    return outputs, params
