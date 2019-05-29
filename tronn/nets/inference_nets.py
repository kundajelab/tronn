# Description: joins various smaller nets to run analyses after getting predictions

import logging

import tensorflow as tf

from tronn.nets.importance_nets import get_task_importances
from tronn.nets.importance_nets import run_dfim

from tronn.nets.motif_nets import get_pwm_scores
from tronn.nets.motif_nets import get_motif_densities
from tronn.nets.motif_nets import filter_for_any_sig_pwms
from tronn.nets.motif_nets import run_dmim
from tronn.nets.motif_nets import extract_null_results
from tronn.nets.motif_nets import get_sig_mut_logits
from tronn.nets.motif_nets import get_sig_mut_motifs

from tronn.nets.mutate_nets import mutate_weighted_motif_sites
from tronn.nets.mutate_nets import mutate_weighted_motif_sites_combinatorially

from tronn.nets.sequence_nets import calc_gc_content
from tronn.nets.sequence_nets import decode_onehot_sequence

from tronn.util.utils import DataKeys


def sequence_to_pwm_scores(inputs, params):
    """Go from sequence (N, 1, pos, 4) to motif hits (N, motif)
    """
    # get importances
    outputs, params = get_task_importances(inputs, params)
    outputs, _ = calc_gc_content(outputs, params)
    params.update({"decode_key": DataKeys.ORIG_SEQ_ACTIVE})
    outputs, _ = decode_onehot_sequence(outputs, params)
    
    # move to CPU - GPU mostly needed for gradient calc in model
    with tf.device("/cpu:0"):
        outputs, params = get_pwm_scores(outputs, params)
        outputs, params = get_motif_densities(outputs, params)
    
    return outputs, params


def importance_scores_to_pwm_scores(inputs, params):
    """given external importance scores, get motif hits
    """
    inputs[DataKeys.ORIG_SEQ_ACTIVE] = inputs[DataKeys.ORIG_SEQ]
    inputs[DataKeys.ORIG_SEQ_ACTIVE_SHUF] = inputs[DataKeys.ORIG_SEQ_SHUF]
    inputs[DataKeys.WEIGHTED_SEQ_ACTIVE] = inputs[DataKeys.WEIGHTED_SEQ]
    inputs[DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF] = inputs[DataKeys.WEIGHTED_SEQ_SHUF]

    # for Mahfuza
    inputs[DataKeys.ORIG_SEQ_ACTIVE_SHUF] = tf.expand_dims(
        inputs[DataKeys.ORIG_SEQ_ACTIVE_SHUF], axis=1)
    inputs[DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF] = tf.expand_dims(
        inputs[DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF], axis=1)
    inputs[DataKeys.LOGITS_SHUF] = tf.expand_dims(inputs[DataKeys.LOGITS_SHUF], axis=-1)
    
    with tf.device("/cpu:0"):
        outputs, params = get_pwm_scores(inputs, params)
    
    return outputs, params


def pwm_scores_to_dmim(inputs, params):
    """For a grammar, get back the delta deeplift results on motifs, another way
    to extract dependencies at the motif level
    """
    # here - assume sequence to motif scores has already been run
    # if not set up in another fn
    print "WARNING ASSUMES PROCESSED INPUTS"
    outputs = dict(inputs)

    # filter and mutate
    with tf.device("/cpu:0"):
        outputs, params = filter_for_any_sig_pwms(inputs, params)
        outputs, params = mutate_weighted_motif_sites(outputs, params)

    # run dfim
    outputs, params = run_dfim(outputs, params)

    # dmim and sig results
    with tf.device("/cpu:0"):
        outputs, params = run_dmim(outputs, params)
        outputs, _ = extract_null_results(outputs, params)
        outputs, _ = get_sig_mut_motifs(outputs, params)
        outputs, _ = get_sig_mut_logits(outputs, params)
        
    return outputs, params


def sequence_to_synergy(inputs, params):
    """For a grammar, get back the delta deeplift results on motifs, another way
    to extract dependencies at the motif level
    """
    # here - assume sequence to dmim scores has already been run
    # if not set up in another fn
    print "WARNING ASSUMES PROCESSED INPUTS"
    outputs = dict(inputs)
    
    # mutate
    with tf.device("/cpu:0"):
        outputs, params = mutate_weighted_motif_sites_combinatorially(outputs, params)

    # run model
    outputs, params = run_dfim(outputs, params)
    
    return outputs, params


def variants_to_scores(inputs, params):
    """variants tensor to outputs
    """
    # first run importance scores/pwm scores
    params.update({"use_filtering": False})
    outputs, params = sequence_to_pwm_scores(inputs, params)

    # TO CONSIDER
    # run some nulls? <- figure out how to harness mutagenizer to do a few, if desired
    # note that would need to run nulls for BOTH ref and alt allele...

    # adjust tensors, put variant in axis=1
    for key in outputs.keys():
        outputs[key] = tf.reshape(
            outputs[key],
            [-1, 2] + list(outputs[key].get_shape().as_list()[1:]))

    # some manual adjustments
    outputs[DataKeys.ORIG_SEQ_PWM_HITS] = tf.reduce_max(
        outputs[DataKeys.ORIG_SEQ_PWM_HITS], axis=1)
    outputs["check_metadata.string"] = outputs[DataKeys.SEQ_METADATA] # a backcheck
    outputs[DataKeys.SEQ_METADATA] = outputs[DataKeys.SEQ_METADATA][:,0]

    # so just calculate_deltas
    outputs[DataKeys.DFIM_SCORES] = tf.subtract(
        outputs[DataKeys.WEIGHTED_SEQ_ACTIVE][:,0],
        outputs[DataKeys.WEIGHTED_SEQ_ACTIVE][:,1]) # {N, task, seqlen, 4}
    dfim_scores = tf.expand_dims(outputs[DataKeys.DFIM_SCORES], axis=1) # {N, 1, task, seqlen, 4}

    # set up variant mask
    seq_len = outputs[DataKeys.ORIG_SEQ].get_shape().as_list()[3]
    variant_pos = tf.one_hot(
        outputs[DataKeys.VARIANT_IDX][:,0],
        seq_len) # {N, 1000}
    variant_pos = tf.reshape(
        variant_pos,
        [-1, 1, seq_len, 1])
    variant_mask = tf.nn.max_pool(
        variant_pos, [1,1,10,1], [1,1,1,1], padding="SAME")
    left_clip = params["left_clip"]
    right_clip = params["right_clip"]
    variant_mask = variant_mask[:,:,left_clip:right_clip]
    variant_mask = tf.expand_dims(variant_mask, axis=1) # {N, 1, 1, seqlen, 1}
    
    # first dmim for the non variant sites
    outputs[DataKeys.FEATURES] = dfim_scores
    outputs[DataKeys.MUT_MOTIF_POS] = variant_pos
    outputs[DataKeys.MUT_MOTIF_MASK] = variant_mask
    final_outputs, _ = run_dmim(outputs, params)
    
    # and then for the variant site itself
    outputs[DataKeys.MUT_MOTIF_POS] = tf.cast(
        tf.equal(variant_pos, 0),
        tf.float32) # flip
    outputs[DataKeys.MUT_MOTIF_MASK] = tf.cast(
        tf.equal(variant_mask, 0),
        tf.float32) # flip 
    final_outputs[DataKeys.VARIANT_DMIM] = run_dmim(
        outputs, params)[0][DataKeys.DMIM_SCORES]
    
    return final_outputs, params
