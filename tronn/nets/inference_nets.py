# Description: joins various smaller nets to run analyses after getting predictions

import tensorflow as tf

from tronn.nets.filter_nets import filter_by_accuracy
from tronn.nets.filter_nets import filter_singleton_labels

from tronn.nets.importance_nets import get_task_importances
from tronn.nets.importance_nets import run_dfim

from tronn.nets.manifold_nets import score_distances_on_manifold
from tronn.nets.manifold_nets import filter_by_manifold_distances

from tronn.nets.motif_nets import get_pwm_scores
from tronn.nets.motif_nets import get_motif_densities
from tronn.nets.motif_nets import filter_for_significant_pwms
from tronn.nets.motif_nets import filter_for_any_sig_pwms
from tronn.nets.motif_nets import run_dmim

from tronn.nets.mutate_nets import dfim
from tronn.nets.mutate_nets import motif_dfim
from tronn.nets.mutate_nets import delta_logits
from tronn.nets.mutate_nets import blank_motif_sites

from tronn.nets.mutate_nets import mutate_weighted_motif_sites
from tronn.nets.mutate_nets import mutate_weighted_motif_sites_combinatorially

# TODO move this out
from tronn.nets.sequence_nets import onehot_to_string

from tronn.nets.util_nets import build_stack

# clean up
from tronn.nets.variant_nets import get_variant_importance_scores
from tronn.nets.variant_nets import blank_variant_sequence
from tronn.nets.variant_nets import reduce_alleles

from tronn.util.utils import DataKeys


def sequence_to_importance_scores_from_regression(inputs, params):
    """Go from sequence (N, 1, pos, 4) to importance scores (N, 1, pos, 4)
    """
    # get task importances
    outputs, params = get_task_importances(inputs, params)
    
    return outputs, params


def sequence_to_motif_scores_from_regression(inputs, params):
    """Go from sequence (N, 1, pos, 4) to motif hits (N, motif)
    """

    # get importances
    outputs, params = get_task_importances(inputs, params)

    # move to CPU - GPU mostly needed for gradient calc in model
    with tf.device("/cpu:0"):
        # scan motifs
        outputs, params = get_pwm_scores(outputs, params)
        outputs, params = get_motif_densities(outputs, params)
        # TODO when to relu the scores?
        #outputs, params = pwm_relu(outputs, params)


        # convert sequences to strings
    
    print sorted(outputs.keys())
    
    return outputs, params


def sequence_to_dmim(inputs, params):
    """For a grammar, get back the delta deeplift results on motifs, another way
    to extract dependencies at the motif level
    """
    # here - assume sequence to motif scores has already been run
    # if not set up in another fn
    print "WARNING ASSUMES PROCESSED INPUTS"
    outputs = dict(inputs)

    with tf.device("/cpu:0"):
        
        # filtering
        outputs, params = filter_for_any_sig_pwms(inputs, params)
        #outputs, params = filter_for_significant_pwms(inputs, params) # still do this, requires {N, M}
        #outputs, params = score_distances_on_manifold(outputs, params) # throw this away
        #outputs, params = filter_by_manifold_distances(outputs, params) # throw this away
        
        # mutate
        outputs, params = mutate_weighted_motif_sites(outputs, params)

    # run dfim
    outputs, params = run_dfim(outputs, params)

    with tf.device("/cpu:0"):
        # and then run dmim
        outputs, params = run_dmim(outputs, params)
        
    return outputs, params


def sequence_to_synergy(inputs, params):
    """For a grammar, get back the delta deeplift results on motifs, another way
    to extract dependencies at the motif level
    """
    # here - assume sequence to motif scores has already been run
    # if not set up in another fn
    outputs = dict(inputs)
    
    # mutate
    with tf.device("/cpu:0"):
        outputs, params = mutate_weighted_motif_sites_combinatorially(outputs, params)

    # run model
    outputs, params = run_dfim(outputs, params)
    
    return outputs, params





def variants_to_predictions(inputs, params):
    """assumes an interleaved set of features to run in model
    """

    # need to read out:
    # logits on the ref/alt - {N, task, 2} ref/alt
    # delta motif scores at the original site - {N, delta_motif}
    # importance scores at the original site - {N, task, 2} ref/alt - Ignore this for now
    # delta motif scores elsewhere - {N, delta_motif} (reverse mask of above)
    # position of the delta motif scores relative to position

    # params
    params["is_training"] = False
    params["raw-sequence-key"] = "raw-sequence"
    params["raw-sequence-clipped-key"] = "raw-sequence-clipped"
    params["raw-pwm-scores-key"] = "raw-pwm-scores"
    
    # set up importance logits
    inputs["importance_logits"] = inputs["logits"]

    if params.get("raw-sequence-key") is not None:
        inputs[params["raw-sequence-key"]] = inputs["features"]
    if params.get("raw-sequence-clipped-key") is not None:
        inputs[params["raw-sequence-clipped-key"]] = inputs["features"]
    
    inference_stack = [
        (onehot_to_string, {}),
        
        # get importance scores        
        (multitask_importances, {"relu": False}),
        (threshold_gaussian, {"stdev": 3, "two_tailed": True}),
        (normalize_to_weights, {"weight_key": "probs"}), 

        # collect importance scores at these locations
        (get_variant_importance_scores, {}),

        # think about whether to have this here or not
        (filter_singles_twotailed, {"window": 7, "min_fract": float(2)/7}),
        
        # blank the sequence, but then put together? {N, task+mask, seqlen, 4}
        (blank_variant_sequence, {}),
        
        # clip edges NOTE this has to happen after, for the coordinates to line up
        (clip_edges, {"left_clip": 400, "right_clip": 600, "clip_string": True}),

        # scan motifs
        # NOTE - no RELU here!!
        (pwm_match_filtered_convolve, {"positional-pwm-scores-key": None}), # TODO set up to adjust keys?
        (pwm_position_squeeze, {"squeeze_type": "max"}),

        # and then readjust for variant read outs
        (reduce_alleles, {})
    ]

    # build inference stack
    outputs, params = build_inference_stack(
        inputs, params, inference_stack)

    return outputs, params
