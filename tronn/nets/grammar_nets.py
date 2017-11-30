"""Contains nets to help run grammars
"""

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tronn.util.tf_utils import get_fan_in
from tronn.util.initializers import pwm_simple_initializer


def _load_grammar_file(grammar_file):
    """Load a grammar file in a specific format
    """

    # TODO(dk) organize code below into this function

    
    
    return



def single_grammar(features, labels, model_params, is_training=False):
    """Sets up a linear grammar
    """
    # look in grammar file to get number of motifs
    num_motifs = 0
    with open(model_params["grammar_file"], 'r') as fp:
        for line in fp:
            if "num_motifs" in line:
                num_motifs = int(line.strip().split()[-1])
                break
    assert num_motifs != 0, "Grammar file is missing num_motifs"
            
    # get tables of weights from grammar file
    df = pd.read_table(model_params["grammar_file"], header=None, names=range(num_motifs), comment="#")
    table_names = ["motif_names", "non_interacting_coefficients", "pairwise_interacting_coefficients"]
    groups = df[0].isin(table_names).cumsum()
    tables = {g.iloc[0,0]: g.iloc[1:] for k,g in df.groupby(groups)}

    # adjust names etc
    motif_names_list = tables["motif_names"].iloc[0,:].values.tolist()
    tables["non_interacting_coefficients"].columns = motif_names_list
    tables["pairwise_interacting_coefficients"].columns = motif_names_list
    tables["pairwise_interacting_coefficients"].index = motif_names_list

    # select which PWMs you actually want from file
    # they MUST be ordered correctly
    # also get max length of motif
    pwm_list = []
    max_size = 0
    for pwm_name in motif_names_list:
        pwm = model_params["pwms"][pwm_name]
        pwm_list.append(pwm)
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]
    
    # set up motif out vector
    num_filters = len(pwm_list)
    conv1_filter_size = [1, max_size]
    with slim.arg_scope(
            [slim.conv2d],
            padding='VALID',
            activation_fn=None,
            weights_initializer=pwm_simple_initializer(
                conv1_filter_size,
                pwm_list,
                get_fan_in(features),
                num_filters),
            biases_initializer=None,
            scope="motif_scan"):
        net = slim.conv2d(features, num_filters, conv1_filter_size)

        # get max motif val for each motif
        width = net.get_shape()[2]
        net = slim.max_pool2d(net, [1, width], stride=[1, 1])

        # and squeeze out other dimensions
        net = tf.squeeze(net)

    # get linear coefficients and multiply
    independent_vals = tf.multiply(
        net, tables["non_interacting_coefficients"].values)

    # get pairwise coefficients and multiply
    pairwise_multiply = tf.multiply(
        tf.stack([net for i in range(num_filters)], axis=1),
        tf.stack([net for i in range(num_filters)], axis=2))
    pairwise_vals = tf.multiply(
        pairwise_multiply, tables["pairwise_interacting_coefficients"].values)
    
    final_score = tf.add(
        tf.reduce_sum(independent_vals, axis=1),
        tf.reduce_sum(pairwise_vals, axis=[1, 2]))
    
    return final_score


def multiple_grammars(features, labels, model_params, is_training=False):
    """ Run multiple linear grammars
    """
    grammar_param_sets = model_params["grammars"]
    
    # set up multiple grammars
    scores = [single_grammar(
        features,
        labels,
        {"pwms": model_params["pwms"],
         "grammar_file": grammar_param_set},
        is_training=False)
              for grammar_param_set in grammar_param_sets]

    # add in a max score
    max_score = tf.reduce_max(tf.stack(scores, axis=1), axis=1)
    print max_score.get_shape()
    scores.append(max_score)

    score_tensor = tf.stack(scores, axis=1)
    print score_tensor.get_shape()
    
    return score_tensor


def _grammar_module(features, grammar, threshold=False, nonlinear=False):
    """Set up grammar part of graph
    This is separated out in case we do more complicated grammar analyses 
    (such as involving distances, make decision trees, etc)
    """

    # bad: clean up later
    from tronn.interpretation.motifs import PWM
    
    # get various sizes needed to instantiate motif matrix
    num_filters = len(grammar.pwms)

    max_size = 0
    for pwm in grammar.pwms:
        # TODO write out order so that we have it
        
        if pwm.weights.shape[1] > max_size:
            max_size = pwm.weights.shape[1]


    # run through motifs and select max score of each
    conv1_filter_size = [1, max_size]
    with slim.arg_scope([slim.conv2d],
                        padding='VALID',
                        activation_fn=None,
                        weights_initializer=pwm_initializer(conv1_filter_size, grammar.pwms, get_fan_in(features), num_filters),
                        biases_initializer=None):
        net = slim.conv2d(features, num_filters, conv1_filter_size)

        width = net.get_shape()[2]
        net = slim.max_pool2d(net, [1, width], stride=[1, 1])

        # and squeeze it
        net = tf.squeeze(net)

        if len(net.get_shape()) == 1:
            return net

        # and then just do a summed score for now? averaged score?
        # TODO there needs to be some sort of cooperative nonlinearity?
        # consider a joint multiplication of everything
        if nonlinear:
            net = tf.reduce_prod(net, axis=1)
        else:
            net = tf.reduce_mean(net, axis=1)
        
        # TODO throw in optional threshold
        if threshold:
            print "not yet implemented!"

    return net


def grammar_scanner(features, grammars, normalize=True):
    """Sets up grammars to run

    Args:
      features: input sequence (one hot encoded)
      grammars: dictionary, grammar name to grammar object

    Returns:
      out_tensor: vector of output values after running grammars
    """

    if normalize:
        features_norm = (features - 0.25) / 0.4330127
    else:
        features_norm = features
    
    grammar_out = []
    for grammar_key in sorted(grammars.keys()):
        grammar_out.append(_grammar_module(features_norm, grammars[grammar_key]))
        
    # then pack it into a vector and return vector
    out_tensor = tf.stack(grammar_out, axis=1)

    return out_tensor
