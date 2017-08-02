"""Contains nets to help run grammars
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim


def single_grammar(features, labels, model_params, is_training=False):
    """Sets up a linear grammar
    """

    # get tables of weights
    df = pd.read_table(model_params["grammar_file"], header=None, names=range(len(model_params["pwms"])))
    table_names = ["Non_interacting_coefficients", "Pairwise_interacting_coefficients"]
    groups = df[0].isin(table_names).cumsum()
    tables = {g.iloc[0,0]: g.iloc[1:] for k,g in df.groupby(groups)}

    # set up motif out vector
    num_filters = len(model_params["pwms"])
    conv1_filter_size = [1, max_size]
    with slim.arg_scope([slim.conv2d],
                        padding='VALID',
                        activation_fn=None,
                        weights_initializer=pwm_initializer(conv1_filter_size, model_params["pwms"], get_fan_in(features), num_filters),
                        biases_initializer=None,
                        scope="motif_scan"):
        net = slim.conv2d(features, num_filters, conv1_filter_size)

        # get max motif val
        width = net.get_shape()[2]
        net = slim.max_pool2d(net, [1, width], stride=[1, 1])

        print net.get_shape()

    # get linear coefficients and multiply
    independent_vals = tf.multiply(net, tables["Non_interacting_coefficients"])

    # get pairwise coefficients and multiply
    pairwise_multiply = tf.multiply(
        tf.stack([net for i in range(num_filters)], axis=0),
        tf.stack([net for i in range(num_filters)], axis=1))
    pairwise_vals = tf.multiply(pairwise_multiply, tables["Pairwise_interacting_coefficients"])

    final_score = tf.add(tf.reduce_mean(independent_vals), tf.reduce_mean(pairwise_vals))

    # TODO run an activation function and train it?
    
    return final_score


def multiple_grammars(features, labels, model_params, is_training=False):
    """ Run multiple linear grammars
    """

    grammar_param_sets = model_params["grammars"]

    # set up multiple grammars
    scores = [single_grammar(features, labels, grammar_param_set, is_training=False)
              for grammar_param_set in grammar_param_sets]
    
    # run a max
    final_score = tf.reduce_max(scores)
    
    return final_score




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
