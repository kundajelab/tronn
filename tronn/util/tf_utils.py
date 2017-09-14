"""Description: Contains various tensorflow utility functions
"""

import tensorflow as tf


def get_fan_in(tensor, type='NHWC'):
    """Get the fan in (number of in channels)

    Args:
      tensor: the incoming tensor

    Returns:
      the shape of the final dimension
    """
    return int(tensor.get_shape()[-1])


def add_var_summaries(var):
    """Add variable summaries
    
    Args:
      var: variable you want information on

    Returns:
      None
    """
    with tf.name_scope('summaries'):
        with tf.name_scope(filter(str.isalnum, str(var.name))):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

            
def add_summaries(name_value):
    """Add summaries for a variable

    Args:
      name_value: a dictionary of names and values
    
    Returns:
      None
    """
    for name, value in name_value.iteritems():
        if value.get_shape().ndims==0:
            tf.summary.scalar(name, value)
        else:
            tf.summary.histogram(name, value)


def make_summary_op(metric_values, print_out=False):
    """After adding summaries, merge and make op
    """
    if print_out:
        values_for_printing = [
            tf.train.get_global_step(),
            metric_values['mean_loss'],
            metric_values['loss'],
            metric_values['total_loss'],
            metric_values['mean_accuracy'],
            metric_values['mean_auprc'],
            metric_values['mean_auroc']]
        summary_op = tf.Print(tf.summary.merge_all(), values_for_printing)
    else:
        summary_op = tf.summary.merge_all()

    return summary_op
            
            
def print_param_count():
    """Get number of params for a model of interest (after setting it up in graph)
    """
    model_params = sum(v.get_shape().num_elements() for v in tf.model_variables())
    trainable_params = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
    total_params = sum(v.get_shape().num_elements() for v in tf.global_variables())
    for var in sorted(tf.trainable_variables(), key=lambda var: (var.name, var.get_shape().num_elements())):
        num_elems = var.get_shape().num_elements()
        if num_elems > 500:
            print var.name, var.get_shape().as_list(), num_elems
    print 'Num params (model/trainable/global): %d/%d/%d' % (model_params, trainable_params, total_params)
            
    return


            
def setup_tensorflow_session():
    """Start up session in a graph
    """
    # set up session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # start queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    return sess, coord, threads


def close_tensorflow_session(coord, threads):
    """Cleanly close a running graph
    """
    coord.request_stop()
    coord.join(threads)
    
    return None


def get_checkpoint_steps(checkpoint):
    """Add on steps in checkpoint to num steps to get final steps
    """
    checkpoint_steps = int(checkpoint.split("-")[-1].split(".")[0])
    
    return checkpoint_steps
