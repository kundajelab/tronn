"""Description: Contains various tensorflow utility functions
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim


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


def get_stop_step(checkpoint_dir, num_steps):
    """Add on steps in checkpoint to num steps to get final steps
    """
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint_steps = int(checkpoint.split("-")[-1].split(".")[0])
    
    return checkpoint_steps + num_steps
