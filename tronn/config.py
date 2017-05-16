import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
session_config = tf.ConfigProto(gpu_options=gpu_options)
