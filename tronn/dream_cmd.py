# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from tronn.models import ModelManager
from tronn.datalayer import ArrayDataLoader
from tronn.nets.nets import net_fns

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import read_pwm_file


def run(args):
    """Run activation maximization
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Dreaming (activation maximization)")
    
    # annotations - motifs (to see which changed most)
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))

    # set up random sequence if starting from random
    seq_len = 1000 # convert to parameter
    onehot_vectors = np.eye(4)
    sequence = onehot_vectors[np.random.choice(onehot_vectors.shape[0], size=seq_len)]
    #sequence = np.ones((1000, 4)) * 0.25
    sequence = np.expand_dims(np.expand_dims(sequence, axis=0), axis=0) # {1, 1, seq_len, 4}

    # desired pattern - TODO factor out
    desired_pattern = np.linspace(5, -5, num=10).astype(np.float32)
    
    # set up dataloader
    feed_dict = {
        "features": sequence,
        "labels": np.ones((1, 119)), # TODO fix this
        "example_metadata": np.array(["random"])}
    array_names = ["features", "labels", "example_metadata"]
    array_types = [tf.float32, tf.float32, tf.string]
    dataloader = ArrayDataLoader(feed_dict, array_names, array_types)
    input_fn = dataloader.build_input_fn(1)
    
    # set up model
    model_manager = ModelManager(
        net_fns[args.model["name"]],
        args.model)
    
    # set up dream generator
    with h5py.File(args.sequence_file, "a") as hf:
        num_examples = hf["raw-sequence"].shape[0]
        dream_generator = model_manager.dream(
            hf["raw-sequence"],
            input_fn,
            feed_dict,
            args.out_dir,
            net_fns[args.inference_fn],
            inference_params={
                "use_filtering": False,
                "backprop": args.backprop,
                "importance_task_indices": args.inference_tasks,
                "pwms": pwm_list,
                "dream": True,
                "keep_gradients": True,
                "all_grad_ys": desired_pattern},
            checkpoint=args.model_checkpoints[0])

        # run dream generator and save to hdf5
        model_manager.dream_and_save_to_h5(
            dream_generator,
            hf,
            "dream.results")
        
    return None

