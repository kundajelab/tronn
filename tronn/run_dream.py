# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from tronn.graphs import TronnGraph
from tronn.graphs import TronnNeuralNetGraph

from tronn.datalayer import load_data_from_feed_dict
from tronn.nets.nets import net_fns

from tronn.util.tf_utils import setup_tensorflow_session
from tronn.util.tf_utils import close_tensorflow_session

from tronn.util.tf_ops import restore_variables_op


def run(args):
    """Run activation maximization
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Dreaming")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir
    
    # motif annotations
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))

    # set up random sequence
    onehot_vectors = np.eye(4)
    sequence = onehot_vectors[np.random.choice(onehot_vectors.shape[0], size=1000)]
    #sequence = np.ones((1000, 4)) * 0.25
    sequence = np.expand_dims(np.expand_dims(sequence, axis=0), axis=0)
    sequences = [sequence]
        
    # set up dataloader
    feed_dict = {
        "features:0": sequences[0],
        "labels:0": np.ones((1, 119)), # TODO fix this
        "metadata:0": np.array(["random"])}
    dataloader = ArrayDataLoader(feed_dict)
    input_fn = dataloader.build_input_fn(feed_dict, 1)
    
    # set up model
    model_manager = ModelManager(
        net_fns[args.model["name"]],
        args.model)
    
    # set up dream generator
    with h5py.File(args.sequence_file, "r") as hf:
        num_examples = hf["features"].shape[0]
        dream_generator = model_manager.dream(
            hf["features"],
            input_fn,
            args.out_dir,
            net_fns[args.inference_fn],
            inference_params={
                "backprop": args.backprop,
                "importance_task_indices": args.inference_tasks,
                "pwms": pwm_list,
                "dream": True,
                "dream_pattern": desired_pattern},
            checkpoint=args.model_checkpoints[0])

        # run dream generator and save to hdf5
        dream_and_save_to_h5(
            dream_generator,
            feed_dict,
            args.sequence_file,
            "dream.results",
            num_iter=num_examples)
                
    return None

