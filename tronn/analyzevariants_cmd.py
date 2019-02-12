# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter
from tronn.datalayer import H5DataLoader
from tronn.interpretation.inference import run_inference
from tronn.preprocess.variants import generate_new_fasta
from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.utils import DataKeys


def run(args):
    """command to analyze variants
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Analyzing variants")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir

    # adjust for ataloader
    args.targets = []
    args.target_indices = []
    args.filter_targets = []
    args.singleton_filter_targets = []
    args.dataset_examples = None
    args.processed_inputs = False
        
    # set up ref fasta
    args.ref_fasta = "{}/{}.ref.fasta".format(
        args.tmp_dir,
        os.path.basename(args.fasta.split(".fa")[0]))
    if not os.path.isfile(args.ref_fasta):
        generate_new_fasta(args.vcf_file, args.fasta, args.ref_fasta)

    # set up alt fasta
    args.alt_fasta = "{}/{}.alt.fasta".format(
        args.tmp_dir,
        os.path.basename(args.fasta.split(".fa")[0]))
    if not os.path.isfile(args.alt_fasta):
        generate_new_fasta(args.vcf_file, args.fasta, args.alt_fasta, ref=False)

    # prefer to have a prediction sample!
    # collect a prediction sample if ensemble (for cross model quantile norm)
    # always need to do this if you're repeating backprop
    if args.model["name"] == "ensemble":
        assert args.prediction_sample is not None
    
    # run inference
    inference_files = run_inference(args)

    # add in PWM names to the datasets
    for inference_file in inference_files:
        add_pwm_names_to_h5(
            inference_file,
            [pwm.name for pwm in args.pwm_list],
            other_keys=[DataKeys.FEATURES])
    
    return None

