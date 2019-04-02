# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import json
import h5py
import glob
import logging

import networkx as nx

from tronn.datalayer import H5DataLoader
from tronn.interpretation.inference import run_inference
from tronn.util.h5_utils import add_pwm_names_to_h5
from tronn.util.h5_utils import copy_h5_datasets
from tronn.util.formats import write_to_json
from tronn.util.utils import DataKeys


def run(args):
    """Scan motifs from a PWM file
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running motif scan")

    # subset the pwms going into the dataloader
    grammar = nx.read_gml(args.grammar)
    pwm_indices = sorted([
        val[1]
        for val in list(grammar.nodes(data="pwmidx"))])
    args.grammar_pwms = [args.pwm_list[i] for i in pwm_indices]
    
    # if ensemble, require a prediction sample
    
    
    # run inference
    inference_files = run_inference(args)

    # add in PWM names to the datasets
    for inference_file in inference_files:
        add_pwm_names_to_h5(
            inference_file,
            [pwm.name for pwm in args.pwm_list],
            other_keys=[DataKeys.FEATURES])
    
    # put the files into a dataloader
    results_data_log = "{}/dataset.{}.json".format(args.out_dir, args.subcommand_name)
    results_data_loader = H5DataLoader(
        data_dir=args.out_dir, data_files=inference_files, fasta=args.fasta)
    dataset = results_data_loader.describe()
    dataset.update({
        "targets": args.targets,
        "target_indices": args.target_indices})
    write_to_json(dataset, results_data_log)

    # save out relevant inference run details for downstream runs
    infer_log = "{}/infer.{}.json".format(args.out_dir, args.subcommand_name)
    infer_vals = {
        "infer_dir": args.out_dir,
        "model_json": args.model_json,
        "targets": args.targets,
        "inference_targets": args.inference_targets,
        "target_indices": args.target_indices,
        "pwm_file": args.pwm_file}
    write_to_json(infer_vals, infer_log)

    # and now plot out the results
    plot_cmd = "plot-h5.simul_results.R {} {} {} {} {}".format(
        "{}/{}.{}.h5".format(args.out_dir, args.prefix, args.subcommand_name),
        "simul.pwm.dist",
        DataKeys.LOGITS,
        "grammar.string",
        "{}/{}.{}".format(args.out_dir, args.prefix, args.subcommand_name))
    print plot_cmd
    os.system(plot_cmd)
    
    return None

