"""description: wrappers for inference runs
"""

import os
import h5py
import json

from tronn.datalayer import setup_data_loader
from tronn.models import setup_model_manager
from tronn.util.utils import DataKeys


def _setup_input_skip_keys(args):
    """reduce tensors pulled from data files to save time/space
    """
    if (args.subcommand_name == "dmim") or (args.subcommand_name == "synergy"):
        skip_keys = [
            DataKeys.ORIG_SEQ_SHUF,
            DataKeys.ORIG_SEQ_ACTIVE_SHUF,
            DataKeys.ORIG_SEQ_PWM_SCORES,
            DataKeys.ORIG_SEQ_PWM_SCORES_THRESH,
            DataKeys.ORIG_SEQ_SHUF_PWM_SCORES,
            DataKeys.ORIG_SEQ_PWM_DENSITIES,
            DataKeys.ORIG_SEQ_PWM_MAX_DENSITIES,
            DataKeys.WEIGHTED_SEQ_SHUF,
            DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
            DataKeys.WEIGHTED_SEQ_PWM_HITS,
            DataKeys.WEIGHTED_SEQ_PWM_SCORES,
            DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH,
            DataKeys.WEIGHTED_SEQ_SHUF_PWM_SCORES,
            DataKeys.MUT_MOTIF_ORIG_SEQ,
            "{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ),
            DataKeys.MUT_MOTIF_WEIGHTED_SEQ,
            DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT,
            DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT,
            DataKeys.MUT_MOTIF_POS,
            DataKeys.MUT_MOTIF_PRESENT,
            DataKeys.MUT_MOTIF_WEIGHTED_SEQ_CI,
            DataKeys.MUT_MOTIF_WEIGHTED_SEQ_CI_THRESH,
            DataKeys.MUT_MOTIF_LOGITS,
            DataKeys.MUT_MOTIF_LOGITS_SIG,
            DataKeys.MUT_MOTIF_LOGITS_MULTIMODEL,
            DataKeys.DFIM_SCORES,
            DataKeys.DFIM_SCORES_DX,
            DataKeys.DMIM_SCORES,
            DataKeys.DMIM_SCORES_SIG,
            DataKeys.FEATURES]
        for key in skip_keys:
            prevent_null_skip = [
                "{}.string".format(DataKeys.MUT_MOTIF_ORIG_SEQ),
                DataKeys.MUT_MOTIF_POS]
            if key in prevent_null_skip:
                continue
            if "motif_mut" in key:
                skip_keys.append(key.replace("motif_mut", "null_mut"))
    elif (args.subcommand_name == "scanmotifs"):
        skip_keys = []
    else:
        skip_keys = []
        
    return skip_keys


def _setup_output_skip_keys(args):
    """reduce tensors pulled from data files to save time/space
    """
    if (args.subcommand_name == "dmim") or (args.subcommand_name == "synergy"):
        skip_keys = []
    elif args.subcommand_name == "scanmotifs":
        skip_keys = [
            DataKeys.ORIG_SEQ_SHUF,
            DataKeys.ORIG_SEQ_ACTIVE_SHUF,
            DataKeys.ORIG_SEQ_PWM_SCORES,
            DataKeys.WEIGHTED_SEQ_SHUF,
            DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
            DataKeys.WEIGHTED_SEQ_PWM_SCORES,
            DataKeys.WEIGHTED_SEQ_PWM_HITS,
            DataKeys.FEATURES,
            DataKeys.LOGITS_SHUF]
        if args.lite:
            # add other tensors to skip
            lite_keys = [
                DataKeys.IMPORTANCE_GRADIENTS,
                DataKeys.WEIGHTED_SEQ,
                DataKeys.ORIG_SEQ_PWM_SCORES_THRESH,
                DataKeys.ORIG_SEQ_PWM_HITS,
                DataKeys.ORIG_SEQ_PWM_DENSITIES,
                DataKeys.ORIG_SEQ_PWM_MAX_DENSITIES,
                DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH,
                DataKeys.WEIGHTED_SEQ_PWM_HITS]
            skip_keys += lite_keys
    elif args.subcommand_name == "simulategrammar":
        skip_keys = [
            DataKeys.ORIG_SEQ_SHUF,
            DataKeys.ORIG_SEQ_ACTIVE_SHUF,
            DataKeys.ORIG_SEQ_PWM_SCORES,
            DataKeys.WEIGHTED_SEQ_SHUF,
            DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
            DataKeys.WEIGHTED_SEQ_PWM_SCORES,
            DataKeys.WEIGHTED_SEQ_PWM_HITS,
            DataKeys.FEATURES,
            DataKeys.LOGITS_SHUF,
            DataKeys.ORIG_SEQ_PWM_DENSITIES,
            DataKeys.ORIG_SEQ_PWM_MAX_DENSITIES]
    elif args.subcommand_name == "analyzevariants":
        skip_keys = [
            DataKeys.ORIG_SEQ_SHUF,
            DataKeys.ORIG_SEQ_ACTIVE_SHUF,
            DataKeys.ORIG_SEQ_PWM_SCORES,
            DataKeys.ORIG_SEQ_PWM_SCORES_THRESH,
            DataKeys.WEIGHTED_SEQ_SHUF,
            DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
            DataKeys.WEIGHTED_SEQ_PWM_SCORES,
            DataKeys.WEIGHTED_SEQ_PWM_SCORES_THRESH,
            DataKeys.WEIGHTED_SEQ_PWM_HITS,
            DataKeys.FEATURES,
            DataKeys.LOGITS_SHUF,
            DataKeys.ORIG_SEQ_PWM_DENSITIES,
            DataKeys.ORIG_SEQ_PWM_MAX_DENSITIES]
    elif args.subcommand_name == "buildtracks":
        skip_keys = [
            DataKeys.LOGITS_CI,
            DataKeys.LOGITS_CI_THRESH,
            DataKeys.LOGITS_SHUF,
            DataKeys.LOGITS_MULTIMODEL,
            DataKeys.LOGITS_MULTIMODEL_NORM,
            DataKeys.IMPORTANCE_GRADIENTS,
            DataKeys.WEIGHTED_SEQ,
            DataKeys.WEIGHTED_SEQ_ACTIVE_CI,
            DataKeys.WEIGHTED_SEQ_ACTIVE_CI_THRESH,
            DataKeys.WEIGHTED_SEQ_ACTIVE_SHUF,
            DataKeys.WEIGHTED_SEQ_THRESHOLDS,
            DataKeys.ORIG_SEQ_ACTIVE_SHUF,
            DataKeys.FEATURES]
    else:
        skip_keys = []
        
    return skip_keys


def run_inference(args, warm_start=False):
    """convenience wrapper to mask away multi model vs single model runs
    """
    # adjust prefix and inference fn if doing a prediction warm start
    if warm_start:
        out_prefix = "{}/{}.{}.prediction_sample".format(
            args.out_dir, args.prefix, args.subcommand_name)
        real_inference_fn = args.inference_params["inference_fn_name"]
        args.inference_params["inference_fn_name"] = "empty_net"
        
        # now adjust the prefix if there was a prediction sample
        # this way, since file exists, inference won't run
        if args.prediction_sample is not None:
            out_prefix = args.prediction_sample.split(".h5")[0]
    else:
        out_prefix = None
        
    # run all files together or run rotation of models
    if args.model["name"] == "kfold_models":
        inference_files = run_multi_model_inference(
            args, out_prefix=out_prefix, positives_only=True)
    else:
        inference_files = run_single_model_inference(
            args, out_prefix=out_prefix, positives_only=True)

    if warm_start:
        args.model["params"]["prediction_sample"] = inference_files[0]
        args.inference_params["inference_fn_name"] = real_inference_fn
        args.inference_params["prediction_sample"] = inference_files[0] # TODO fix this later

    return inference_files


def run_single_model_inference(
        args,
        out_prefix=None,
        positives_only=True,
        kfold=False):
    """wrapper for inference
    """
    # set up out_file
    if out_prefix is None:
        out_file = "{}/{}.{}.h5".format(
            args.out_dir, args.prefix, args.subcommand_name)
    else:
        out_file = "{}.h5".format(out_prefix)
        
    # set up dataloader
    data_loader = setup_data_loader(args)

    # set up model, add to inference params
    model_manager = setup_model_manager(args)
    args.inference_params.update({"model": model_manager})

    # adjust for positives (normally only run inference on positives)
    if positives_only:
        data_loader = data_loader.setup_positives_only_dataloader()
    
    # adjust for kfold in model (normally no)
    if kfold:
        keep_chromosomes = model_manager.model_dataset["test"]
        data_loader = data_loader.filter_for_chromosomes(
            keep_chromosomes)

    # data file adjustments done, set up input fn
    input_fn = data_loader.build_input_fn(
        args.batch_size,
        shuffle=not args.fifo if args.fifo is not None else True,
        targets=args.targets,
        target_indices=args.target_indices,
        filter_targets=args.filter_targets,
        singleton_filter_targets=args.singleton_filter_targets,
        examples_subset=args.dataset_examples,
        use_queues=True,
        skip_keys=_setup_input_skip_keys(args))

    # skip some outputs
    args.inference_params.update(
        {"skip_outputs": _setup_output_skip_keys(args)})
    
    # also check if processed inputs, if processed then remove model
    # and replace with empty net (just send tensors through)
    if args.processed_inputs:
        args.model["name"] = "empty_net"
        args.inference_params.update({"model_reuse": False})
        model_manager = setup_model_manager(args)
    else:
        args.inference_params.update({"model_reuse": True})

    # set up inference generator
    inference_generator = model_manager.infer(
        input_fn,
        args.out_dir,
        args.inference_params,
        checkpoint=model_manager.model_checkpoint,
        yield_single_examples=True)

    # run inference and save out
    #os.system("rm {}".format(out_file)) # TODO for nautilus remove this later
    if not os.path.isfile(out_file):
        model_manager.infer_and_save_to_h5(
            inference_generator,
            out_file,
            args.sample_size,
            debug=args.debug)

        # get chrom tags and transfer in
        with h5py.File(out_file, "a") as hf:
            hf["/"].attrs["chromosomes"] = data_loader.get_chromosomes()
    
    return [out_file]


def run_multi_model_inference(
        args,
        out_prefix=None,
        positives_only=True):
    """run inference on one model
    """
    if out_prefix is None:
        out_prefix = "{}/{}.{}".format(
            args.out_dir, args.prefix, args.subcommand_name)
    
    # get the model jsons
    model_jsons = args.model["params"]["models"]
    out_files = []
    for model_idx in xrange(len(model_jsons)):

        # load the model json into args.model
        model_json = model_jsons[model_idx]
        with open(model_json, "r") as fp:
            args.model = json.load(fp)

        # generate the out file
        out_file = "{}.model-{}.h5".format(out_prefix, model_idx)
        out_files.append(out_file)
        
        # run inference
        run_inference(
            args,
            out_file=out_file,
            positives_only=positives_only,
            kfold=True)
    
    return out_files
