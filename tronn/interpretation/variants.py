# description: code for dealing with variants/mutants

import h5py
import re
import logging

import numpy as np
import pandas as pd

from tronn.stats.nonparametric import run_delta_permutation_test
from tronn.stats.nonparametric import threshold_by_qvalues

from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys


def get_differential_variants(
        h5_file,
        stdev_thresh=2,
        logits_key=DataKeys.LOGITS):
    """calculate differences and assume a gaussian distribution of differences
    """
    with h5py.File(h5_file, "r") as hf:
        variant_logits = hf[logits_key][:]

    # get diffs
    diffs = variant_logits[:,0] - variant_logits[:,1]


    
    # get mean and stds
    stdevs = np.std(diffs, axis=0, keepdims=True)
    means = np.mean(diffs, axis=0, keepdims=True)
    differential = np.abs(diffs) > stdev_thresh * stdevs
    
    # and save into file
    with h5py.File(h5_file, "a") as hf:
        if hf.get(DataKeys.VARIANT_SIG) is not None:
            del hf[DataKeys.VARIANT_SIG]
        hf.create_dataset(DataKeys.VARIANT_SIG, data=differential)
        
    return


def annotate_variants(
        h5_file,
        pwms,
        sig_only=True):
    """annotate variants with extra information
    """
    # get indices and diff
    with h5py.File(h5_file, "r") as hf:
        if sig_only:
            indices = np.where(
                np.any(hf[DataKeys.VARIANT_SIG][:] != 0, axis=1))[0]
        else:
            indices = np.array(range(hf[DataKeys.VARIANT_SIG].shape[0]))
        diffs = hf[DataKeys.LOGITS][:,0] - hf[DataKeys.LOGITS][:,1] # ref - alt
        diffs = diffs[indices]
        variant_ids = hf[DataKeys.VARIANT_ID][:,0][indices]
        orig_indices = np.arange(hf[DataKeys.VARIANT_SIG].shape[0])[indices]

    # ignore H3K4me1 for now
    diffs = diffs[:,0:16]
    
    # get max best PWM at variant site
    with h5py.File(h5_file, "r") as hf:
        #for key in sorted(hf.keys()): print key, hf[key].shape
        variant_dmim = hf[DataKeys.VARIANT_DMIM] # {N, 1, 10, 187}
        variant_dmim = np.max(np.abs(variant_dmim), axis=(1,2)) # {N, 187}

        # set up a filter fn
        dmim_zero = np.equal(np.sum(variant_dmim, axis=1), 0)
        dmim_zero = np.expand_dims(dmim_zero, axis=1)

        # get top k and blank out rows where no motif implicated
        variant_dmim = np.argsort(variant_dmim, axis=1)[:,-3:]
        variant_dmim = np.where(dmim_zero, [[-1,-1,-1]], variant_dmim)
        variant_dmim = variant_dmim[indices]
        
        if False:
            variant_dmim = np.argmax(variant_dmim, axis=1) # {N}
            variant_dmim = np.where(dmim_zero, -1, variant_dmim)
            variant_dmim = variant_dmim[indices]
    print variant_dmim.shape
            
    # get names
    best_pwm_names = []
    for var_idx in range(variant_dmim.shape[0]):
        pwm_names = []
        for indices_idx in range(variant_dmim.shape[1]):
            if variant_dmim[var_idx,indices_idx] != -1:
                pwm_name = pwms[variant_dmim[var_idx,indices_idx]].name
                pwm_name = re.sub("HCLUST-\d+.", "", pwm_name)
                pwm_name = pwm_name.replace(".UNK.0.A", "")
                pwm_names.append(pwm_name)
        best_pwm_names.append(",".join(pwm_names))
            
    # aggregate results
    results = {
        "diff_abs": np.abs(np.max(diffs, axis=1)),
        "pwm": best_pwm_names,
        "example_indices": orig_indices,
        "id": variant_ids}
    for task_idx in range(diffs.shape[1]):
        key = "diffs-{:02d}".format(task_idx)
        results[key] = diffs[:,task_idx]

    if False:
        for key in results.keys():
            try:
                print key, results[key].shape
            except:
                print key
        
    summary = pd.DataFrame(results)
    summary = summary.sort_values("diff_abs", ascending=False)
    summary.to_csv("test.txt", sep="\t")
    
    # get max best DMIM PWM (at variant site) (top 3)

    # get max best DMIM PWM (at non variant site) (top 3)

    
    return
