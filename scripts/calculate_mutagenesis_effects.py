"""description: code to calculate perturbation effects after synergy
"""

import os
import re
import sys
import glob
import h5py
import logging
import argparse

import numpy as np

from tronn.interpretation.combinatorial import setup_combinations
from tronn.stats.nonparametric import run_delta_permutation_test
from tronn.util.h5_utils import AttrKeys
from tronn.util.utils import DataKeys



def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="calculate mutagenesis effects")

    # required args
    parser.add_argument(
        "--synergy_file",
        help="h5 file with synergy results")
    parser.add_argument(
        "--calculations", nargs="+", default=[],
        help="calculations to perform, in the format {FOREGROUND}/{BACKGROUND} - ex 110;100")

    # parse args
    args = parser.parse_args()

    return args


def track_runs(args):
    """track command and github commit
    """
    # keeps track of restores (or different commands) in folder
    subcommand_name = "calculate_mutagenesis_effects"
    num_restores = len(glob.glob('{0}/{1}.command*'.format(args.out_dir, subcommand_name)))
    logging_file = '{0}/{1}.command_{2}.log'.format(args.out_dir, subcommand_name, num_restores)
    
    # track github commit
    git_repo_path = os.path.dirname(os.path.realpath(__file__))
    os.system('echo "commit:" > {0}'.format(logging_file))
    os.system('git --git-dir={0}/.git rev-parse HEAD >> {1}'.format(
        git_repo_path.split("/scripts")[0], logging_file))
    os.system('echo "" >> {0}'.format(logging_file))
    
    # write out the command
    with open(logging_file, 'a') as f:
        f.write(' '.join(sys.argv)+'\n\n')
    
    return logging_file


def _setup_logs(args):
    """set up logging
    """
    logging_file = track_runs(args)
    reload(logging)
    logging.basicConfig(
        filename=logging_file,
        level=logging.DEBUG, # TODO ADJUST BEFORE RELEASE
        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    for arg in sorted(vars(args)):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    logging.info("")

    return


def parse_calculation_strings(args):
    """form the strings into arrays
    """
    calculations = []
    for calculation in args.calculations:
        calculation = calculation.split("/")
        foreground = np.fromstring(
            ",".join(calculation[0].replace("x", "0")), sep=",")
        background = np.fromstring(
            ",".join(calculation[1].replace("x", "0")), sep=",")
        calculations.append((foreground, background))
    args.calculations = calculations
    
    return None


def _setup_output_string(pwm_names, presence):
    """set up string to be more human readable
    """
    presence_strings = []
    for pwm_i in xrange(len(presence)):
        if presence[pwm_i] != 0:
            presence_strings.append("{}+".format(pwm_names[pwm_i]))
        else:
            presence_strings.append("{}-".format(pwm_names[pwm_i]))

    return presence_strings

def _make_conditional_string(foreground_strings, background_strings):
    """make string into conditional
    """
    changing = []
    conditioned_on = []
    for i in xrange(len(foreground_strings)):
        foreground_presence = foreground_strings[i]
        background_presence = background_strings[i]

        if foreground_presence == background_presence:
            conditioned_on.append(foreground_presence)
        else:
            changing.append(foreground_presence)

    assert len(changing) == 1

    # make string
    new_string = "f({} | {})".format(changing[0], ",".join(conditioned_on))
    
    return new_string


def main():
    """run calculations
    """
    # set up args
    args = parse_args()
    args.out_dir = os.path.dirname(args.synergy_file)
    _setup_logs(args)

    # now set up the indices
    parse_calculation_strings(args)

    # read in data
    with h5py.File(args.synergy_file, "r") as hf:
        outputs = hf[DataKeys.MUT_MOTIF_LOGITS][:] # {N, mutM_combos, logit}
        sig_pwms_names = hf[DataKeys.MUT_MOTIF_LOGITS].attrs[AttrKeys.PWM_NAMES]

    # clean up names
    sig_pwms_names = [
        re.sub(r"HCLUST-\d+_", "", pwm_name)
        for pwm_name in sig_pwms_names]
    sig_pwms_names = [
        re.sub(r".UNK.0.A", "", pwm_name)
        for pwm_name in sig_pwms_names]
    
    # set up combination matrix
    num_mut_motifs = len(sig_pwms_names)
    combinations = setup_combinations(num_mut_motifs)
    combinations = 1 - combinations
    
    # go through calculations
    results = np.zeros((outputs.shape[0], len(args.calculations), outputs.shape[2]))
    labels = []
    for i in xrange(len(args.calculations)):

        # extract foreground idx
        foreground = args.calculations[i][0]
        foreground_idx = np.where(
            (np.transpose(combinations) == foreground).all(axis=1))[0][0]

        # for logging
        foreground_strings = _setup_output_string(sig_pwms_names, foreground)
        #print "foreground:", " ".join(foreground_strings)
            
        # extract background idx
        background = args.calculations[i][1]
        background_idx = np.where(
            (np.transpose(combinations) == background).all(axis=1))[0][0]

        # for logging
        background_strings = _setup_output_string(sig_pwms_names, background)
        #print "background:", " ".join(background_strings)

        # and adjust to conditional string
        conditional_string = _make_conditional_string(foreground_strings, background_strings)
        labels.append(conditional_string)
        
        # log scale, so subtract
        results[:,i] = outputs[:,foreground_idx] - outputs[:,background_idx]
        
        # convert out of log?
        #results[:,i] = np.power(2, outputs[:,foreground_idx] - outputs[:,background_idx])
        
        # print mean result?
        print np.mean(np.power(2, results[:,i]), axis=0)

    # TODO calculate all sig levels?
    for i in xrange(results.shape[1]):
        for j in xrange(results.shape[1]):
            if i >= j:
                continue
            
            # calculate sig
            print "{} vs {}".format(labels[i], labels[j])
            delta_results = results[:,i] - results[:,j]
            pvals = run_delta_permutation_test(delta_results)
            print pvals
    
    # save out into h5 file
    # TODO consider saving out under new keys each time
    with h5py.File(args.synergy_file, "a") as hf:
        if hf.get(DataKeys.SYNERGY_SCORES) is not None:
            del hf[DataKeys.SYNERGY_SCORES]
        hf.create_dataset(DataKeys.SYNERGY_SCORES, data=results)
        hf[DataKeys.SYNERGY_SCORES].attrs[AttrKeys.PLOT_LABELS] = labels

    # and plot
    
    
    
    return None


if __name__ == "__main__":
    main()


