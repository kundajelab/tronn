import os
import sys
import glob
import logging

import numpy as np

"""description: common tools for scripts/cmd line
"""


def track_runs(args, prefix):
    """track command and github commit
    """
    # keeps track of restores (or different commands) in folder
    num_restores = len(glob.glob('{0}/{1}.command*'.format(args.out_dir, prefix)))
    logging_file = '{0}/{1}.command_{2}.log'.format(args.out_dir, prefix, num_restores)
    
    # track github commit
    git_repo_path = os.path.dirname(os.path.realpath(__file__))
    os.system('echo "commit:" > {0}'.format(logging_file))
    os.system('git --git-dir={0}/.git rev-parse HEAD >> {1}'.format(
        git_repo_path.split("/util")[0], logging_file))
    os.system('echo "" >> {0}'.format(logging_file))
    
    # write out the command
    with open(logging_file, 'a') as f:
        f.write(' '.join(sys.argv)+'\n\n')
    
    return logging_file


def setup_run_logs(args, prefix):
    """set up logging
    """
    logging_file = track_runs(args, prefix)
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


# TODO provide option to load in as json
def parse_multi_target_selection_strings(key_strings):
    """given an arg string list, parse out into a dict
    assumes a format of: key=indices,key=indices::param=val,param=val
    
    this is for smart indexing into tensor dicts (mostly label sets).

    Returns:
      list of tuples, where tuple is (list of tuples of keys and indices, param dict)
    """
    if key_strings is None:
        return []
    
    parsed_results = []
    for key_string in key_strings:
        targets_and_params = key_string.split("::")
        assert len(targets_and_params) <= 2
        
        # set up keys/indices
        targets = targets_and_params[0].split(":")
        parsed_targets = []
        for target in targets:
            target = target.split("=")
            if len(target) > 1:
                indices = [int(i) for i in target[1].split(",")]
            else:
                indices = []
            parsed_targets.append((target[0], indices))

        # set up params
        if len(targets_and_params) == 2:
            params = targets_and_params[1].split(",")
            parsed_params = {}
            for param in params:
                param = param.split("=")
                parsed_params[param[0]] = param[1]
        else:
            parsed_params = {}
            
        # save out
        parsed_results.append((parsed_targets, parsed_params))
    
    return parsed_results



def load_selected_targets(data_loader, targets, params):
    """
    """
    # get targets and concatenate
    selected_targets = []
    for target_key, target_indices in targets:
        target_data = data_loader.load_dataset(target_key)
        if len(target_indices) > 0:
            target_data = target_data[:,target_indices]
        selected_targets.append(target_data)
    selected_targets = np.concatenate(selected_targets, axis=1) # {N, target}

    # reduce
    reduce_type = params.get("reduce_type", "any")
    if reduce_type == "any":
        reduced_targets = np.any(selected_targets, axis=1, keepdims=True)
    elif reduce_type == "all":
        reduced_targets = np.all(selected_targets, axis=1, keepdims=True)
    else:
        reduced_targets = selected_targets
        
    return reduced_targets
