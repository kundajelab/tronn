import os
import sys
import glob
import logging

"""description: common tools for scripts/cmd line
"""


def track_runs(args):
    """track command and github commit
    """
    # keeps track of restores (or different commands) in folder
    num_restores = len(glob.glob('{0}/{1}.command*'.format(args.out_dir, args.subcommand_name)))
    logging_file = '{0}/{1}.command_{2}.log'.format(args.out_dir, args.subcommand_name, num_restores)
    
    # track github commit
    git_repo_path = os.path.dirname(os.path.realpath(__file__))
    os.system('echo "commit:" > {0}'.format(logging_file))
    os.system('git --git-dir={0}/.git rev-parse HEAD >> {1}'.format(
        git_repo_path.split("/bin")[0], logging_file))
    os.system('echo "" >> {0}'.format(logging_file))
    
    # write out the command
    with open(logging_file, 'a') as f:
        f.write(' '.join(sys.argv)+'\n\n')
    
    return logging_file


def setup_run_logs(args):
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
