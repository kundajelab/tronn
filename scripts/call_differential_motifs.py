#!/usr/bin/env python

"""
description: script to call differential motifs
between a background set and foreground set
"""

import argparse


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="call differential motifs")

    # required args
    parser.add_argument(
        "--foreground_data_files", nargs="+",
        required=True,
        help="data files for foreground calculations")
    parser.add_argument(
        "--background_data_files", nargs="+",
        required=True,
        help="data files for background calculations")
    parser.add_argument(
        "--foregrounds", nargs="+",
        required=True,
        help="foregrounds to test")
    parser.add_argument(
        "--background",
        help="background set (single background for ALL foregrounds)")
    
    # outputs
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="outputs directory")
    parser.add_argument(
        "--out_file", default="grammars.merged.gml",
        help="out file name")
    
    args = parser.parse_args()

    
    return args


def track_runs(args):
    """track command and github commit
    """
    # keeps track of restores (or different commands) in folder
    subcommand_name = "annotate_grammars"
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


def main():


    return



if __name__ == "__main__":
    main()
