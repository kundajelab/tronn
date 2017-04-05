"""Automatically schedule gpu experiments on file"""

import subprocess
import argparse
import random

def launch_cmds_in_file(gpu, cmd_file, shuffle):
    with open(cmd_file) as f:
        cmds = f.readlines()
    if shuffle:
        random.shuffle(cmds)
    for cmd in cmds:
        cmd = 'CUDA_VISIBLE_DEVICES=%d %s' % (gpu, cmd)
        print '-'*100
        print cmd
        subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRoNN scheduler')
    parser.add_argument('--gpu', type=int, required=True, help='gpu on which to launch jobs')
    parser.add_argument('--cmd_file', help='file from which to read commands used to launch jobs')
    parser.add_argument('--shuffle', help='shuffle commands in file')
    args = parser.parse_args()
    launch_cmds_in_file(args.gpu, args.cmd_file, args.shuffle)
