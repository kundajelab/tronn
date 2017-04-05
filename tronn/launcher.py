"""Automatically schedule gpu experiments on file"""

import subprocess
import argparse

def launch_cmds_in_file(gpu, cmd_file):
    with open(cmd_file) as f:
        for cmd in f:
            cmd = 'CUDA_VISIBLE_DEVICES=%d %s' % (gpu, cmd)
            print cmd
            subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRoNN scheduler')
    parser.add_argument('--gpu', type=int, required=True, help='gpu on which to launch jobs')
    parser.add_argument('--cmd_file', help='file from which to read commands used to launch jobs')
    args = parser.parse_args()
    launch_cmds_in_file(args.gpu, args.cmd_file)
