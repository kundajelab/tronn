"""Run multiple experiments in parallel with different parameters."""

import random
import subprocess
import os
import sys
import argparse

MODEL_DIR = 'expts'

arg2options = {
    'epochs': [20],
    'batch_size': [128],
    'metric': ['mean_auPRC'],
    'patience': [2],
    'days': [0],
    'model': ['danq']
}

danq_arg2options = {
    'filters': [80, 160, 320],
    'kernel': [26],
    'rnn_layers': [1],
    'rnn_units': [80, 160, 320],
    'fc_layers': [1],
    'fc_units': [320, 925],
    'conv_drop': [0.0, 0.2],
    'rnn_drop': [0.0, 0.3, 0.5],
    'flag1': [None]
}

model = 'danq'
model_arg2options = danq_arg2options


def sample(arg2options):
    arg_values = {}
    for arg, options in arg2options.iteritems():
        arg_values[arg] = random.choice(options)
    # if arg2options is danq_arg2options:
    #     arg_values['rnn_units'] = arg_values['filters']
    #     arg_values['conv_drop'] = 0.2 if (arg_values['conv_drop'] == 0.5) else 0.0
    #     arg_values['fc_units'] = 925 if (arg_values['filters'] == 320) else 320
    return arg_values



def format_args(arg_values, prefix=''):
    string = ''
    for arg, value in sorted(arg_values.items()):
        if 'flag' not in arg:
            string += ' %s%s=%s' % (prefix, arg, value)
        elif value is not None:
            string += ' %s%s' % (prefix, value)
    return string.strip()

old_expts = set()
def get_cmd(allow_repeats=False):
    arg_values = sample(arg2options)
    model_arg_values = sample(model_arg2options)
    out_dir = dirname_for_expt(arg_values, model_arg_values)
    if out_dir in old_expts or os.path.exists(out_dir):
        return None
    old_expts.add(out_dir)
    # if os.path.exists(out_dir):
    #     if allow_repeats:
    #         out_dir += len(filter(lambda dir: dir.starts_with(out_dir), os.listdir()))
    #     else:
    #         return None
    cmd = '--out_dir=%s %s --model %s %s' % (
                out_dir,
                format_args(arg_values, prefix='--'),
                model,
                format_args(model_arg_values))
    return cmd


def dirname_for_expt(arg_values, model_arg_values):
    dirname = model + ',' \
        + ','.join(['%s%s' % (arg, value)
                    for (arg, value) in
                        (sorted(arg_values.items()) +
                            sorted(model_arg_values.items()))])
    dirname = os.path.join(MODEL_DIR, dirname)
    return dirname

def random_launcher(gpu):
    for attempt in xrange(100000):
        cmd = get_cmd()
        if cmd is None: continue
        cmd = 'CUDA_VISIBLE_DEVICES=%d python run_tronn.py --train %s' % (gpu, cmd)
        print cmd
        subprocess.call(cmd, shell=True)


def launch_cmds_in_file(gpu, cmd_file):
    with open(cmd_file) as f:
        for cmd in f:
            print cmd
            cmd = 'CUDA_VISIBLE_DEVICES=%d python run_tronn.py --train %s' % (gpu, cmd)
            subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TRoNN')
    parser.add_argument('--gpu', type=int, required=True, help='gpu on which to launch jobs')
    parser.add_argument('--dir', default='expts', help='gpu on which to launch jobs')
    parser.add_argument('--cmd_file', help='file from which to read commands used to launch jobs')
    args = parser.parse_args()
    MODEL_DIR = args.dir
    if args.cmd_file:
        launch_cmds_in_file(args.gpu, args.cmd_file)
    else:
        random_launcher(args.gpu)
