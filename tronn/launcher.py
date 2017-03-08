"""Run multiple experiments in parallel with different parameters."""

import random
import subprocess
import os
import sys

MODEL_DIR = 'models'

arg2options = {
    'epochs': [20],
    'batch_size': [128],
    'metric': ['mean_auPRC'],
    'patience': [2],
    'days': [0],
    'model': ['danq']
}

danq_arg2options = {
    'filters': [320],
    'kernel': [26],
    'rnn_layers': [1],
    'rnn_units': [320],
    'fc_layers': [925],
    'fc_units': [925],
    'conv_drop': [0.2],
    'rnn_drop': [0.5],
    'flag1': [None, 'untied_rnn', 'untied_fc']
}

model = 'danq'
model_arg2options = danq_arg2options


def sample(arg2options):
    arg_values = {}
    for arg, options in arg2options.iteritems():
        arg_values[arg] = random.choice(options)
    return arg_values


def format_args(arg_values, prefix=''):
    string = ''
    for arg, value in sorted(arg_values.items()):
        if 'flag' not in arg:
            string += ' %s%s=%s' % (prefix, arg, value)
        elif value is not None:
            string += ' %s%s' % (prefix, value)
    return string.strip()


def get_cmd(allow_repeats=False):
    arg_values = sample(arg2options)
    model_arg_values = sample(model_arg2options)
    out_dir = dirname_for_expt(arg_values, model_arg_values)
    if os.path.exists(out_dir):
        if allow_repeats:
            out_dir += len(filter(lambda dir: dir.starts_with(out_dir), os.listdir()))
        else:
            return None
    cmd = '%s --out_dir=%s --model %s %s' % (
                format_args(arg_values, prefix='--'),
                out_dir,
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


if __name__ == '__main__':
    gpu = sys.argv[1]
    for attempt in xrange(1000):
        cmd = get_cmd()
        if cmd is None: continue
        cmd = 'CUDA_VISIBLE_DEVICES=%s python run_tronn.py --train %s' % (gpu, cmd)
        print cmd
        subprocess.call(cmd, shell=True)
