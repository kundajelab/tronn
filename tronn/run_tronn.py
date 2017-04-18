# test main script for running tronn package

import learning
import models
import datalayer
import interpretation

import sys
import os
import subprocess
import argparse
import glob

import tensorflow as tf

def parse_args():
    '''
    Setup arguments
    '''

    parser = argparse.ArgumentParser(description='Run TRoNN')

    parser.add_argument('--dataset', help='hdf5 file [encode, ggr]', required=True)
    parser.add_argument('--expt_dir', default='expts', help='path to save model')
    parser.add_argument('--out_dir', help='path to save model')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--tasks', nargs='+', default=[], type=int, help='tasks over which to train multitask model on')

    parser.add_argument('--restore', help='restore from last checkpoint')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model')
    parser.add_argument('--interpret', action='store_true', help='run interpretation tools')
    parser.add_argument('--metric', default='mean_auPRC', type=str, help='metric to use for early stopping')
    parser.add_argument('--patience', default=2, type=int, help='metric to use for early stopping')

    parser.add_argument('--model', nargs='+', help='choose model and provide configs', required=True)

    args = parser.parse_args()

    #parse model configs
    model_config = {}
    model_config['name'] = args.model[0]
    for model_arg in args.model[1:]:
        if '=' in model_arg:
            name, value = model_arg.split('=', 1)
            model_config[name] = eval(value)
        else:
            model_config[model_arg] = True
    args.model = model_config

    #set out_dir
    if args.restore:
        out_dir = args.restore
    else:
        if args.dataset == 'ggr':
            out_dir = '%s/ggr,tasks%s,model%s' % (args.expt_dir, ''.join(map(str, sorted(args.tasks))), ','.join(['%s%s'%(k, v) for k,v in sorted(args.model.items())]))
        elif args.dataset == 'encode':
            out_dir = '%s/encode,model%s' % (args.expt_dir, ','.join(['%s%s'%(k, v) for k,v in sorted(args.model.items())]))
        else:
            raise
        if args.out_dir:
            out_dir = '%s,%s' % (out_dir, args.out_dir)

        num_similar_expts = len(glob.glob('%s*'%out_dir))
        if num_similar_expts>0:
            out_dir += '_%d' % num_similar_expts
    out_dir = "".join(c if c not in "[]()" else '' for c in out_dir.replace(' ', ''))
    args.out_dir = out_dir
    print 'out_dir: %s' % args.out_dir
    print 'model args: %s' % args.model
    return args

def main():
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    num_restores = len(glob.glob(os.path.join(args.out_dir, 'command')))
    with open(os.path.join(args.out_dir, 'command%d.txt'%num_restores), 'w') as f:
        git_checkpoint_label = subprocess.check_output(["git", "describe", "--always"])
        f.write(git_checkpoint_label+'\n')
        f.write(' '.join(sys.argv)+'\n')

    # TODO fix input of info to make easier to run
    if args.dataset == 'ggr':
        DATA_DIR = '/mnt/lab_data/kundaje/users/dskim89/ggr/chromatin/data/nn.atac.idr_regions.2016-11-30.hdf5/h5'
    else:
        DATA_DIR = '/srv/scratch/shared/indra/dskim89/ggr/sequence_model.nn.2017-03-27.roadmap_encode_pretrain_tmp/data/h5'
    data_files = glob.glob('{}/*.h5'.format(DATA_DIR))
    print 'Found {} chrom files'.format(len(data_files))
    train_files = data_files[0:15]
    valid_files = data_files[15:20]

    if args.train:

        # This all needs to be cleaned up into some kind of init function...
        num_train_examples = datalayer.get_total_num_examples(train_files)
        train_steps = num_train_examples / args.batch_size - 100
        print train_steps
        #train_steps = 10000 # for now. just to make it easier to test
        print 'Num train examples: {}'.format(num_train_examples)

        num_valid_examples = datalayer.get_total_num_examples(valid_files)
        valid_steps = num_valid_examples / args.batch_size - 100
        print 'Num valid examples: {}'.format(num_valid_examples)

        # Should epoch level be where things are exposed here? or is this loop abstractable too?
        metric_best = None
        consecutive_bad_epochs = 0
        for epoch in xrange(args.epochs):
            print "EPOCH:", str(epoch)
            restore = args.restore is not None
            if epoch > 0:
                restore = True
            
            if restore:
                curr_step = checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.out_dir))
                curr_step = int(checkpoint_path.split('-')[1].split('.')[0])
                target_step = curr_step + train_steps
            else:
                target_step = train_steps

            # Run training
            learning.train(datalayer.load_data_from_filename_list, 
                models.models[args.model['name']],
                tf.nn.sigmoid,
                tf.losses.sigmoid_cross_entropy,
                #tf.train.AdamOptimizer,{'learning_rate': 0.001, 'beta1':0.9, 'beta2':0.999},
                tf.train.RMSPropOptimizer,{'learning_rate': 0.001, 'decay':0.98, 'momentum':0.0},
                restore,
                'Not yet implemented',
                args,
                train_files,
                '{}/train'.format(args.out_dir),
                target_step)

            # Get last checkpoint
            checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.out_dir)) # fix this to save checkpoints elsewhere

            # Evaluate after training
            eval_metrics = learning.evaluate(datalayer.load_data_from_filename_list,
                models.models[args.model['name']],
                tf.nn.sigmoid,
                tf.losses.sigmoid_cross_entropy,
                checkpoint_path,
                args,
                valid_files,
                '{}/valid'.format(args.out_dir),
                num_evals=valid_steps)
            if metric_best is None or ('loss' in args.metric) != (eval_metrics[args.metric]>metric_best):
                consecutive_bad_epochs = 0
                metric_best = eval_metrics[args.metric]
                with open(os.path.join(args.out_dir, 'best.txt'), 'w') as f:
                    f.write('epoch %d\n'%epoch)
                    f.write(str(eval_metrics))
            else:
                consecutive_bad_epochs += 1
                if consecutive_bad_epochs>args.patience:
                    print 'early stopping triggered'
                    break

    # extract importance
    if args.interpret:

        # Look at keratin loci for now
        data_files = ['{}/skin_atac_idr_chr12.h5'.format(DATA_DIR)]

        # checkpoint file
        checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.out_dir))
        print checkpoint_path

        interpretation.interpret(datalayer.load_data_from_filename_list,
            data_files,
            models.models[args.model['name']],
            tf.losses.sigmoid_cross_entropy,
            checkpoint_path,
            args,
            'importances.h5')

    # run test set when needed for final evaluation

if __name__ == '__main__':
    main()