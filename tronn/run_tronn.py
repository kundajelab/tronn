# test main script for running tronn package

import tronn
import sys
import os
import subprocess
import argparse
import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim

def parse_args():
    '''
    Setup arguments
    '''

    parser = argparse.ArgumentParser(description='Run TRoNN')

    parser.add_argument('--data_file', help='(currently only) hdf5 file')
    parser.add_argument('--out_dir', default='out', help='path to save model')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    parser.add_argument('--restore', action='store_true', help='restore from last checkpoint')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model')
    parser.add_argument('--interpret', action='store_true', help='run interpretation tools')
    parser.add_argument('--metric', default='mean_auPRC', type=str, help='metric to use for early stopping')
    parser.add_argument('--patience', default=2, type=int, help='metric to use for early stopping')

    parser.add_argument('--model', help='choose model from models.models')
    
    args, unknowns = parser.parse_known_args()
    model_config = {}
    for unk in unknowns:
        if '=' in unk:
            name, value = unk.split('=', 1)
            model_config[name] = eval(value)
        else:
            model_config[unk] = True
    print model_config

    return args, model_config


def main():
    # TODO fix input of info to make easier to run
    DATA_DIR = '/mnt/lab_data/kundaje/users/dskim89/ggr/chromatin/data/nn.atac.idr_regions.2016-11-30.hdf5/h5'
    data_files = glob.glob('{}/*.h5'.format(DATA_DIR))
    print 'Found {} chrom files'.format(len(data_files))

    train_files = data_files[0:15]
    valid_files = data_files[15:20]

    args, model_config = parse_args()

    os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, 'command.txt'), 'w') as f:
        git_checkpoint_label = subprocess.check_output(["git", "describe", "--always"])
        f.write(git_checkpoint_label+'\n')
        f.write(' '.join(sys.argv)+'\n')

    if args.train:

        # This all needs to be cleaned up into some kind of init function...
        num_train_examples = tronn.nn_utils.get_total_num_examples(train_files)
        train_steps = num_train_examples / args.batch_size - 100
        print train_steps
        #train_steps = 10000 # for now. just to make it easier to test
        print 'Num train examples: {}'.format(num_train_examples)

        num_valid_examples = tronn.nn_utils.get_total_num_examples(valid_files)
        valid_steps = num_valid_examples / args.batch_size - 100
        print 'Num valid examples: {}'.format(num_valid_examples)

        # Should epoch level be where things are exposed here? or is this loop abstractable too?
        metric_best = None
        consecutive_bad_epochs = 0
        for epoch in xrange(args.epochs):
            print "EPOCH:", str(epoch)
            restore = args.restore
            if epoch > 0:
                restore = True

            # Run training
            tronn.learning.train(tronn.load_data_from_filename_list, 
                tronn.models.models[args.model],
                model_config,
                tf.nn.sigmoid,
                tf.losses.sigmoid_cross_entropy,
                #tf.train.AdamOptimizer,{'learning_rate': 0.001, 'beta1':0.9, 'beta2':0.999},
                tf.train.RMSPropOptimizer,{'learning_rate': 0.001, 'decay':0.98, 'momentum':0.0},
                restore,
                'Not yet implemented',
                args,
                train_files,
                '{}/train'.format(args.out_dir),
                (epoch+1)*train_steps)

            # Get last checkpoint
            checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.out_dir)) # fix this to save checkpoints elsewhere

            # Evaluate after training
            eval_metrics = tronn.learning.evaluate(tronn.load_data_from_filename_list,
                tronn.models.models[args.model],
                model_config,
                tf.nn.sigmoid,
                tf.losses.sigmoid_cross_entropy,
                tronn.streaming_metrics_tronn,
                checkpoint_path,
                args,
                valid_files,
                '{}/valid'.format(args.out_dir),
                num_evals=valid_steps)
            if metric_best is None or ('loss' in args.metric != eval_metrics[args.metric]>metric_best):
                metric_best = eval_metrics[args.metric]
                with open(os.path.join(args.out_dir, 'best.txt'), 'w') as f:
                    f.write('epoch %d\n'%epoch)
                    f.write(eval_metrics)
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

        tronn.interpretation.interpret(tronn.load_data_from_filename_list,
            data_files,
            tronn.models.models[args.model],
            model_config,
            tf.losses.sigmoid_cross_entropy,
            checkpoint_path,
            args,
            'importances.h5')

    # run test set when needed for final evaluation

if __name__ == '__main__':
    main()