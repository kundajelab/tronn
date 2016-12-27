# test main script for running tronn package

import tronn
import argparse
import threading
import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim


def parse_args():
    '''
    Setup arguments
    '''

    parser = argparse.ArgumentParser(description='Run TRoNN')

    parser.add_argument('--data_file', help='(currently only) hdf5 file')
    parser.add_argument('--epochs', default=20, help='number of epochs')
    parser.add_argument('--batch_size', default=128, help='batch size')

    parser.add_argument('--restore', action='store_true', help='restore from last checkpoint')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model')
    
    args = parser.parse_args()
    
    return args


def main():


    # TODO take all this and abstract away?
    OUT_DIR = './log'

    DATA_DIR = '/mnt/lab_data/kundaje/users/dskim89/ggr/chromatin/data/nn.atac.idr_regions.2016-11-30.hdf5/h5'

    data_files = glob.glob('{}/*.h5'.format(DATA_DIR))
    print 'Found {} chrom files'.format(len(data_files))

    train_files = data_files[0:15]
    valid_files = data_files[15:20]

    args = parse_args()

    # This all needs to be cleaned up into some kind of init function...
    num_train_examples, seq_length, num_tasks = tronn.check_dataset_params(train_files)
    train_steps = num_train_examples / args.batch_size - 100
    print train_steps
    print 'Num train examples: {}'.format(num_train_examples)

    num_valid_examples, seq_length, num_tasks = tronn.check_dataset_params(valid_files)
    valid_steps = num_valid_examples / args.batch_size - 100
    print 'Num valid examples: {}'.format(num_valid_examples)

    # Should epoch level be where things are exposed here? or is this loop abstractable too?
    for epoch in xrange(args.epochs):
        print "EPOCH:", str(epoch)

        if epoch == 0:
            restore = False
        else:
            restore = True

        # Run training
        tronn.train(tronn.load_data_from_filename_list, 
            tronn.basset_like,
            slim.losses.sigmoid_cross_entropy,
            tf.train.RMSPropOptimizer,
            {'learning_rate': 0.002, 'decay':0.98, 'momentum':0.0, 'epsilon':1e-8},
            tronn.streaming_metrics_tronn,
            restore,
            'Not yet implemented',
            args,
            train_files,
            '{}/train'.format(OUT_DIR),
            (epoch+1)*750)

        # Get last checkpoint
        checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(OUT_DIR)) # fix this to save checkpoints elsewhere
        print checkpoint_path

        # Evaluate after training
        tronn.evaluate(tronn.load_data_from_filename_list,
            tronn.basset_like,
            tronn.streaming_metrics_tronn,
            checkpoint_path,
            args,
            valid_files,
            '{}/valid'.format(OUT_DIR))

    # extract importance
    with tf.Graph.as_default() as interpretation_graph:

        # Here, reinstantiate the graph and get useful things
        pass

    # run test set when needed for final evaluation


    return None

main()
