# test main script for running tronn package

import tronn
import argparse
import threading
import tensorflow as tf
import tensorflow.contrib.slim as slim


def parse_args():
    '''
    Setup arguments
    '''

    parser = argparse.ArgumentParser(description='Run TRoNN')

    parser.add_argument('data_file', help='(currently only) hdf5 file')
    parser.add_argument('--epochs', default=20, help='number of epochs')
    parser.add_argument('--batch_size', default=128, help='batch size')

    parser.add_argument('--restore', action='store_true', help='restore from last checkpoint')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model')
    
    args = parser.parse_args()
    
    return args


def main():

    OUT_DIR = './log'

    args = parse_args()
    num_train_examples, seq_length, num_tasks = tronn.get_data_params(args.data_file)    

    for epoch in xrange(args.epochs):
        print "EPOCH:", str(epoch)

        if epoch == 0:
            restore = False
        else:
            restore = True

        # Run training
        tronn.train(tronn.load_data, 
            tronn.basset,
            slim.losses.sigmoid_cross_entropy,
            tf.train.RMSPropOptimizer,
            {'learning_rate': 0.002, 'decay':0.98, 'momentum':0.0, 'epsilon':1e-8},
            tronn.streaming_metrics_tronn,
            restore,
            'Not yet implemented',
            args,
            seq_length,
            num_tasks,
            '{}/train'.format(OUT_DIR),
            (epoch+1)*100)

        # Get last checkpoint
        checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(OUT_DIR)) # fix this to save checkpoints elsewhere
        print checkpoint_path

        # Evaluate after training
        tronn.evaluate(tronn.load_data,
            tronn.basset,
            tronn.streaming_metrics_tronn,
            checkpoint_path,
            args,
            seq_length,
            num_tasks,
            '{}/valid'.format(OUT_DIR))

    # extract importance
    with tf.Graph.as_default() as interpretation_graph:

        # Here, reinstantiate the graph and get useful things
        pass

    # run test set when needed for final evaluation


    return None

main()
