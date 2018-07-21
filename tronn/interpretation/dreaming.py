# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from tronn.datalayer import ArrayDataLoader

from tronn.nets.nets import net_fns

from tronn.util.tf_utils import setup_tensorflow_session
from tronn.util.tf_utils import close_tensorflow_session

from tronn.util.tf_ops import restore_variables_op


"""
the modalities:

1) given original sequences, mutate to optimize logits (ie, sharpen the diff? provide desired output?)
2) given random sequence, mutate to produce desired output

"""


def dream_one_sequence(
        sequence,
        sess,
        feed_dict,
        output_tensors,
        key,
        max_iter=100,
        num_bp_per_iter=10):
    """run dream on one sequence
    """
    for i in xrange(max_iter):

        # new feed dict with updated sequence
        new_feed_dict = {}
        for key in feed_dict.keys():
            if key == "features":
                new_feed_dict[sess.graph.get_tensor_by_name("dataloader/{}:0".format(key))] = sequence
            else:
                new_feed_dict[sess.graph.get_tensor_by_name("dataloader/{}:0".format(key))] = feed_dict[key]
        
        # run model and get gradients and logits
        outputs = sess.run(output_tensors, feed_dict=new_feed_dict)
        gradients = outputs["gradients"]
        #logits = outputs["logits"]

        # calculate step size for updating sequence
        step_size = 1.0 / (gradients.std() + 1e-8)
        
        # only change top k base pairs
        max_grad_by_bp = np.squeeze(np.max(gradients, axis=3))
        best_indices = np.argpartition(max_grad_by_bp, -num_bp_per_iter)[-num_bp_per_iter:]
                    
        # generate sequence based on grad
        new_sequence = np.zeros_like(sequence)
        new_sequence[0, 0, np.arange(sequence.shape[2]), gradients.argmax(axis=3)] = 1
        #print np.sum(np.multiply(sequence[0,0,best_indices,:], new_sequence[0,0,best_indices,:]))
                    
        # replace
        sequence[0,0,best_indices,:] = new_sequence[0,0,best_indices,:]

        # save to outputs also
        outputs[key] = sequence
        
    return outputs



def run_old(args):
    
    # set up a file loader and the input dict
    data_loader_fn = load_data_from_feed_dict
    input_dict = {
        "features:0": sequences[0],
        "labels:0": np.ones((1, 119)),
        "metadata:0": np.array(["random"])}
    
    # set up graph
    tronn_graph = TronnNeuralNetGraph(
        {"data": input_dict}, # this is only necessary here to set up tensor shapes
        [],
        data_loader_fn,
        1,
        net_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        inference_fn=net_fns[args.inference_fn],
        importances_tasks=args.inference_tasks,
        shuffle_data=False)

    # checkpoint file (unless empty net)
    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    elif args.model["name"] == "empty_net":
        checkpoint_path = None
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)

        
    with tf.Graph().as_default() as g:

        # TODO inference params include gradient start points
        # importance task indices
        desired_pattern = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]).astype(np.float32)
        desired_pattern = np.linspace(-3, 3, num=10).astype(np.float32)
        desired_pattern = np.linspace(5, -5, num=10).astype(np.float32)
        #desired_pattern = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float32)
        print desired_pattern.shape
        
        # need to set up a graph where we get back the gradients alone
        output_tensors = tronn_graph.build_inference_graph(
            {"importances_fn": args.backprop,
             "dream": True,
             "dream_pattern": desired_pattern})

        # set up session
        sess, coord, threads = setup_tensorflow_session()

        # restore from checkpoint as needed
        if args.model_checkpoint is not None:
            # TODO - create an restore_variables_ensemble_op
            init_assign_op, init_feed_dict = restore_variables_op(
                args.model_checkpoint, skip=["pwm"])
            sess.run(init_assign_op, init_feed_dict)
        else:
            print "WARNING WARNING WARNING: did not use checkpoint. are you sure?"

        # now go through each sequence
        for sequence in sequences:
            
            orig_sequence = sequence

            # perform optimization runs
            for i in xrange(args.max_iter):
                
                # change the data feed
                input_dict["features:0"] = sequence
                
                # run model and get back gradient and loss value
                outputs = sess.run(
                    output_tensors,
                    feed_dict=input_dict)

                gradients = outputs["gradients"]
                logits = outputs["logits"]
                
                # calculate the step size for updating the sequence
                step_size = 1.0 / (gradients.std() + 1e-8)
                print gradients.std()
                print step_size
                #step_size = 1.0

                if False:
                    # update the sequence by adding the scaled gradient
                    sequence += step_size * gradients
                    
                    # and clip
                    sequence = np.clip(sequence, 0.0, 1.0)
                    
                # re-convert to a onehot sequence
                if False:
                    new_sequence = np.zeros_like(sequence)
                    new_sequence[0, 0, np.arange(sequence.shape[2]), gradients.argmax(axis=3)] = 1
                    print "overlap:", np.sum(np.multiply(new_sequence, sequence))
                    sequence = new_sequence

                if False:
                    # only allow x bp change at a time?
                    best_idx = np.abs(np.squeeze(np.max(gradients, axis=3))).argmax()
                    print best_idx
                    new_sequence = np.array(sequence)
                    new_sequence[0, 0, best_idx, :] = 0
                    new_sequence[0, 0, best_idx, gradients[0,0,best_idx,:].argmax()] = 1
                    print "overlap:", np.sum(np.multiply(new_sequence, sequence))
                    sequence = new_sequence


                if True:
                    k = 100
                    
                    # adjust top 10 positions
                    #abs_grad = np.abs(np.squeeze(np.max(gradients, axis=3)))
                    #best_indices = np.argpartition(abs_grad, -10)[-10:]
                    max_grad_by_bp = np.squeeze(np.max(gradients, axis=3))
                    best_indices = np.argpartition(max_grad_by_bp, -k)[-k:]

                    old_sequence = sequence
                    
                    # generate sequence based on grad
                    new_sequence = np.zeros_like(sequence)
                    new_sequence[0, 0, np.arange(sequence.shape[2]), gradients.argmax(axis=3)] = 1


                    print np.sum(np.multiply(sequence[0,0,best_indices,:], new_sequence[0,0,best_indices,:]))
                    
                    # replace
                    sequence[0,0,best_indices,:] = new_sequence[0,0,best_indices,:]
                    
                    #print "overlap:", np.sum(np.multiply(sequence, old_sequence))
                    

                    
                #import ipdb
                #ipdb.set_trace()
                    
                # print out the logits
                # ideally the logits should be converging on the desired pattern
                print logits[0,args.inference_tasks]
                

            if args.plot:

                # plot out the importance scores on the latest iteration
                pass
                

                
    return None

