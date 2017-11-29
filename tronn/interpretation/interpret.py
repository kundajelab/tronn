"""Description: Contains methods and routines for interpreting models
"""

import os
import glob
import json

import tensorflow as tf

from tronn.datalayer import load_data_from_filename_list
from tronn.models import models
from tronn.interpretation.importances import generate_importance_scores
from tronn.interpretation.importances import visualize_sample_sequences
from tronn.interpretation.motifs import bootstrap_fdr
from tronn.interpretation.motifs import run_pwm_convolution
from tronn.interpretation.motifs import extract_positives_from_motif_mat


def interpret(
        tronn_graph,
        model_checkpoint,
        h5_file,
        sample_size=None,
        pwm_list=None,
        method="input_x_grad",
        keep_negatives=False,
        h5_batch_size=128):
    """Set up a graph and run inference stack
    """
    with tf.Graph().as_default() as g:

        # build graph
        if method == "input_x_grad":
            print "using input_x_grad"
            outputs = tronn_graph.build_inference_graph(pwm_list=pwm_list, normalize=True)
        elif method == "guided_backprop":
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                print "using guided backprop"
                outputs = tronn_graph.build_inference_graph_v3(pwm_list=pwm_list, normalize=True)
            
        # set up session
        sess, coord, threads = setup_tensorflow_session()

        # restore
        init_assign_op, init_feed_dict = restore_variables_op(
            model_checkpoint, skip=["pwm"])
        sess.run(init_assign_op, init_feed_dict)
        
        # set up hdf5 file to store outputs
        with h5py.File(h5_file, 'w') as hf:

            h5_handler = H5Handler(
                hf, outputs, sample_size, resizable=True, batch_size=4096)

            # set up outlayer
            example_generator = ExampleGenerator(
                sess,
                outputs,
                64, # Fix this later
                reconstruct_regions=False,
                keep_negatives=keep_negatives,
                filter_by_prediction=True,
                filter_tasks=tronn_graph.importances_tasks)

            # run all samples unless sample size is defined
            try:
                total_examples = 0
                while not coord.should_stop():
                    
                    region, region_arrays = example_generator.run()
                    region_arrays["example_metadata"] = region

                    h5_handler.store_example(region_arrays)
                    total_examples += 1

                    # check condition
                    if (sample_size is not None) and (total_examples >= sample_size):
                        break

            except tf.errors.OutOfRangeError:
                print "Done reading data"
                # add in last of the examples

            finally:
                time.sleep(60)
                h5_handler.flush()
                h5_handler.chomp_datasets()

        # catch the exception ValueError - (only on sherlock, come back to this)
        try:
            close_tensorflow_session(coord, threads)
        except:
            pass

    return None



def interpret_old(
        args,
        data_loader,
        data_files,
        model,
        loss_fn,
        prefix,
        out_dir, 
        task_nums, # manual
        dendro_cutoffs, # manual
        motif_file,
        motif_sim_file,
        motif_offsets_file,
        rna_file,
        rna_conversion_file,
        checkpoint_path,
        scratch_dir='./',
        sample_size=220000):
    """placeholder for now"""

    importances_mat_h5 = '{0}/{1}.importances.h5'.format(scratch_dir, prefix)

    # ---------------------------------------------------
    # generate importance scores across all open sites
    # ---------------------------------------------------

    if not os.path.isfile(importances_mat_h5):
        generate_importance_scores(
            data_loader,
            data_files,
            model,
            loss_fn,
            checkpoint_path,
            args,
            importances_mat_h5,
            guided_backprop=True, 
            method='importances',
            sample_size=sample_size) # TODO change this, it's a larger set than this

    # ---------------------------------------------------
    # for each task, do the following:
    # ---------------------------------------------------

    for task_num_idx in range(len(task_nums)):

        task_num = task_nums[task_num_idx]
        print "Working on task {}".format(task_num)

        if args.plot_importances:
            # visualize a few samples
            sample_seq_dir = 'task_{}.sample_seqs'.format(task_num)
            os.system('mkdir -p {}'.format(sample_seq_dir))
            visualize_sample_sequences(importances_mat_h5, task_num, sample_seq_dir)
            
        # ---------------------------------------------------
        # Run all task-specific importance scores through PWM convolutions
        # IN: sequences x importance scores
        # OUT: sequences x motifs
        # ---------------------------------------------------
        motif_mat_h5 = 'task_{}.motif_mat.h5'.format(task_num)
        if not os.path.isfile(motif_mat_h5):
            run_pwm_convolution(
                data_loader,
                importances_mat_h5,
                motif_mat_h5,
                args.batch_size * 2,
                motif_file,
                task_num)

        # ---------------------------------------------------
        # extract the positives to cluster in R and visualize
        # IN: sequences x motifs
        # OUT: positive sequences x motifs
        # ---------------------------------------------------
        pos_motif_mat = 'task_{}.motif_mat.positives.txt.gz'.format(task_num)
        if not os.path.isfile(pos_motif_mat):
            extract_positives_from_motif_mat(motif_mat_h5, pos_motif_mat, task_num)

        # ---------------------------------------------------
        # Cluster positives in R and output subgroups
        # IN: positive sequences x motifs
        # OUT: subgroups of sequences
        # ---------------------------------------------------
        cluster_dir = 'task_{}.positives.clustered'.format(task_num)
        if not os.path.isdir(cluster_dir):
            os.system('mkdir -p {}'.format(cluster_dir))
            prefix = 'task_{}'.format(task_num)
            os.system('run_region_clustering.R {0} 50 {1} {2}/{3}'.format(pos_motif_mat,
                                                                          dendro_cutoffs[task_num_idx],
                                                                          cluster_dir,
                                                                          prefix))        


        # ---------------------------------------------------
        # Now for each subgroup of sequences, get a grammar back
        # ---------------------------------------------------
        for subgroup_idx in range(dendro_cutoffs[task_num_idx]):
            
            index_group = "{0}/task_{1}.group_{2}.indices.txt.gz".format(cluster_dir, task_num, subgroup_idx+1)
            out_prefix = '{}'.format(index_group.split('.indices')[0])
            
            
            # ---------------------------------------------------
            # Run boostrap FDR to get back significant motifs
            # IN: subgroup x motifs
            # OUT: sig motifs
            # ---------------------------------------------------
            bootstrap_fdr_cutoff_file = '{}.fdr_cutoff.zscore_cutoff.txt'.format(out_prefix)
            if not os.path.isfile(bootstrap_fdr_cutoff_file):
                bootstrap_fdr(motif_mat_h5, out_prefix, task_num, index_group)

            # Filter with RNA evidence
            rna_filtered_file = '{}.rna_filtered.txt'.format(bootstrap_fdr_cutoff_file.split('.txt')[0])
            if not os.path.isfile(rna_filtered_file):
                add_rna_evidence = ("filter_w_rna.R "
                                    "{0} "
                                    "{1} "
                                    "{2} "
                                    "{3}").format(bootstrap_fdr_cutoff_file, rna_conversion_file, rna_file, rna_filtered_file)
                print add_rna_evidence
                os.system(add_rna_evidence)

            # make a network grammar
            # size of node is motif strength, links are motif similarity
            grammar_network_plot = '{}.grammar.network.pdf'.format(rna_filtered_file.split('.txt')[0])
            
            if not os.path.isfile(grammar_network_plot):
                make_network_plot = 'make_network_grammar_v2.R {0} {1} {2}'.format(rna_filtered_file, motif_sim_file, grammar_network_plot)
                print make_network_plot
                os.system(make_network_plot)


            
                
    return None


def run(args):

    # find data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    print 'Found {} chrom files'.format(len(data_files))

    # checkpoint file
    checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.model_dir))
    print checkpoint_path

    # set up scratch_dir
    os.system('mkdir -p {}'.format(args.scratch_dir))

    # load external data files
    with open(args.annotations, 'r') as fp:
        annotation_files = json.load(fp)

    # current manual choices
    task_nums = [0, 9, 10, 14]
    dendro_cutoffs = [7, 6, 7, 7]
    
    interpret(args,
              load_data_from_filename_list,
              data_files,
              models[args.model['name']],
              tf.losses.sigmoid_cross_entropy,
              args.prefix,
              args.out_dir,
              task_nums, 
              dendro_cutoffs, 
              annotation_files["motif_file"],
              annotation_files["motif_sim_file"],
              annotation_files["motif_offsets_file"],
              annotation_files["rna_file"],
              annotation_files["rna_conversion_file"],
              checkpoint_path,
              scratch_dir=args.scratch_dir,
              sample_size=args.sample_size)

    return
