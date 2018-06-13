"""Description: Contains functions/tools for analyzing weighted kmers
extracted from importance weighted sequences.
"""

import os
import glob
import h5py
import json
import scipy.stats

import phenograph

import numpy as np
import pandas as pd
import tensorflow as tf

from tronn.preprocess import one_hot_encode
from tronn.visualization import plot_weights
from tronn.interpretation.kmers import kmer_array_to_hash
from tronn.interpretation.kmers import kmer_hash_to_string
from tronn.interpretation.kmers import kmer_array_to_string
from tronn.interpretation.motifs import extract_positives_from_motif_mat


def kmerize(importances_h5, importances_key, task_num, kmer_lens=[6, 8, 10], num_bases=5):
    """Convert weighted sequence into weighted kmers

    Args:
      importances_h5: file with importances h5 file

    Returns:
      matrix of kmer scores
    """

    # determine total cols of matrix
    total_cols = 0
    for kmer_len in kmer_lens:
        total_cols += num_bases**kmer_len
    print total_cols
        
    # determine number of sequences and generate sparse matrix
    with h5py.File(importances_h5, 'r') as hf:
        num_pos_examples = np.sum(hf['labels'][:,task_num] > 0)
        num_examples = hf['regions'].shape[0]

        pos_indices = np.where(hf['labels'][:,task_num] > 0)

    wkm_mat = np.zeros((num_pos_examples, total_cols))
    onehot_wkm_mat = np.zeros((num_pos_examples, total_cols, 4, max(kmer_lens)))
    
    # now start reading into file
    with h5py.File(importances_h5, 'r') as hf:

        # for each kmer:
        for kmer_len_idx in range(len(kmer_lens)):
            print 'kmer len', kmer_lens[kmer_len_idx]
            kmer_len = kmer_lens[kmer_len_idx]

            min_pos = kmer_len - 2
            
            if kmer_len_idx == 0:
                start_idx = 0
            else:
                start_idx = kmer_len_idx * (num_bases**kmer_lens[kmer_len_idx-1])

            current_idx = 0
            for example_idx in pos_indices[0]:

                if current_idx % 500 == 0:
                    print current_idx

                # go through the sequence
                sequence = hf[importances_key][example_idx,:,:]

                for i in range(sequence.shape[1] - kmer_len):
                    weighted_kmer = sequence[:,i:(i+kmer_len)]
                    kmer = (weighted_kmer > 0).astype(int)

                    if np.sum(kmer) < min_pos:
                        continue

                    kmer_idx = kmer_array_to_hash(kmer)
                    wkm_score = np.sum(weighted_kmer) # bp adjusted score?
                    wkm_mat[current_idx, start_idx+kmer_idx] += wkm_score

                    # TODO - adjust kmer as needed
                    # TODO consider adjusting for base pair importances - ex CNNGT for p63 is only two bp of importance
                    onehot_wkm_mat[current_idx, start_idx+kmer_idx,:,0:kmer_len] += weighted_kmer
                    
                current_idx += 1

    onehot_wkm_avg = np.mean(onehot_wkm_mat, axis=0)

    return wkm_mat, onehot_wkm_avg


def select_kmers(wkm_mat, kmer_len, prefix, top_k=200):
    """Keep the kmers with the highest scores, up to top_k total.
    """
    # using top k kmer mode for now
    wkm_colsums = np.sum(wkm_mat, axis=0)
    kmer_indices = np.arange(wkm_mat.shape[1])[
        np.argpartition(wkm_colsums, -top_k)[-top_k:]]
    kmer_strings = [kmer_hash_to_string(kmer_idx, kmer_len=kmer_len)
                    for kmer_idx in kmer_indices]
    reduced_wkm_mat = wkm_mat[:,np.argpartition(wkm_colsums, -top_k)[-top_k:]]
    
    # make pandas dataframe
    wkm_df = pd.DataFrame(data=reduced_wkm_mat, columns=kmer_strings)
    wkm_df.to_csv("{}.txt".format(prefix), sep='\t')
    
    return wkm_df


def agglom_motifs(pwm_list, cut_fract, debug=False):
    """go through hierarchical until no more combinations
    also set up score tracking to choose when to stop merging
    """
    # set up
    motif_list = list(pwm_list)
    all_motif_lists = []
    scores = np.zeros((len(motif_list)))
    score_idx = 0

    print "agglomerating..."
    while True:

        motif_list = [motif.chomp() for motif in motif_list]

        if debug:
            print [kmer_array_to_string(motif.normalize(in_place=False).weights) for motif in motif_list]

        # end if agglomerated all motifs
        if len(motif_list) == 1:
            break

        # set up current agglom run
        num_motifs = len(motif_list)
        xcor_mat = np.zeros((num_motifs, num_motifs))
        
        # calculate every pair of xcor
        for i in range(len(motif_list)):
            for j in range(len(motif_list)):
                score, offset = motif_list[i].xcor(motif_list[j])
                xcor_mat[i,j] = score

        # take the best xcor (above cutoff) and merge
        np.fill_diagonal(xcor_mat, 0)
        pwm1_idx, pwm2_idx = np.unravel_index(np.argmax(xcor_mat), dims=xcor_mat.shape)
        #score, offset = xcor_pwms(motif_list[pwm1_idx], motif_list[pwm2_idx])
        
        score, offset = motif_list[pwm1_idx].xcor(motif_list[pwm2_idx])
        
        #print "chose", kmer_array_to_string(motif_list[pwm1_idx].weights), kmer_array_to_string(motif_list[pwm2_idx].weights)
        #print score, offset
        merged_pwm = motif_list[pwm1_idx].merge(motif_list[pwm2_idx], offset)
        
        # add new merged pwm, remove old non-merged pwms
        motif_list.append(merged_pwm)
        del motif_list[pwm1_idx]
        del motif_list[pwm2_idx]

        # track scores across agglom iterations
        scores[score_idx] = score
        all_motif_lists.append(list(motif_list))
        
        score_idx += 1

    scores_fract = scores
    stop_idx = np.argmin(scores_fract > cut_fract) - 2

    return all_motif_lists[stop_idx]
    

def make_motif_sets(clustered_df, wkm_array, prefix, cut_fract=0.7):
    """Given clusters of kmers, make motifs
    """
    # clean up communities and data
    data = clustered_df
    communities = list(set(data['community']))
    if (-1 in communities):
        communities.remove(-1) # remove ones not in a community

    # for each community, get kmers in community and hierarchically agglomerate
    community_motif_sets = []
    for community in communities:
        print "community:", community
        
        # get the kmers
        community_df = data.loc[data['community'] == community]
        kmers = community_df.index.tolist() # sort to make deterministic (for now)
        if 'Unnamed: 0' in kmers:
            kmers.remove('Unnamed: 0')
        kmers_scores = community_df.sum(axis=1).tolist()
        
        #motif_list = [kmers_scores[i] * np.squeeze(one_hot_encode(kmers[i])).transpose(1,0) for i in range(len(kmers))]
        # motifs: you want to take the kmer PWMs (as identified by the NN) but also weight by number of sequences
        motif_weights_list = [wkm_array[
            kmer_to_idx(
                np.squeeze(
                    one_hot_encode(kmers[i])).transpose(1,0)
            ),:,:] for i in range(len(kmers))]
        #motif_list = [kmers_scores[i] * wkm_array[kmer_indices[i],:,:] for i in range(len(kmers))]
        # and sort
        motif_list = [PWM(weights) for weights in motif_weights_list]
        kmers_sort_indices = np.argsort([kmer_to_string2(motif.weights) for motif in motif_list])
        motif_list_sorted = [motif_list[i] for i in kmers_sort_indices] # sort for deterministic behavior

        print "kmers: ", kmers

        # agglom
        motifs = agglom_motifs(motif_list_sorted, cut_fract)
        
        normalized_motifs = [normalize_pwm(motif) for motif in motifs]
        motif_strings = [kmer_to_string(motif) for motif in normalized_motifs]
        print "motifs ({0}): {1}".format(len(motifs), motif_strings)

        # for each motif, normalize and plot out
        for motif_idx in range(len(motifs)):
            motif = normalized_motifs[motif_idx]
            plot_weights(motif, '{0}.community{1}.motif{2}.png'.format(prefix, community, motif_idx), figsize=(motif.shape[1],2))

        community_motif_sets.append(motifs)

    # Save out to PWM file
    flat_motif_list = [motif for community_list in community_motif_sets for motif in community_list]

    flat_motif_ordered_indices = np.argsort([kmer_to_string2(motif) for motif in flat_motif_list]) # sort to make agglom deterministic
    flat_motifs_ordered = [flat_motif_list[i] for i in flat_motif_ordered_indices]
    print [kmer_to_string2(motif) for motif in flat_motifs_ordered]
    master_motifs = agglom_motifs(flat_motifs_ordered, cut_fract=0.8)

    # write out to PWM file
    # TODO put in optional MEME tool to name the motifs by closest hit(s)
    for motif_idx in range(len(master_motifs)):
        motif_name = '{0}.motif_{1}'.format(prefix, motif_idx)
        write_pwm('{}.motif_file.txt'.format(prefix), master_motifs[motif_idx], motif_name)

    # and plot it out so you have a representation of the motif
    normalized_master_motifs = [normalize_pwm(motif) for motif in master_motifs]
    print "master_list:", [kmer_to_string2(motif) for motif in normalized_master_motifs]
    for motif_idx in range(len(master_motifs)):
        motif = normalized_master_motifs[motif_idx]
        plot_weights(motif, '{0}.master.motif{1}.png'.format(prefix, motif_idx), figsize=(motif.shape[1],2))
    
    return None


def get_sequence_communities(text_mat_file, prefix):
    """Cluster sequences by which motifs they have. Uses phenograph - this is good
    because it has a way of ignoring things that don't really belong in clusters
    """
    data = pd.read_table(text_mat_file, index_col=0)

    print data.columns
        
    if 'Unnamed: 0' in data.columns:
        del data['Unnamed: 0']

    # TODO change this later to keep locations
    del data['indices']

    # normalize first
    data_norm = data.apply(scipy.stats.zscore, axis=1)
    
    data_npy = data_norm.as_matrix()
    communities, graph, Q = phenograph.cluster(data_npy)
    data_norm['community'] = communities

    # sort by community
    data_sorted = data_norm.sort_values('community')

    # For each community, save out significant motifs (ie, greater than 1 stdev?)
    communities = list(np.unique(communities))

    # TODO save out motif grammar lists
    grammar_file = '{}.grammars.txt'.format(prefix)
    with open(grammar_file, 'w') as out:
        for community in communities:
            community_data = data_sorted.loc[data_sorted['community'] == community]
            print community_data.shape
            
            del community_data['community']
            community_motif_avg = community_data.mean(axis=0)
            community_motifs = community_data.loc[:,community_motif_avg > 0.5] #TODO move param, actually calc enrichment here

            print community_motifs.columns.tolist()
            out.write('{0}.grammar_{1}\t{2}\n'.format(prefix, community,'\t'.join(community_motifs.columns.tolist())))

            # write out BED file
            community_bed = '{0}.community_{1}.bed.gz'.format(prefix, community)
            regions_df = pd.DataFrame(data=community_data.index, columns=['regions'])
            bed_df = regions_df['regions'].str.split(':', 1, expand=True)
            bed_df.columns = ['chr', 'start-stop']
            bed_df['start'], bed_df['stop'] = bed_df['start-stop'].str.split('-', 1).str
            del bed_df['start-stop']
            bed_df.sort_values(['chr', 'start'], inplace=True)
            bed_df.to_csv(community_bed, sep='\t', header=False, index=False, compression='gzip')
            
    print communities
    
    seq_communities_file = '{}.seq_communities.txt'.format(prefix)
    data_sorted.to_csv(seq_communities_file, sep='\t')
    
    return grammar_file, seq_communities_file, communities


# def interpret_wkm(
#         args,
#         data_loader,
#         data_files,
#         model,
#         loss_fn,
#         prefix,
#         out_dir, 
#         task_nums, # manual
#         dendro_cutoffs, # manual
#         motif_file,
#         motif_sim_file,
#         motif_offsets_file,
#         rna_file,
#         rna_conversion_file,
#         checkpoint_path,
#         scratch_dir='./',
#         sample_size=220000,
#         pval=0.05):
#     """placeholder for now"""

#     # ---------------------------------------------------
#     # for each task, do the following:
#     # ---------------------------------------------------

#     for task_num_idx in range(len(task_nums)):

#         task_num = task_nums[task_num_idx]
#         print "Working on task {}".format(task_num)

# 	# ---------------------------------------------------
#         # Convert into weighted kmers
#         # IN: sequences x importance scores
#         # OUT: sequences x importance scores
#         # ---------------------------------------------------
#         wkm_h5 = 'task_{}.wkm.h5'.format(task_num)
#         if not os.path.isfile(wkm_h5):
#             # first convert to wkm
                
#             # and here also make some educated guesses for what PWM this matches (tomtom?)
#             # TODO convert PWM to meme format (see basset code)
#             # also convert HOCOMOCO to meme format
            
#             motif_mat_h5 = 'task_{}.wkm.motif_mat.h5'.format(task_num)
#             if not os.path.isfile(motif_mat_h5):
#                 run_pwm_convolution(
#                     data_loader,
#                     importances_mat_h5,
#                     motif_mat_h5,
#                     args.batch_size * 4,
#                     'task_{}.motif_file.txt'.format(task_num),
#                     task_num)

#             # ---------------------------------------------------
#             # extract the positives to cluster in R and visualize
#             # IN: sequences x motifs
#             # OUT: positive sequences x motifs
#             # ---------------------------------------------------
#             pos_motif_mat = 'task_{}.wkm_mat.positives.txt.gz'.format(task_num)
#             if not os.path.isfile(pos_motif_mat):
#                 extract_positives_from_motif_mat(motif_mat_h5, pos_motif_mat, task_num)

#             # TODO isolate this bit here for now
#             test_dir = 'testing_ism_task0'
#             os.system('mkdir -p {}'.format(test_dir))
                
#             #  phenograph here again for the clustering
#             seq_communities_file = '{0}/task_{1}.seq_communities.txt'.format(test_dir, task_num)
#             grammar_file = '{0}/task_{1}.grammars.txt'.format(test_dir, task_num)
#             if not os.path.isfile(seq_communities_file):
#                 grammar_file, seq_communities_file, communities = get_sequence_communities(pos_motif_mat,
#                                                                                            '{0}/task_{1}'.format(test_dir, task_num))

#             # generate new datasets to run ISM
#             # glob the BED files, generate with preprocess code
#             task_grammars = read_grammar_file(grammar_file)
#             pwms = PWM.get_encode_pwms('task_0.motif_file.txt')
#             pwm_dict = {}
#             for pwm in pwms:
#                 pwm_dict[pwm.name] = pwm
#             community_bed_sets = glob.glob('{}/*.bed.gz'.format(test_dir))
#             for community in range(13): # TODO change this!
#                 community_bed = '{0}/task_{1}.community_{2}.bed.gz'.format(test_dir, task_num, community)
#                 # TODO preprocess data
#                 with open(args.preprocess_annotations, 'r') as fp:
#                     annotation_files = json.load(fp)

#                 community_data_dir = '{}/data'.format(test_dir)
#                 if not os.path.isdir(community_data_dir):
#                     generate_nn_dataset(community_bed,
#                                         annotation_files['univ_dhs'],
#                                         annotation_files['ref_fasta'],
#                                         [community_bed],
#                                         community_data_dir,
#                                         'task_0.community_{}'.format(community),
#                                         parallel=12,
#                                         neg_region_num=0)
#                 data_files = glob.glob('{}/h5/*.h5'.format(community_data_dir))
                
                
#                 # read in grammar sets
#                 grammar = task_grammars[int(community)]
#                 print grammar
#                 num_motifs = len(grammar)
#                 # for each pair of motifs, read in and run model
#                 synergies = np.zeros((num_motifs, num_motifs))
#                 indiv_motif_coeffs = np.zeros((num_motifs, num_motifs))
#                 for motif1_idx in range(num_motifs):
#                     for motif2_idx in range(num_motifs):
#                         if motif1_idx >= motif2_idx:
#                             continue
#                         # here, run ISM tests
#                         print motif1_idx, motif2_idx
#                         pwm1 = pwm_dict[grammar[motif1_idx]]
#                         pwm2 = pwm_dict[grammar[motif2_idx]]

#                         # here run ISM and get out multiplier info
#                         synergy_score, pwm1_score, pwm2_score = run_ism_for_motif_pairwise_dependency(data_files,
#                                                                                                       model,
#                                                                                                       args.model,
#                                                                                                       checkpoint_path,
#                                                                                                       pwm1,
#                                                                                                       pwm2) 
#                         indiv_motif_coeffs[motif1_idx, motif2_idx] = pwm1_score
#                         indiv_motif_coeffs[motif2_idx, motif1_idx] = pwm2_score
#                         synergies[motif1_idx, motif2_idx] = synergy_score


#                 # TODO(dk) get average by ROWs for the individual motif coeffs, write out to file
#                 indiv_avg_scores = np.zeros((1, num_motifs))
#                 for i in range(num_motifs):
#                     indiv_avg_scores[0,i] = np.mean(indiv_motif_coeffs[i,:])
                
#                 with open('{}/task_0.grammar.linear.txt'.format(test_dir), 'w') as fp:

#                     header='# Grammar model: Linear w pairwise interactions\n\n'
#                     fp.write(header)
                    
#                     indiv_motif_coeff_header = 'Non_interacting_coefficients\n'
#                     fp.write(indiv_motif_coeff_header)

#                     indiv_df = pd.DataFrame(data=indiv_avg_scores, columns=grammar)
#                     indiv_df.to_csv(fp, sep='\t')

#                     fp.write("\n")
                    
#                     synergy_motif_coeff_header = 'Pairwise_interacting_coefficients\n'
#                     fp.write(synergy_motif_coeff_header)

#                     synergy_df = pd.DataFrame(data=synergies, index=grammar, columns=grammar)
#                     synergy_df.to_csv(fp, sep='\t')

                    
#                 quit()

#             quit()
            
#             continue
            
            
                
#             # ---------------------------------------------------
#             # Cluster positives in R and output subgroups
#             # IN: positive sequences x motifs
#             # OUT: subgroups of sequences
#             # ---------------------------------------------------
#             cluster_dir = 'task_{}.positives.wkm.clustered'.format(task_num)
#             if not os.path.isdir(cluster_dir):
#                 os.system('mkdir -p {}'.format(cluster_dir))
#                 prefix = 'task_{}'.format(task_num)
#                 os.system('run_region_clustering.R {0} 50 {1} {2}/{3}'.format(pos_motif_mat,
#                                                                               dendro_cutoffs[task_num_idx],
#                                                                               cluster_dir,
#                                                                               prefix))

#     return None


# def run(args):
#     """Run all functions to go from importance scores to de novo motifs
#     """

#     # find data files
#     data_files = glob.glob('{}/*.h5'.format(args.data_dir))
#     print 'Found {} chrom files'.format(len(data_files))

#     # checkpoint file
#     checkpoint_path = tf.train.latest_checkpoint('{}/train'.format(args.model_dir))
#     print checkpoint_path

#     # set up scratch_dir
#     os.system('mkdir -p {}'.format(args.scratch_dir))

#     # load external data files
#     with open(args.annotations, 'r') as fp:
#         annotation_files = json.load(fp)

#     # current manual choices
#     task_nums = [0, 9, 10, 14]
#     dendro_cutoffs = [7, 6, 7, 7]
    
#     interpret_wkm(args,
#               load_data_from_filename_list,
#               data_files,
#               models[args.model['name']],
#               tf.losses.sigmoid_cross_entropy,
#               args.prefix,
#               args.out_dir,
#               task_nums, 
#               dendro_cutoffs, 
#               annotation_files["motif_file"],
#               annotation_files["motif_sim_file"],
#               annotation_files["motif_offsets_file"],
#               annotation_files["rna_file"],
#               annotation_files["rna_conversion_file"],
#               checkpoint_path,
#               scratch_dir=args.scratch_dir,
#               sample_size=args.sample_size)
    
#     return None
