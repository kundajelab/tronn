# Description: contains code for seqlets

import h5py
import math

import phenograph

import numpy as np
import pandas as pd

from collections import Counter
from scipy.stats.mstats import zscore


from tronn.outlayer import H5Handler

from tronn.interpretation.kmers import kmer_array_to_hash
from tronn.interpretation.kmers import kmer_hash_to_string
from tronn.interpretation.kmers import kmer_array_to_string

from tronn.interpretation.wkm import agglom_motifs

from tronn.visualization import plot_weights

from tronn.interpretation.motifs import PWM


def extract_seqlets(thresholded_importances_h5_file, tasks, out_h5_file, seq_len=1000, kmer_len=7, stride=1):
    """Given a thresholded importances file, will extract seqlets out to h5 file
    """
    # make resizable array
    min_pos = int(math.ceil(kmer_len / 2.))
    print min_pos
    
    with h5py.File(out_h5_file, "w") as out:
        
        with h5py.File(thresholded_importances_h5_file, 'r') as hf:

            # get importances keys, ordered
            importance_keys = ["importances_task{}".format(task) for task in tasks]

            # get other param
            total_task_num = len(importance_keys)
            
            # get things to skip in creating h5 file
            # and also check seq len
            skip = []
            for key in hf.keys():
                if "importance" in key:
                    skip.append(key)
                    seq_len = hf[key].shape[2]

            example_num = hf["feature_metadata"].shape[0]
            seqlet_max_num = example_num * (seq_len - kmer_len)
            
            h5_handler = H5Handler(
                out,
                hf,
                seqlet_max_num,
                skip=skip,
                is_tensor_input=False)

            # TODO(dk) create dataset for seqlet x task, and seqlet storage
            dataset_shape = (seqlet_max_num, total_task_num)
            h5_handler.add_dataset("seqlet_x_task", dataset_shape, maxshape=dataset_shape)

            dataset_shape = (seqlet_max_num, 4, kmer_len)
            h5_handler.add_dataset("seqlets", dataset_shape, maxshape=dataset_shape)

            # dataset to store kmer id (to easily merge kmers in the end same cluster)
            dataset_shape = (seqlet_max_num, 1)
            h5_handler.add_dataset("kmer_id", dataset_shape, maxshape=dataset_shape)

            current_idx = 0
            tmp_importances_array = np.zeros((total_task_num, 4, seq_len))
            tmp_kmer_array = np.zeros((total_task_num, 4, kmer_len))
            for example_idx in xrange(example_num):
                # go through importances files, get examples from current idx and store in tmp array

                for i in xrange(total_task_num):
                    tmp_importances_array[i,:,:] = hf[importance_keys[i]][example_idx,:,:]
            
                # and then go through and extract kmers of length, by stride, and ignore those that are zero
                # or more than half zeros
                for pos_idx in xrange(seq_len - kmer_len):
                    seqlet = tmp_importances_array[:,:,pos_idx:(pos_idx+kmer_len)]
                    onehot_kmer = (np.sum(seqlet, axis=0) > 0).astype(int)

                    if np.sum(onehot_kmer) < min_pos:
                        continue

                    seqlet_scores = np.sum(seqlet, axis=(1, 2))

                    # make a arrays dict to store with h5_handler
                    array_dict = {}
                    for key in hf.keys():
                        if "feature_metadata" in key:
                            array_dict[key] = hf[key][example_idx,0]
                        elif not "importance" in key:
                            array_dict[key] = hf[key][example_idx,:]

                    array_dict["seqlet_x_task"] = seqlet_scores
                    array_dict["seqlets"] = np.mean(seqlet, axis=0)
                    
                    # also convert kmer to ID representation and store
                    array_dict["kmer_id"] = kmer_array_to_hash(onehot_kmer)

                    h5_handler.store_example(array_dict)
                    current_idx += 1

                    if current_idx % 1000 == 0:
                        print "examples", example_idx
                        print "seqlets", current_idx
    
            h5_handler.flush()
            h5_handler.chomp_datasets()

    return



def reduce_seqlets(seqlet_h5_file, out_h5_file, min_kmer_fract=0.05, top_seqlet_total=100000):
    """Tricks to reduce the seqlet space
    First get kmers with abundance and cluster those

    And ignore low abundance things

    Get a variability score and then keep the top seqlets that are that variable
    """
    # First determine most abundant kmers and cluster those and reduce
    # TODO dont forget to trim
    batch_size = 512
    with h5py.File(seqlet_h5_file, "r") as hf:

        # determine min kmer abundance to chop
        region_counts = Counter(hf["feature_metadata"][:,0])
        total_regions = len(region_counts)
        min_kmer_abundance = min_kmer_fract * total_regions
        print min_kmer_abundance

        # set up kmer hashes
        kmer_hashes = hf["kmer_id"][:,0]
        kmer_hash_counts = Counter(kmer_hashes)
        kmer_hash_to_count = dict(kmer_hash_counts)
        #most_common = kmer_hash_counts.most_common(125)

        # then figure out high variance values
        cv = np.std(hf["seqlet_x_task"], axis=1) / np.mean(hf["seqlet_x_task"], axis=1)
        top_indices = np.argpartition(cv, -top_seqlet_total)[-top_seqlet_total:]
        sorted_top_indices = top_indices[np.argsort(cv[top_indices])]
        cv_cutoff = cv[sorted_top_indices[0]]
        print cv_cutoff

        # and now run through batches and save out into reduced file
        with h5py.File(out_h5_file, 'w') as out:
            h5_handler = H5Handler(
                out,
                hf,
                top_seqlet_total,
                is_tensor_input=False)

            total_batches = int(math.ceil(hf["seqlet_x_task"].shape[0] / float(batch_size)))
            print total_batches
            current_idx = 0
            batch_end = current_idx + batch_size
            
            for batch in xrange(total_batches):
                if batch % 10 == 0:
                    print batch
                
                if batch_end > hf["seqlet_x_task"].shape[0]:
                    batch_end = hf["seqlet_x_task"].shape[0]

                tmp_examples = hf["seqlet_x_task"][current_idx:batch_end,:]

                for example_idx in xrange(current_idx, batch_end):

                    # check conditions
                    if kmer_hash_to_count[hf["kmer_id"][example_idx][0]] < min_kmer_abundance:
                        continue

                    if cv[current_idx] < cv_cutoff:
                        continue

                    # if passes, save out
                    array_dict = {}
                    for key in hf.keys():
                        if "feature_metadata" in key:
                            array_dict[key] = hf[key][example_idx,0]
                        else:
                            array_dict[key] = hf[key][example_idx,:]

                    h5_handler.store_example(array_dict)

                current_idx = batch_end
                batch_end = current_idx + batch_size
        
            h5_handler.flush()
            h5_handler.chomp_datasets()
    
    return


def cluster_seqlets(seqlets_h5_file, out_h5_file):
    """Use phenograph to quickly cluster
    """

    with h5py.File(seqlets_h5_file, "r") as hf:

        communities, graph, Q = phenograph.cluster(hf["seqlet_x_task"], n_jobs=28)
        sort_indices = np.argsort(communities)

        # save results into h5 file (need to keep seqlets)
        with h5py.File(out_h5_file, "w") as out:

            for key in hf.keys():
                tmp_array = hf[key][:]
                
                if "feature_metadata" in key:
                    out.create_dataset(key, data=tmp_array[sort_indices,:])
                elif "seqlets" in key:
                    out.create_dataset(key, data=tmp_array[sort_indices,:,:])
                else:
                    out.create_dataset(key, data=tmp_array[sort_indices,:])
                
            # also add in communities
            out.create_dataset("communities", data=communities[sort_indices])

    return

def write_pwm(pwm_file, pwm, pwm_name):
    """Append a PWM (normalized to center at zero) to a motif file
    """
    normalized_pwm = pwm.normalize(style="probabilities")
    
    with open(pwm_file, 'a') as fp:
        fp.write('>{}\n'.format(pwm_name))
        for i in range(normalized_pwm.weights.shape[1]):
            vals = normalized_pwm.weights[:,i].tolist()
            val_strings = [str(val) for val in vals]
            fp.write('{}\n'.format('\t'.join(val_strings)))
        
    return None


def write_list_to_pwm_file(pwm_list, pwm_file, prefix):
    """Takes list of PWM objects and writes out to file and plots
    """
    for motif_idx in range(len(pwm_list)):
        motif_name = '{0}.motif_{1}'.format(prefix, motif_idx)
        write_pwm('{}.motif_file.txt'.format(prefix), pwm_list[motif_idx], motif_name)
        
    return


def plot_pwm_list(pwm_list, prefix):
    """Plot PWMs 
    """
    # for each motif, normalize and plot out
    for pwm_idx in range(len(pwm_list)):
        pwm = pwm_list[pwm_idx]
        plot_weights(
            pwm.weights.transpose(1, 0),
            '{0}.motif_{1}.png'.format(prefix, pwm_idx),
            figsize=(pwm.weights.shape[1],2))
        
    return

def print_pwm_list_as_string(pwm_list):
    """Convenience function
    """
    motif_strings = [kmer_array_to_string(motif.weights) for motif in pwm_list]
    print "motifs ({0}): {1}".format(len(pwm_list), motif_strings)
    
    return


def make_motif_sets(seqlets_w_communities_h5_file, prefix, debug=False):
    """Now that you have communities and clusters make motif sets
    """
    with h5py.File(seqlets_w_communities_h5_file, "r") as hf:

        # only pull communities that are not -1
        communities = sorted(list(set(hf["communities"])))
        communities_filt = [community for community in communities if community != -1]
        print communities_filt

        # for each community
        all_community_pwms = []
        community_trajectories = np.zeros((len(communities_filt), hf["seqlet_x_task"].shape[1]))
        for community_idx in range(len(communities_filt)):

            community = communities_filt[community_idx]
            
            # first pull together kmers that are the same and aggregate
            community_indices = np.where(hf["communities"][:] == community)[0]
            community_kmers = hf["kmer_id"][:][community_indices,0]
            community_seqlets = hf["seqlets"][:][community_indices,:,:]
            
            kmer_counts = Counter(community_kmers)

            # save aggregate pattern to trajectory
            community_seqlet_x_tasks = hf["seqlet_x_task"][:][community_indices,:]
            community_seqlet_x_tasks_z = zscore(community_seqlet_x_tasks, axis=1) # TODO check this
            community_seqlet_x_tasks_avg = np.mean(community_seqlet_x_tasks_z, axis=0)
            community_trajectories[community_idx,:] = community_seqlet_x_tasks_avg
            print "trajectory:", community_seqlet_x_tasks_avg
            
            # make a tmp array
            tmp_seqlets = np.zeros((len(kmer_counts), 4, hf["seqlets"].shape[2]))
            seed_motifs = []
            
            seqlet_idx = 0
            for kmer_hash in sorted(kmer_counts.keys()):

                kmer_indices = np.where(community_kmers == kmer_hash)[0]
                kmer_seqlets = community_seqlets[kmer_indices,:,:]

                # and then get the average (also normalize?)
                kmer_summed_seqlet = np.sum(kmer_seqlets, axis=0)

                # and save to tmp array
                tmp_seqlets[seqlet_idx,:,:] = kmer_summed_seqlet
                seqlet_idx += 1
                
                seed_motifs.append(PWM(kmer_summed_seqlet))

            # sort to make deterministic
            kmers_sort_indices = np.argsort([kmer_array_to_string(motif.weights) for motif in seed_motifs])
            seed_motifs_sorted = [seed_motifs[i] for i in kmers_sort_indices]
                
            # now hAgglom on the seqlets (keep track of counts?)
            single_community_pwms = agglom_motifs(seed_motifs_sorted, cut_fract=0.7)

            # print to show work
            normalized_motifs = [pwm.normalize(style="probabilities") for pwm in single_community_pwms]
            print_pwm_list_as_string(normalized_motifs)

            # for each motif, normalize and plot out
            community_prefix = "{}.community_{}".format(prefix, community)
            plot_pwm_list(normalized_motifs, community_prefix)
            all_community_pwms.append(single_community_pwms)

            # TODO(dk) for the community, track trajectory pattern

        # master motifs: take all community motifs and hagglom
        all_community_pwms_flattened = [pwm for community_list in all_community_pwms for pwm in community_list]
        sort_indices = np.argsort([kmer_array_to_string(pwm.weights) for pwm in all_community_pwms_flattened])
        all_community_pwms_flattened_sorted = [all_community_pwms_flattened[i] for i in sort_indices]
        print_pwm_list_as_string(all_community_pwms_flattened_sorted)
        
        master_pwms = agglom_motifs(all_community_pwms_flattened_sorted, cut_fract=0.8)
        normalized_master_pwms = [pwm.normalize(style="probabilities") for pwm in master_pwms]
        print_pwm_list_as_string(normalized_master_pwms)

        # save out
        master_prefix = "{}.master".format(prefix)
        master_pwm_file = "{}.pwms.txt".format(master_prefix)
        write_list_to_pwm_file(master_pwms, master_pwm_file, prefix)
        plot_pwm_list(normalized_master_pwms, master_prefix)
        
        # finally, compare the master list to community lists
        community_to_motif = np.zeros((len(communities_filt), len(master_pwms)))
        # for each master motif:
        # search for max fit with a motif in a community list
        # cutoff is 0.8
        for master_pwm_idx in range(len(master_pwms)):
            master_pwm = master_pwms[master_pwm_idx]
            for community_idx in range(len(all_community_pwms)):
                single_community_pwms = all_community_pwms[community_idx]
                xcor = 0
                for community_pwm in single_community_pwms:
                    score, offset = master_pwm.xcor(community_pwm)
                    if score > xcor:
                        xcor = score
                        
                community_to_motif[community_idx, master_pwm_idx] = xcor

        # make a datafrome of this and save out
        motif_strings = [kmer_array_to_string(pwm.weights) for pwm in master_pwms]
        community_to_motif_df = pd.DataFrame(data=community_to_motif, index=communities_filt, columns=motif_strings)
        community_to_motif_file = "{}.community_to_motif.txt".format(master_prefix)
        community_to_motif_df.to_csv(community_to_motif_file, sep='\t')

        # also make a dataframe of trajectory info and save out
        community_trajectories_df = pd.DataFrame(data=community_trajectories, index=communities_filt)
        community_trajectories_file = "{}.community_trajectories.txt".format(master_prefix)
        community_trajectories_df.to_csv(community_trajectories_file, sep='\t')
        
    return
