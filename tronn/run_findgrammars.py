# description: test function for a multitask interpretation pipeline

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter

from tronn.graphs import TronnGraph
from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.nets.nets import model_fns
from tronn.interpretation.importances import extract_importances_and_motif_hits
from tronn.interpretation.importances import get_pwm_hits_from_raw_sequence
from tronn.interpretation.importances import layerwise_relevance_propagation
from tronn.interpretation.importances import visualize_sample_sequences

from tronn.interpretation.importances import split_importances_by_task_positives
from tronn.interpretation.seqlets import extract_seqlets
from tronn.interpretation.seqlets import reduce_seqlets
from tronn.interpretation.seqlets import cluster_seqlets
from tronn.interpretation.seqlets import make_motif_sets

from tronn.datalayer import get_total_num_examples

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import get_encode_pwms


def setup_pwms(pwm_file, task_idx):
    """Janky helper function - to be cleaned up!
    """
    # with inference this produces the importances and motif hits.
    pwm_list = get_encode_pwms(pwm_file)

    pwms_per_task = { # remember 0-indexed
        0: [
            "ETS",      
            "FOSL",
            "NFKB1",
            "RUNX",
            "SOX3"],
        1: [
            "FOSL",
            "NFKB1",
            "RUNX",
            "SOX3"],
        2: [
            "FOSL",
            "NFKB1",
            "RUNX",
            "TEAD",
            "SOX3"],
        3: [
            "NFKB1",
            "FOSL",
            "TEAD",
            "RUNX"],
        4: [
            "FOSL1",
            "NFY",
            "TEAD",
            "RUNX",
            "TP63"],
        5: [
            "CEBPA",
            "KLF4",
            "NFY",
            "TEAD",
            "TP63",        
            "ZNF750"],
        6: [
            "CEBPA",
            "GRHL",
            "KLF4",
            "TP63",        
            "ZNF750"],
        7: [
            "CEBPA",
            "GRHL",
            "KLF4",
            "ZNF750"],
        8: [
            "CEBPA",
            "GRHL",
            "KLF4",
            "ZNF750"],
        9: [
            "CEBPA",
            "GRHL",
            "KLF4",
            "ZNF750"],
        10: [
            "FOSL1",
            "TP63"],
        11: [ # extra task for stable openness, dynamic H3K27ac
            "CEBPA",
            "GRHL",
            "KLF4",
            "ZNF750",
            "ETS"]
    }

    pwm_list_filt = []
    for pwm in pwm_list:
        for pwm_name in pwms_per_task[task_idx]:
            if pwm_name in pwm.name:
                pwm_list_filt.append(pwm)

    print "Using PWMS:", [pwm.name for pwm in pwm_list_filt]

    return pwm_list_filt


def run(args):
    """Find grammars utilizing the timeseries tasks
    """
    os.system('mkdir -p {}'.format(args.tmp_dir))
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logging.info("Found {} chrom files".format(len(data_files)))

    pwms_to_use = []
    with open(args.pwm_list, "r") as fp:
        for line in fp:
            pwms_to_use.append(line.strip().split('\t')[0])

    #args.interpretation_tasks = [16, 18, 19, 23]
    
    # go through each interpretation task
    for i in xrange(len(args.interpretation_tasks)):

        interpretation_task_idx = args.interpretation_tasks[i]
        
        # skip for now
        if interpretation_task_idx == 81:
            continue

        print interpretation_task_idx

        # set up pwms to use
        if False:
            try:
                pwm_list_filt = setup_pwms(args.pwm_file, i)
                pwm_names_filt = [pwm.name for pwm in pwm_list_filt]
            except:
                # didn't get the list set up right
                continue

        if True:
            pwm_list = get_encode_pwms(args.pwm_file)
            pwm_list_filt = []
            for pwm in pwm_list:
                for pwm_name in pwms_to_use:
                    if pwm_name in pwm.name:
                        pwm_list_filt.append(pwm)
            print "Using PWMS:", [pwm.name for pwm in pwm_list_filt]
            print len(pwm_list_filt)
            pwm_names_filt = [pwm.name for pwm in pwm_list_filt]

        # now check model type
        pwm_hits_mat_h5 = '{0}/{1}.task-{2}.pwm-hits.h5'.format(
            args.tmp_dir, args.prefix, interpretation_task_idx)

        if args.model["name"] == "get_top_k_motif_hits":
            # set up graph
            tronn_graph = TronnGraph(
                {"data": data_files},
                [],
                load_data_from_filename_list,
                model_fns[args.model["name"]],
                {"pwms": pwm_list_filt, "k_val": 5},
                args.batch_size,
                shuffle_data=False,
                filter_tasks=[interpretation_task_idx])

            # get pwm_hits
            if not os.path.isfile(pwm_hits_mat_h5):
                get_pwm_hits_from_raw_sequence(
                    tronn_graph,
                    pwm_hits_mat_h5,
                    args.sample_size,
                    pwm_list_filt)

        else:
            # set up graph
            tronn_graph = TronnNeuralNetGraph(
                {'data': data_files},
                args.tasks,
                load_data_from_filename_list,
                args.batch_size / 2,
                model_fns[args.model['name']],
                args.model,
                tf.nn.sigmoid,
                importances_fn=layerwise_relevance_propagation,
                importances_tasks=args.importances_tasks,
                shuffle_data=False,
                filter_tasks=[interpretation_task_idx])

            # checkpoint file
            if args.model_checkpoint is not None:
                checkpoint_path = args.model_checkpoint
            else:
                checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
            logging.info("Checkpoint: {}".format(checkpoint_path))

            # get importances
            if not os.path.isfile(pwm_hits_mat_h5):
                extract_importances_and_motif_hits(
                    tronn_graph,
                    checkpoint_path,
                    pwm_hits_mat_h5,
                    args.sample_size,
                    pwm_list_filt,
                    method="guided_backprop")

        # now in pwm hits file,
        # TODO(dk) factor this code out
        hash_to_motifs_file = "{0}/{1}.task-{2}.hash-to-motifs.txt".format(
            args.tmp_dir, args.prefix, interpretation_task_idx)

        # TODO keep the top hits?
        all_pwms_used = []
        if not os.path.isfile(hash_to_motifs_file):
            # extract hashes
            with h5py.File(pwm_hits_mat_h5, "r") as hf:
                pwm_hits = hf["pwm_hits"][:]
                # shrink first (ran into floating point issue for hash being so large)
                #pwm_hits = pwm_hits[:, ~np.all(pwm_hits == 0, axis=0)]

                pwm_presence = (pwm_hits > 0).astype(int)
                pwm_hash = np.zeros((pwm_hits.shape[0]), dtype=np.float128)
                for i in xrange(pwm_hits.shape[1]):
                    pwm_hash = pwm_hash + pwm_presence[:,i].astype(np.float128) * (np.float128(2)**i)

                # count hashes
                hash_counts = Counter(pwm_hash.tolist())
                hash_counts_ordered = hash_counts.most_common(20)
                print hash_counts_ordered
                logging.info(str(hash_counts_ordered))
                print pwm_hash.shape[0]

                count_threshold = 0.10 * pwm_hash.shape[0] # change this later
                count_threshold = 100
                for hash_val, count in hash_counts_ordered:
                    print hash_val

                    if hash_val == 0.0:
                        continue

                    if count < count_threshold:
                        continue

                    prefix = "{}.task-{}.hash-{}".format(args.prefix, interpretation_task_idx, "{0:.0f}".format(hash_val))

                    # get indices
                    hash_indices = np.where(pwm_hash == hash_val)[0]

                    # and extract regions
                    matching_regions = hf["example_metadata"][:][hash_indices]

                    # save out to file
                    with open("{0}/{1}.regions.txt".format(args.tmp_dir, prefix), "w") as fp:
                        for i in xrange(matching_regions.shape[0]):
                            fp.write("{}\n".format(matching_regions[i,0]))

                    # convert to bed
                    to_bed = (
                        "cat {0}/{1}.regions.txt |"
                        "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
                        "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
                        "sort -k1,1 -k2,2n | "
                        "bedtools merge -i stdin > "
                        "{0}/{1}.regions.bed").format(args.tmp_dir, prefix)
                    os.system(to_bed)

                    # get the motifs into a file
                    hash_motifs_present = (np.sum(pwm_presence[hash_indices], axis=0) > 0).astype(int)

                    motif_indices = np.where(hash_motifs_present)[0]
                    hash_motifs = [pwm_names_filt[i] for i in xrange(len(pwm_names_filt)) if i in motif_indices]
                    all_pwms_used += hash_motifs
                    hash_to_motifs_file = "{0}/{1}.task-{2}.hash-to-motifs.txt".format(
                        args.tmp_dir, args.prefix, interpretation_task_idx)
                    with open(hash_to_motifs_file, "a") as out:
                        out.write("{}\t{}\n".format(hash_val, ";".join(hash_motifs)))

                # get master list of PWMs for this task, and filter pwm hits to get only those columns
                # and save out the hits info with the regions info
                all_pwms_used = list(set(all_pwms_used))
                pwm_indices = [i for i in xrange(len(pwm_names_filt)) if pwm_names_filt[i] in all_pwms_used]
                pwm_used_names = [pwm_names_filt[i].split("_")[0] for i in xrange(len(pwm_names_filt)) if pwm_names_filt[i] in all_pwms_used]
                
                # filter out zeros, and non used pwms
                pwm_hits = pwm_hits[~np.all(pwm_hits == 0, axis=1),:]
                pwm_hits = pwm_hits[:, pwm_indices]
                pwm_hits = pwm_hits[:, ~np.all(pwm_hits == 0, axis=0)]

                # save out
                pwm_hits_df = pd.DataFrame(data=pwm_hits, index=hf["example_metadata"][:,0], columns=pwm_used_names)
                pwm_hits_df.to_csv(
                    "{0}/{1}.task-{2}.pwm_hits.small.mat.txt".format(
                        args.tmp_dir, args.prefix, interpretation_task_idx),
                    sep='\t')


                # backcheck on full set
                all_regions = "{0}/{1}.task-{2}.all.regions.txt".format(args.tmp_dir, args.prefix, interpretation_task_idx)
                with open(all_regions, "w") as fp:
                    for i in xrange(hf["example_metadata"][:].shape[0]):
                        fp.write("{}\n".format(hf["example_metadata"][i,0]))

                # convert to bed
                all_regions_bed = "{}.bed".format(all_regions.split(".txt")[0])
                to_bed = (
                    "cat {0} |"
                    "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
                    "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
                    "awk -F '\t' '{{ print $1\"\t\"$2+500\"\t\"$3+400 }}' | "
                    "sort -k1,1 -k2,2n | "
                    "bedtools merge -i stdin > "
                    "{1}").format(all_regions, all_regions_bed)
                os.system(to_bed)

    quit()

    # condense the results (ie make matrices that plot well)
    # first, take all the motifs (ordered) and keep hashes for them
    all_used_pwms = []
    for i in xrange(len(args.interpretation_tasks)):
        all_used_pwms += [pwm.name for pwm in setup_pwms(args.pwm_file, i)]

    all_used_pwms = sorted(list(set(all_used_pwms)))

    # set up a dict to easily multiply and get "universal" hash
    pwm_to_multiplier = {}
    pwm_to_idx = {}
    for i in xrange(len(all_used_pwms)):
        pwm_to_multiplier[all_used_pwms[i]] = 2**i
        pwm_to_idx[all_used_pwms[i]] = i

    # then for each task, read in the text dict and move to "universal" hash
    # ie, for the hash, get the motif set. using those motifs, recalc the universal hash.
    # outfile: {task}\t{univ_hash}\t{motif one hot series}
    master_combo_file = "{}/{}.motif-combos.master.txt".format(args.tmp_dir, args.prefix)
    with open(master_combo_file, "w") as out:
        # header
        out.write("task\thash\t{}\n".format("\t".join(all_used_pwms)))

        for i in xrange(len(args.interpretation_tasks)):
            
            interpretation_task_idx = args.interpretation_tasks[i]
            
            # skip for now
            if interpretation_task_idx == 81:
                continue

            # extract universal hash and save
            hash_to_motifs_file = "{0}/{1}.task-{2}.hash-to-motifs.txt".format(
                args.tmp_dir, args.prefix, interpretation_task_idx)
            try:
                with open(hash_to_motifs_file, "r") as fp:
                    for line in fp:
                        motif_combo = line.strip().split("\t")[1].split(";")
                        univ_hash = 0
                        onehot_motifs = [0 for i in xrange(len(all_used_pwms))]
                        for motif in motif_combo:
                            univ_hash += pwm_to_multiplier[motif]
                            onehot_motifs[pwm_to_idx[motif]] = 1
                        out.write("{}\t{}\t{}\n".format(interpretation_task_idx, univ_hash, "\t".join(map(str, onehot_motifs))))
            except IOError:
                continue
    quit()



    # per task file (use parallel processing):
    # extract the seqlets into other files with timepoints (seqlet x task)
    # AND keep track of seqlet size
    for task in args.interpretation_tasks:
        
        task_importance_file = "{}.task_{}.h5".format(prefix, task)
        task_seqlets_file = "{}.task_{}.seqlets.h5".format(prefix, task)

        if not os.path.isfile(task_seqlets_file):
            extract_seqlets(task_importance_file, args.importances_tasks, task_seqlets_file)

        # TODO filter seqlets
        task_seqlets_filt_file = "{}.task_{}.seqlets.filt.h5".format(prefix, task)

        if not os.path.isfile(task_seqlets_filt_file):
            reduce_seqlets(task_seqlets_file, task_seqlets_filt_file)
        
        # then cluster seqlets - phenograph
        # output: (seqlet, task) but clustered
        #cluster_seqlets("{}.task_{}.seqlets.h5".format(prefix, task))
        seqlets_w_communities_file = "{}.task_{}.seqlets.filt.communities.h5".format(prefix, task)

        if not os.path.isfile(seqlets_w_communities_file):
            cluster_seqlets(task_seqlets_filt_file, seqlets_w_communities_file)

        # then hAgglom the seqlets
        # output: motifs
        motif_dir = "{}/{}.motifs.task_{}".format(args.tmp_dir, args.prefix, task)
        motif_prefix = "{}/{}.task_{}".format(motif_dir, args.prefix, task)
        if not os.path.isdir(motif_dir):
            os.system("mkdir -p {}".format(motif_dir))
            make_motif_sets(seqlets_w_communities_file, motif_prefix)

        # and generate quick plots
        trajectories_txt = "{}.master.community_trajectories.txt".format(motif_prefix)
        communities_txt = "{}.master.community_to_motif.txt".format(motif_prefix)
        trajectories_png = "{}.master.community_trajectories.hclust.png".format(motif_prefix)
        communities_png = "{}.master.community_to_motif.hclust.png".format(motif_prefix)
        communities_corr_png = "{}.master.community_to_motif.corr.png".format(motif_prefix)
        os.system("Rscript make_motif_maps.R {} {} {} {} {}".format(
            communities_txt,
            trajectories_txt,
            trajectories_png,
            communities_png,
            communities_corr_png))
        
    return
