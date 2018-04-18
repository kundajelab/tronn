# description: scan motifs and get motif sets (co-occurring motifs) back

import os
import h5py
import glob
import logging

import phenograph

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter

#from tronn.graphs import TronnGraph
#from tronn.graphs import TronnNeuralNetGraph
from tronn.graphs import TronnGraphV2

#from tronn.datalayer import load_data_from_filename_list
#from tronn.datalayer import load_step_scaled_data_from_filename_list
#from tronn.datalayer import load_data_with_shuffles_from_filename_list
from tronn.datalayer import H5DataLoader
from tronn.nets.nets import net_fns

from tronn.interpretation.interpret import interpret
from tronn.interpretation.interpret import interpret_v2

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import read_pwm_file
#from tronn.interpretation.motifs import setup_pwms
#from tronn.interpretation.motifs import setup_pwm_metadata

from tronn.interpretation.motifs import get_minimal_motifsets
from tronn.interpretation.motifs import distill_to_linear_models
from tronn.interpretation.motifs import threshold_motifs
from tronn.interpretation.motifs import reduce_pwm_redundancy

from tronn.interpretation.clustering import cluster_by_task
from tronn.interpretation.clustering import enumerate_metaclusters
from tronn.interpretation.clustering import generate_simple_metaclusters
from tronn.interpretation.clustering import refine_clusters
from tronn.interpretation.clustering import visualize_clusters

from tronn.interpretation.clustering import get_correlation_file

from tronn.interpretation.learning import build_lasso_regression_models


def h5_dataset_to_text_file(h5_file, key, text_file, col_keep_indices, colnames):
    """Grab a dataset out of h5 (2D max) and save out to a text file
    """
    with h5py.File(h5_file, "r") as hf:
        dataset = hf[key][:][:,np.array(col_keep_indices)]
        
        # set up dataframe and save out
        dataset_df = pd.DataFrame(dataset, index=hf["example_metadata"][:][:,0], columns=colnames)
        dataset_df.to_csv(text_file, sep='\t')

    return None


def phenograph_cluster(mat_file, sorted_mat_file, num_threads=24):
    """Use to quickly get a nicely clustered (sorted) output file to visualize examples
    """
    # read in file and adjust as needed
    mat_df = pd.read_table(mat_file, index_col=0)
    print "total examples used:", mat_df.shape    
    communities, graph, Q = phenograph.cluster(mat_df, n_jobs=num_threads)

    # save out the sorted info into a new mat sorted by community
    mat_df["community"] = communities
    mat_df = mat_df.sort_values("community", axis=0)
    mat_df.to_csv(sorted_mat_file, sep='\t')
    mat_df = mat_df.drop(["community"], axis=1)

    return None


def enumerate_motifspace_communities(
        community_files,
        indices,
        prefix,
        pwm_list,
        sig_threshold=0.005,
        visualize=False):
    """given communities for each timepoint, 
    merge into one file and enumerate along start to finish
    use the indices to extract subset of regions from h5 file to save
    """
    # pull in the communities and merge to make a matrix of communities
    data = pd.DataFrame()
    for i in xrange(len(community_files)):
        community_file = community_files[i]
        task_index = indices[i]

        data_tmp = pd.read_table(community_file, sep="\t", index_col=0)
        if data.shape[0] == 0:
            data["id"] = data_tmp.index
            data.index = data_tmp.index
            data["task-{}".format(task_index)] = data_tmp["community"]
        else:
            data_tmp["id"] = data_tmp.index
            data_tmp = data_tmp[["id", "community"]]
            data_tmp.columns = ["id", "task-{}".format(task_index)]
            data = data.merge(data_tmp, how="inner", on="id")

    data.index = data["id"]
    data = data.drop(["id"], axis=1)
    
    # enumerate
    data["enumerated"] = ["" for i in xrange(data.shape[0])]
    for i in xrange(data.shape[1]):
        print i
        data["enumerated"] = data["enumerated"] + data.iloc[:, data.shape[1]-i-2].astype(str).str.zfill(2)

    # figure out which ones are significant (ie a size threshold) and only keep those
    community_enumerations = pd.DataFrame()
    from collections import Counter
    counts = Counter(data["enumerated"].tolist())
    enumerated_clusters = list(set(data["enumerated"].tolist()))

    for enumerated_cluster in enumerated_clusters:
        count = counts[enumerated_cluster]
        if float(count) / data.shape[0] >= sig_threshold:
            # keep
            communities_vector = data[data["enumerated"] == enumerated_cluster].iloc[0,:]
            community_enumerations = community_enumerations.append(
                communities_vector, ignore_index=True)[communities_vector.index.tolist()]

    community_enumerations.to_csv("{}.metacommunity_means.txt".format(prefix), sep="\t")
    
    # for each set of patterns, want to go through files and extract profiles
    for metacommunity_idx in xrange(community_enumerations.shape[0]):
        print metacommunity_idx
        metacommunity_prefix = "{}.metacommunity_{}".format(prefix, metacommunity_idx)
        
        timeseries_motif_scores = pd.DataFrame()
        metacommunity_h5_file = "{}.h5".format(metacommunity_prefix)
        timeseries_motif_file = "{}.means.txt".format(metacommunity_prefix)
        timeseries_region_ids_file = "{}.region_ids.txt".format(metacommunity_prefix)
        timeseries_bed_file = "{}.region_ids.bed".format(metacommunity_prefix)
        
        # get the related regions to text file
        regions = data[data["enumerated"] == community_enumerations["enumerated"].iloc[metacommunity_idx]]
        regions.to_csv(timeseries_region_ids_file, columns=[], header=False)
        
        # make a bed
        to_bed = (
            "cat {0} | "
            "awk -F ';' '{{ print $3 }}' | "
            "awk -F '=' '{{ print $2 }}' | "
            "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
            "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
            "sort -k1,1 -k2,2n | "
            "bedtools merge -i stdin > "
            "{1}").format(
                timeseries_region_ids_file,
                timeseries_bed_file)
        print to_bed
        os.system(to_bed)

        # save out motif x timepoint, and region arrays
        with h5py.File(metacommunity_h5_file, "w") as hf:
            # make dataset
            features = hf.create_dataset(
                "features",
                (regions.shape[0], len(pwm_list), len(community_files)))
            example_metadata = hf.create_dataset(
                "example_metadata",
                (regions.shape[0],), dtype="S1000")
            pwm_names = hf.create_dataset(
                "pwm_names",
                (len(pwm_list),), dtype="S100")
            
            example_metadata[:] = regions.index.tolist()
            pwm_names[:] = [pwm.name for pwm in pwm_list]
            
            # go through community files
            for i in xrange(len(community_files)):
                community_file = community_files[i]
                index = indices[i]

                # extract the metacommunity
                data_tmp = pd.read_table(community_file, sep="\t", index_col=0)
                data_tmp = data_tmp.loc[regions.index,:]
                data_tmp = data_tmp.drop("community", axis=1)
                features[:,:,i] = data_tmp
                
                # get the mean across columns
                data_mean = data_tmp.mean(axis=0)

                # append
                timeseries_motif_scores = timeseries_motif_scores.append(data_mean, ignore_index=True)

            # save out mean vectors
            timeseries_motif_scores = timeseries_motif_scores.fillna(0)
            timeseries_motif_scores.to_csv(timeseries_motif_file, sep="\t")

            # visualize
            if visualize:
                viz_file = "{}.pdf".format(timeseries_motif_file.split(".txt")[0])
                plot_pwm_x_task = "plot.pwm_x_task.R {} {}".format(
                    timeseries_motif_file, viz_file)
                print(plot_pwm_x_task)
                os.system(plot_pwm_x_task)
                
    return None


def run(args):
    """Scan motifs from a PWM file
    """
    # setup
    logger = logging.getLogger(__name__)
    logger.info("Running motif scan")
    if args.tmp_dir is not None:
        os.system('mkdir -p {}'.format(args.tmp_dir))
    else:
        args.tmp_dir = args.out_dir
    
    # data files
    data_files = glob.glob('{}/*.h5'.format(args.data_dir))
    logger.info("Found {} chrom files".format(len(data_files)))
    
    # motif annotations
    # TODO - clean this up
    #pwm_name_to_hgnc, hgnc_to_pwm_name = setup_pwm_metadata(args.pwm_metadata_file)
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    #pwm_names_clean = [pwm_name.split(".")[0].split("_")[1] for pwm_name in pwm_names]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))

    # set up file loader, dependent on importance fn
    #if args.backprop == "integrated_gradients":
    #    data_loader_fn = load_step_scaled_data_from_filename_list
    #elif args.backprop == "deeplift":
    #    data_loader_fn = load_data_with_shuffles_from_filename_list
    #else:
    #    data_loader_fn = load_data_from_filename_list
        # TESTING FOR SHUFFLE NULL
        #data_loader_fn = load_data_with_shuffles_from_filename_list
        #print "WARNING USING SHUFFLES"
    dataloader = H5DataLoader(
        {"data": data_files},
        filter_tasks=args.filter_tasks)
        
    # set up graph
    # TODO somewhere here need to pass forward the
    # shuffles/steps information from dataloader into the graph,
    # to be stored in the config for inference.
    if False:
        tronn_graph = TronnNeuralNetGraph(
            {'data': data_files},
            args.tasks,
            dataloader,
            args.batch_size,
            net_fns[args.model['name']],
            args.model,
            tf.nn.sigmoid,
            inference_fn=net_fns[args.inference_fn],
            importances_tasks=args.inference_tasks,
            shuffle_data=True,
            filter_tasks=args.filter_tasks,
            checkpoints=args.model_checkpoints)

    # set up graph
    tronn_graph = TronnGraphV2(
        dataloader,
        net_fns[args.model["name"]],
        args.model,
        args.batch_size,
        final_activation_fn=tf.nn.sigmoid,
        checkpoints=args.model_checkpoints)

    # run interpretation graph
    results_h5_file = "{0}/{1}.inference.h5".format(
        args.tmp_dir, args.prefix)
    if not os.path.isfile(results_h5_file):
        infer_params = {
            "inference_fn": net_fns[args.inference_fn],
            "importances_fn": args.backprop,
            "importance_task_indices": args.inference_tasks,
            "pwms": pwm_list}
        interpret_v2(tronn_graph, results_h5_file, infer_params)

    # validation tools
    #if args.diagnose:
    #    visualize = True
    #    validate_grammars = True
    #else:
    #    visualize = args.plot_importance_sample
    #    validate_grammars = False

    #if visualize == True:
    #    viz_dir = "{}/viz".format(args.out_dir)
    #    os.system("mkdir -p {}".format(viz_dir))
        
    # run region clustering/motif sets. default is true, but user can turn off
    # TODO split this out into another function
    pwm_scores_h5 = results_h5_file
    if not args.no_groups:
        visualize = True
        dataset_keys = ["pwm-scores.taskidx-{}".format(i)
                        for i in args.inference_tasks] # eventually, this is the right one
        
        # 1) cluster communities 
        cluster_key = "louvain_clusters" # later, change to pwm-louvain-clusters
        if False:
        #if cluster_key not in h5py.File(pwm_scores_h5, "r").keys():
            cluster_by_task(pwm_scores_h5, dataset_keys, cluster_key)
            if visualize:
                for i in xrange(len(dataset_keys)):
                    visualize_clusters(
                        pwm_scores_h5,
                        dataset_keys[i],
                        cluster_key, i)

        # refine - remove small clusters
        refined_cluster_key = "task-clusters-refined"
        #if refined_cluster_key not in h5py.File(pwm_scores_h5, "r").keys():
        if False:
        #if True:
            refine_clusters(pwm_scores_h5, cluster_key, refined_cluster_key)
            if visualize:
                for i in xrange(len(dataset_keys)):
                    visualize_clusters(
                        pwm_scores_h5,
                        dataset_keys[i],
                        refined_cluster_key, i,
                        remove_final_cluster=1)
            
        # 2) optional - correlation matrix.
        correlations_key = "task-pwm_x_pwm-correlations"
        if correlations_key not in h5py.File(pwm_scores_h5, "r").keys():
            # TODO
            pass
    
        # 3) enumerate metaclusters. dont visualize here because you must refine first
        # HERE - need to figure out a better way to metacluster.
        # enumeration is losing things that are dynamic
        metacluster_key = "metaclusters"
        if metacluster_key not in h5py.File(pwm_scores_h5, "r").keys():
        #if True:
            generate_simple_metaclusters(pwm_scores_h5, dataset_keys, metacluster_key)
        
        #if metacluster_key not in h5py.File(pwm_scores_h5, "r").keys():
        #if True:
        #    enumerate_metaclusters(pwm_scores_h5, cluster_key, metacluster_key)

        # refine - remove small clusters
        # TODO - put out BED files - write a separate function to pull BED from cluster set
        refined_metacluster_key = "metaclusters-refined"
        if refined_metacluster_key not in h5py.File(pwm_scores_h5, "r").keys():
        #if True:
            refine_clusters(pwm_scores_h5, metacluster_key, refined_metacluster_key, null_cluster_present=False)
            if visualize:
                for i in xrange(len(dataset_keys)):
                    visualize_clusters(
                        pwm_scores_h5,
                        dataset_keys[i],
                        refined_metacluster_key, 0,
                        remove_final_cluster=1)

        # 4) extract the constrained motif set for each metacommunity, for each task
        # new pwm vectors for each dataset..
        # save out initial grammar file, use labels to set a threshold
        # if visualize - save out mean vector and plot, also network vis?
        metacluster_motifs_key = "metaclusters-motifs"
        motifset_metacluster_key = "metaclusters-motifset-refined"
        if True:
        #if metacluster_motifs_key not in h5py.File(pwm_scores_h5, "r").keys():
            distill_to_linear_models(
                pwm_scores_h5,
                dataset_keys,
                refined_metacluster_key,
                motifset_metacluster_key,
                metacluster_motifs_key,
                pwm_list, pwm_dict,
                pwm_file=args.pwm_file,
                label_indices=args.inference_tasks) # eventually shouldnt need this, access name
            # or access some kind of attribute
            if visualize:
                for i in xrange(len(dataset_keys)):
                    visualize_clusters(
                        pwm_scores_h5,
                        dataset_keys[i],
                        motifset_metacluster_key, 0,
                        remove_final_cluster=1)

        # 5) optional - separately, get metrics on all tasks and save out (AUPRC, etc)
        # think of as a giant confusion matrix
        
        



    
    quit()
    
    if not args.no_groups:
        logger.info("Clustering regions per task.")
    
        # now for each timepoint task, go through and calculate communities
        for i in xrange(len(args.inference_tasks)):

            interpretation_task_idx = args.inference_tasks[i]
            logger.info("finding communities for task {}".format(interpretation_task_idx))

            # extract motif mat (region, motif) and save out to text file (to handle in R or python)
            region_x_pwm_mat_file = "{0}/{1}.task-{2}.region_x_pwm.txt".format(
                args.tmp_dir, args.prefix, interpretation_task_idx)
            logger.info("extracting region x pwm matrix")
            if not os.path.isfile(region_x_pwm_mat_file):
                h5_dataset_to_text_file(
                    pwm_scores_h5,
                    "pwm-scores.taskidx-{}".format(i), # use i because the ordering in the file is just 0-10
                    region_x_pwm_mat_file,
                    range(len(pwm_list)),
                    pwm_names)

            # get a sorted (ie clustered) version of the motif mat using phenograph (Louvain)
            # TODO somehow need to get the labels to exist with this file also....
            region_x_pwm_sorted_mat_file = "{0}.phenograph_sorted.txt".format(
                region_x_pwm_mat_file.split(".txt")[0])
            logger.info("using louvain communities to cluster regions")
            if not os.path.isfile(region_x_pwm_sorted_mat_file):
                phenograph_cluster(region_x_pwm_mat_file, region_x_pwm_sorted_mat_file)

            if visualize:
                # here, plot a example x pwm plot
                viz_file = "{0}/{1}.pdf".format(
                    viz_dir,
                    os.path.basename(region_x_pwm_sorted_mat_file).split(".txt")[0])
                plot_example_x_pwm = "plot.example_x_pwm.R {0} {1}".format(
                    region_x_pwm_sorted_mat_file, viz_file)
                print(plot_example_x_pwm)
                os.system(plot_example_x_pwm)
                
            # get the correlation matrix to look at, not really necessary?
            pwm_x_pwm_corr_file = "{0}/{1}.task-{2}.pwm_x_pwm.corr.mat.txt".format(
                args.tmp_dir, args.prefix, interpretation_task_idx)
            logger.info("get the correlation matrix")
            if not os.path.isfile(pwm_x_pwm_corr_file):
                get_correlation_file(
                    region_x_pwm_sorted_mat_file,
                    pwm_x_pwm_corr_file,
                    corr_method="continuous_jaccard")

            if visualize:
                # plot the correlation plot too
                viz_file = "{0}/{1}.pdf".format(
                    viz_dir,
                    os.path.basename(pwm_x_pwm_corr_file).split(".mat.txt")[0])
                plot_pwm_x_pwm = "plot.pwm_x_pwm.R {0} {1}".format(
                    pwm_x_pwm_corr_file, viz_file)
                print(plot_pwm_x_pwm)
                os.system(plot_pwm_x_pwm)

        # and then enumerate
        community_files = [
            "{0}/{1}.task-{2}.region_x_pwm.phenograph_sorted.txt".format(
                args.out_dir, args.prefix, i)
            for i in args.inference_tasks]

        # maybe put these into separate folders?
        # TODO figure out how to get labels in these h5 files too
        logger.info("enumerating metacommunities")
        metacommunity_files = sorted(
            glob.glob("{}/{}.metacommunity_*.h5".format(
                args.out_dir, args.prefix)))
        if len(metacommunity_files) == 0:
            enumerate_motifspace_communities(
                community_files,
                args.inference_tasks,
                "{}/{}".format(args.out_dir, args.prefix),
                pwm_list)
        metacommunity_files = sorted(
            glob.glob("{}/{}.metacommunity_*.h5".format(
                args.out_dir, args.prefix)))

        if visualize:
            # plot!
            pass

        # get the constrained motif set
        for i in xrange(len(metacommunity_files)):

            metacommunity_file = metacommunity_files[i]
            metacommunity_prefix = os.path.basename(metacommunity_file).split(".h5")[0]
            metacommunity_region_file = "{}.region_ids.refined.txt".format(metacommunity_file.split(".h5")[0])
            metacommunity_bed_file = "{}.bed".format(metacommunity_region_file.split(".txt")[0])
            metacommunity_regions = []
            metacommunity_pwms = []
            metacommunity_pwm_x_task_means_file = "{}.means.txt".format(
                metacommunity_region_file.split(".txt")[0])
            print metacommunity_file
            
            with h5py.File(metacommunity_file, "r") as hf:
                for task_idx in xrange(len(args.inference_tasks)):
                    inference_task_idx = args.inference_tasks[task_idx]

                    # get arrays
                    data_tmp = hf["features"][:,:,task_idx] # {N, pwm}
                    pwm_names = hf["pwm_names"][:]
                    regions = hf["example_metadata"][:]
                    
                    # threshold motifs                    
                    pwm_keep_indices = threshold_motifs(data_tmp)
                    data_tmp = data_tmp[:, pwm_keep_indices[0]]
                    pwm_names_thresh = pwm_names[pwm_keep_indices[0]]
                    
                    # reduce by motif similarity
                    if len(pwm_names) > 1:
                        task_pwms = [pwm_dict[pwm_name] for pwm_name in pwm_names_thresh]
                        pwm_subset = reduce_pwm_redundancy(task_pwms, pwm_dict, data_tmp)
                        pwm_subset = [pwm.name for pwm in pwm_subset]
                        pwm_keep_indices = np.where([True if pwm.name in pwm_subset else False
                                                     for pwm in task_pwms])
                        data_tmp = data_tmp[:, pwm_keep_indices[0]]
                        pwm_names = pwm_names[pwm_keep_indices[0]]
                    else:
                        pwm_subset = pwm_names
                    
                    # check set coverage
                    region_keep_indices = np.where(~np.any(data_tmp == 0, axis=1))
                    data_tmp = data_tmp[region_keep_indices[0],:]                    
                    regions = regions[region_keep_indices[0]]

                    # TODO - set up thresholds here. threshold at FDR 0.05?
                    # get back TPR and FPR
                    # now need a label matrix
                    global_pwm_keep_indices = np.where([True if pwm_name in pwm_subset else False
                                                        for pwm_name in pwm_names])
                    scores = np.sum(hf["features"][:,:,task_idx][:, global_pwm_keep_indices[0]], axis=1) # {N}
                    
                    
                    # save out a grammar file of the core PWMs
                    grammar_file = "{}.motifset.grammar".format(
                        metacommunity_file.split(".h5")[0])
                    node_dict = {}
                    for pwm in pwm_subset:
                        node_dict[pwm] = 1.0 # motifset, so all are equal
                    metacommunity_task_grammar = Grammar(
                        args.pwm_file,
                        node_dict,
                        {},
                        "taskidx={0};type=metacommunity;directed=no".format(task_idx),
                        "{0}.taskidx-{1}".format(
                            metacommunity_prefix,
                            inference_task_idx))
                    metacommunity_task_grammar.to_file(grammar_file)

                    # keep union of all regions
                    metacommunity_regions += regions.tolist()
                    metacommunity_regions = list(set(metacommunity_regions))

                    # keep union of all pwms
                    metacommunity_pwms += pwm_subset
                    metacommunity_pwms = list(set(metacommunity_pwms))
                    
                # use union of all regions and union of all pwms to get a pwm x task matrix
                pwm_x_task_scores = pd.DataFrame()
                for task_idx in xrange(len(args.inference_tasks)):
                    inference_task_idx = args.inference_tasks[task_idx]
                    
                    # get arrays into a dataframe
                    data_df = pd.DataFrame(
                        hf["features"][:,:,task_idx],
                        index=hf["example_metadata"][:],
                        columns=hf["pwm_names"][:])

                    # filter
                    data_df = data_df.loc[metacommunity_pwms, metacommunity_pwms]

                    # get the mean
                    data_mean = data_df.mean(axis=0)
                    
                    # append
                    pwm_x_task_scores = pwm_x_task_scores.append(
                        data_mean, ignore_index=True)

                # and save out
                pwm_x_task_scores = pwm_x_task_scores.fillna(0)
                pwm_x_task_scores.to_csv(metacommunity_pwm_x_task_means_file, sep="\t")
                
                # visualize
                if visualize:
                    viz_file = "{}.pdf".format(
                        metacommunity_pwm_x_task_means_file.split(".txt")[0])
                    plot_pwm_x_task = "plot.pwm_x_task.R {} {}".format(
                        metacommunity_pwm_x_task_means_file, viz_file)
                    print(plot_pwm_x_task)
                    os.system(plot_pwm_x_task)

                # write out master region set to file
                with open(metacommunity_region_file, "w") as out:
                    for region in metacommunity_regions:
                        out.write("{}\n".format(region))

                # convert to a bed file
                to_bed = (
                    "cat {0} | "
                    "awk -F ';' '{{ print $3 }}' | "
                    "awk -F '=' '{{ print $2 }}' | "
                    "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
                    "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
                    "sort -k1,1 -k2,2n | "
                    "bedtools merge -i stdin > "
                    "{1}").format(
                        metacommunity_region_file,
                        metacommunity_bed_file)
                print to_bed
                os.system(to_bed)

    return None

