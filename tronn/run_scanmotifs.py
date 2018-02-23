# description: test function for a multitask interpretation pipeline


import matplotlib
matplotlib.use("Agg")

import os
import h5py
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from collections import Counter
from multiprocessing import Pool

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from tronn.graphs import TronnGraph
from tronn.graphs import TronnNeuralNetGraph
from tronn.datalayer import load_data_from_filename_list
from tronn.datalayer import load_step_scaled_data_from_filename_list
from tronn.datalayer import load_data_with_shuffles_from_filename_list
from tronn.nets.nets import net_fns

from tronn.interpretation.interpret import interpret

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import read_pwm_file
from tronn.interpretation.motifs import setup_pwms
from tronn.interpretation.motifs import setup_pwm_metadata

from tronn.interpretation.grammars import get_significant_correlations
from tronn.interpretation.grammars import reduce_corr_mat_by_motif_similarity
from tronn.interpretation.grammars import get_significant_motifs

#from tronn.interpretation.grammars import read_grammar_file
from tronn.interpretation.grammars import Grammar

import phenograph


def h5_dataset_to_text_file(h5_file, key, text_file, col_keep_indices, colnames):
    """Grab a dataset out of h5 (2D max) and save out to a text file
    """
    with h5py.File(h5_file, "r") as hf:
        dataset = hf[key][:][:,np.array(col_keep_indices)]
        
        # set up dataframe and save out
        dataset_df = pd.DataFrame(dataset, index=hf["example_metadata"][:][:,0], columns=colnames)
        dataset_df.to_csv(text_file, sep='\t')

    return None



def phenograph_cluster(mat_file, sorted_mat_file):
    """Use to quickly get a nicely clustered (sorted) output file to visualize examples
    """
    # read in file and adjust as needed
    mat_df = pd.read_table(mat_file, index_col=0)
    print "total examples used:", mat_df.shape    
    communities, graph, Q = phenograph.cluster(mat_df)

    # save out the sorted info into a new mat sorted by community
    mat_df["community"] = communities
    mat_df = mat_df.sort_values("community", axis=0)
    mat_df.to_csv(sorted_mat_file, sep='\t')
    mat_df = mat_df.drop(["community"], axis=1)

    return None


def get_correlation_file(
        mat_file,
        corr_file,
        corr_method="intersection_size", # continuous_jaccard, pearson
        corr_min=0.4,
        pval_thresh=0.05):
    """Given a matrix file, calculate correlations across the columns
    """
    mat_df = pd.read_table(mat_file, index_col=0)
    mat_df = mat_df.drop(["community"], axis=1)
            
    corr_mat, pval_mat = get_significant_correlations(
        mat_df.as_matrix(),
        corr_method=corr_method,
        corr_min=corr_min,
        pval_thresh=pval_thresh)
    
    corr_mat_df = pd.DataFrame(corr_mat, index=mat_df.columns, columns=mat_df.columns)
    corr_mat_df.to_csv(corr_file, sep="\t")

    return None



def enumerate_motifspace_communities(
        community_files,
        indices,
        prefix,
        pwm_list,
        sig_threshold=0.005):
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
            
    return None


def threshold_motifs(array, std_thresh=3):
    """Given a matrix, threshold out motifs (columns) that are low in signal
    """
    # opt 1 - just get top k
    # opt 2 - fit a normal distr and use standard dev cutoff
    # opt 3 - shuffled vals?

    # row normalize
    array_norm = np.divide(array, np.max(array, axis=1, keepdims=True))
    array_means = np.mean(array_norm, axis=0)
    
    # for now - across all vals, get a mean and standard dev
    mean_val = np.mean(array_means)
    std_val = np.std(array_means)

    # and threshold
    keep_indices = np.where(array_means > (mean_val + (std_val * std_thresh)))
    
    return keep_indices


def correlate_pwm_pair(input_list):
    """get cor and ncor for pwm1 and pwm2
    Set up this way because multiprocessing pool only takes 1
    input
    """
    i = input_list[0]
    j = input_list[1]
    pwm1 = input_list[2]
    pwm2 = input_list[3]
    
    motif_cor = pwm1.rsat_cor(pwm2)
    motif_ncor = pwm1.rsat_cor(pwm2, ncor=True)

    return i, j, motif_cor, motif_ncor


def correlate_pwms(
        pwms,
        cor_thresh=0.6,
        ncor_thresh=0.4,
        num_threads=24):
    """Correlate PWMS
    """
    # set up
    num_pwms = len(pwms)
    cor_mat = np.zeros((num_pwms, num_pwms))
    ncor_mat = np.zeros((num_pwms, num_pwms))

    pool = Pool(processes=num_threads)
    pool_inputs = []
    # for each pair of motifs, get correlation information
    for i in xrange(num_pwms):
        for j in xrange(num_pwms):

            # only calculate upper triangle
            if i > j:
                continue

            pwm_i = pwms[i]
            pwm_j = pwms[j]
            
            pool_inputs.append((i, j, pwm_i, pwm_j))

    # run multiprocessing
    pool_outputs = pool.map(correlate_pwm_pair, pool_inputs)

    for i, j, motif_cor, motif_ncor in pool_outputs:
        # if passes cutoffs, save out to matrix
        if (motif_cor >= cor_thresh) and (motif_ncor >= ncor_thresh):
            cor_mat[i,j] = motif_cor
            ncor_mat[i,j] = motif_ncor        

    # and reflect over the triangle
    lower_triangle_indices = np.tril_indices(cor_mat.shape[0], -1)
    cor_mat[lower_triangle_indices] = cor_mat.T[lower_triangle_indices]
    ncor_mat[lower_triangle_indices] = ncor_mat.T[lower_triangle_indices]

    # multiply each by the other to double threshold
    cor_present = (cor_mat > 0).astype(float)
    ncor_present = (ncor_mat > 0).astype(float)

    # and mask
    cor_filt_mat = cor_mat * ncor_present
    ncor_filt_mat = ncor_mat * cor_present

    return cor_filt_mat, ncor_filt_mat


def hagglom_pwms(
        cor_mat_file,
        pwm_dict,
        array,
        ic_thresh=0.4,
        cor_thresh=0.8,
        ncor_thresh=0.65):
    """hAgglom on the PWMs to reduce redundancy
    """
    # read in table
    cor_df = pd.read_table(cor_mat_file, index_col=0)

    # set up pwm lists
    # set up (PWM, weight)
    hclust_pwms = [(pwm_dict[key], 1.0) for key in cor_df.columns.tolist()]
    non_redundant_pwms = []
    pwm_position = {}
    for i in xrange(len(hclust_pwms)):
        pwm, _ = hclust_pwms[i]
        pwm_position[pwm.name] = i

    # hierarchically cluster
    hclust = linkage(squareform(1 - cor_df.as_matrix()), method="ward")

    # keep a list of pwms in hclust, when things get merged add to end
    # (to match the scipy hclust structure)
    # put a none if not merging
    # if the motif did not successfully merge with its partner, pull out
    # it and its partner. if there was a successful merge, keep in there
    for i in xrange(hclust.shape[0]):
        idx1, idx2, dist, cluster_size = hclust[i,:]

        # check if indices are None
        pwm1, pwm1_weight = hclust_pwms[int(idx1)]
        pwm2, pwm2_weight = hclust_pwms[int(idx2)]

        if (pwm1 is None) and (pwm2 is None):
            hclust_pwms.append((None, None))
            continue
        elif (pwm1 is None):
            # save out PWM 2
            #print "saving out {}".format(pwm2.name)
            non_redundant_pwms.append(pwm2)
            hclust_pwms.append((None, None))
            continue
        elif (pwm2 is None):
            # save out PWM1
            #print "saving out {}".format(pwm1.name)
            non_redundant_pwms.append(pwm1)
            hclust_pwms.append((None, None))
            continue

        # try check
        try:
            cor_val, offset = pwm1.pearson_xcor(pwm2, ncor=False)
            ncor_val, offset = pwm1.pearson_xcor(pwm2, ncor=True)
        except:
            import ipdb
            ipdb.set_trace()

        if (cor_val > cor_thresh) and (ncor_val >= ncor_thresh):
            # if good match, now check the mat_df for which one
            # is most represented across sequences, and keep that one
            pwm1_presence = np.where(array[:,pwm_position[pwm1.name]] > 0)
            pwm2_presence = np.where(array[:,pwm_position[pwm2.name]] > 0)

            if pwm1_presence[0].shape[0] >= pwm2_presence[0].shape[0]:
                # keep pwm1
                #print "keep {} over {}".format(pwm1.name, pwm2.name)
                hclust_pwms.append((pwm1, 1.0))
            else:
                # keep pwm2
                #print "keep {} over {}".format(pwm2.name, pwm1.name)
                hclust_pwms.append((pwm2, 1.0))
        else:
            #print "saving out {}".format(pwm1.name)
            #print "saving out {}".format(pwm2.name)
            non_redundant_pwms.append(pwm1)
            non_redundant_pwms.append(pwm2)
            hclust_pwms.append((None, None))

    return non_redundant_pwms


def reduce_pwm_redundancy(
        pwms,
        pwm_dict,
        array,
        tmp_prefix="motifs",
        ic_thresh=0.4,
        cor_thresh=0.6,
        ncor_thresh=0.4,
        num_threads=28):
    """

    Note that RSAT stringent thresholds were ncor 0.65, cor 0.8
    Nonstringent is ncor 0.4 and cor 0.6
    """
    # trim pwms
    pwms = [pwm.chomp(ic_thresh=ic_thresh) for pwm in pwms]
    for key in pwm_dict.keys():
        pwm_dict[key] = pwm_dict[key].chomp(ic_thresh=ic_thresh)
    pwms_ids = [pwm.name for pwm in pwms]
    
    # correlate pwms - uses multiprocessing
    cor_mat_file = "{}.cor.motifs.mat.txt".format(tmp_prefix)
    ncor_mat_file = "{}.ncor.motifs.mat.txt".format(tmp_prefix)

    cor_filt_mat, ncor_filt_mat = correlate_pwms(
        pwms,
        cor_thresh=cor_thresh,
        ncor_thresh=ncor_thresh,
        num_threads=num_threads)
        
    # pandas and save out
    cor_df = pd.DataFrame(cor_filt_mat, index=pwms_ids, columns=pwms_ids)
    cor_df.to_csv(cor_mat_file, sep="\t")
    ncor_df = pd.DataFrame(ncor_filt_mat, index=pwms_ids, columns=pwms_ids)
    cor_df.to_csv(ncor_mat_file, sep="\t")

    # read in matrix to save time
    pwm_subset = hagglom_pwms(
        ncor_mat_file,
        pwm_dict,
        array,
        ic_thresh=ic_thresh,
        cor_thresh=cor_thresh,
        ncor_thresh=ncor_thresh)

    # once done, clean up
    os.system("rm {} {}".format(cor_mat_file, ncor_mat_file))

    return pwm_subset


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
    pwm_name_to_hgnc, hgnc_to_pwm_name = setup_pwm_metadata(args.pwm_metadata_file)
    pwm_list = read_pwm_file(args.pwm_file)
    pwm_names = [pwm.name for pwm in pwm_list]
    pwm_names_clean = [pwm_name.split("_")[0] for pwm_name in pwm_names]
    pwm_dict = read_pwm_file(args.pwm_file, as_dict=True)
    logger.info("{} motifs used".format(len(pwm_list)))

    # set up file loader, dependent on importance fn
    if args.backprop == "integrated_gradients":
        data_loader_fn = load_step_scaled_data_from_filename_list
    elif args.backprop == "deeplift":
        data_loader_fn = load_data_with_shuffles_from_filename_list
    else:
        data_loader_fn = load_data_from_filename_list
    
    # set up graph
    tronn_graph = TronnNeuralNetGraph(
        {'data': data_files},
        args.tasks,
        data_loader_fn,
        args.batch_size,
        net_fns[args.model['name']],
        args.model,
        tf.nn.sigmoid,
        inference_fn=net_fns[args.inference_fn],
        importances_tasks=args.inference_tasks,
        shuffle_data=True,
        filter_tasks=args.filter_tasks) # example - filter for dynamic tasks, or H3K27ac tasks

    # checkpoint file (unless empty net)
    if args.model_checkpoint is not None:
        checkpoint_path = args.model_checkpoint
    elif args.model["name"] == "empty_net":
        checkpoint_path = None
    else:
        checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
        
    # validation tools
    if args.validate:
        visualize = True
    else:
        visualize = args.plot_importance_sample
        
    # get pwm scores
    logger.info("calculating pwm scores...")
    pwm_scores_h5 = '{0}/{1}.pwm-scores.h5'.format(
        args.tmp_dir, args.prefix)
    if not os.path.isfile(pwm_scores_h5):
        interpret(
            tronn_graph,
            checkpoint_path,
            args.batch_size,
            pwm_scores_h5,
            args.sample_size,
            {"importances_fn": args.backprop,
             "pwms": pwm_list},
            keep_negatives=False,
            visualize=visualize, # TODO check this
            scan_grammars=False,
            validate_grammars=False,
            filter_by_prediction=True)

    # run region clustering/motif sets. default is true, but user can turn off
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
            region_x_pwm_sorted_mat_file = "{0}.phenograph_sorted.txt".format(
                region_x_pwm_mat_file.split(".txt")[0])
            logger.info("using louvain communities to cluster regions")
            if not os.path.isfile(region_x_pwm_sorted_mat_file):
                phenograph_cluster(region_x_pwm_mat_file, region_x_pwm_sorted_mat_file)

            if args.validate:
                # here, plot a example x pwm plot
                pass
            
            # get the correlation matrix to look at, not really necessary?
            pwm_x_pwm_corr_file = "{0}/{1}.task-{2}.pwm_x_pwm.corr.mat.txt".format(
                args.tmp_dir, args.prefix, interpretation_task_idx)
            logger.info("get the correlation matrix")
            if not os.path.isfile(pwm_x_pwm_corr_file):
                get_correlation_file(
                    region_x_pwm_sorted_mat_file,
                    pwm_x_pwm_corr_file,
                    corr_method="continuous_jaccard")

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

        # get the constrained motif set
        for i in xrange(len(metacommunity_files)):

            metacommunity_file = metacommunity_files[i]
            metacommunity_prefix = os.path.basename(metacommunity_file).split(".h5")[0]
            metacommunity_region_file = "{}.region_ids.refined.txt".format(metacommunity_file.split(".h5")[0])
            metacommunity_bed_file = "{}.bed".format(metacommunity_region_file.split(".txt")[0])
            metacommunity_regions = []
            #if os.path.isfile(metacommunity_bed_file):
            #    continue
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
                    pwm_names = pwm_names[pwm_keep_indices[0]]
                    
                    # reduce by motif similarity
                    if len(pwm_names) > 1:
                        task_pwms = [pwm_dict[pwm_name] for pwm_name in pwm_names]
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

                    # keep the UNION of set coverages (since the motif presence will have been seen somewhere)
                    # ^ purpose - a master bed file for this metacommunity, where ALL regions have a consistent grammar
                    # at minimally 1 cell state.
                    metacommunity_regions += regions.tolist()
                    metacommunity_regions = list(set(metacommunity_regions))

                    if args.validate:
                        # keep the region x pwm matrix (optionally) to visually show that all regions in group have motif
                        # do this per task
                        pass

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


        # a procedure for increasong community membership:
        # NOTE: the below may not be necessary or helpful? you lose some edges but may be better to be more conservative in approach
        # the means of the enumerated communities (mean vector: motif x task)
        # are the seed points
        # for each enumerated community, calculate jaccard from the mean vector
        # to all others in community. this gives you a dist x task for each example.
        # calculate a covariance matrix, and throw the mean and covariance into multivariate normal

        # now for each unassigned region, get dists to each community
        # see which groups it gets assigned to.
        # assign to the community it is closest to.
        
