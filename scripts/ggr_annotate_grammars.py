#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import re
import sys
import glob
import logging
import argparse

import numpy as np
import pandas as pd
import networkx as nx

from scipy.stats import pearsonr
from scipy.stats import zscore

from tronn.interpretation.networks import get_bed_from_nx_graph
from tronn.interpretation.networks import stringize_nx_graph
from tronn.util.pwms import MotifSetManager
from tronn.util.scripts import setup_run_logs


# manually curated pwms to blacklist
MANUAL_BLACKLIST_PWMS = [
    "SMARC",
    "ZFP82",
    "ZNF667",
    "ZNF547",
    "ZNF317",
    "ZNF322"]

# manually curated genes of interest
MANUAL_INTERESTING_GENES = []

# manually curated functional terms
GOOD_GO_TERMS = [
    "stem cell differentiation",
    "hemidesmosome",
    "hair",
    "cell migration",
    "skin",
    "keratinocyte",
    "cell cycle",
    "epiderm",
    "cell junction",
    "cell proliferation",
    "adhesion",
    "lipase activity",
    "fatty acid",
    "sphingomyelin",
    "glycerolipid"]


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # required args
    parser.add_argument(
        "--grammars", nargs="+",
        help="all gml files to be annotated and compiled")
    parser.add_argument(
        "--tss",
        help="BED file of TSS positions with gene IDs in the name column")
    parser.add_argument(
        "--foreground_rna",
        help="gene set for linking grammars")
    parser.add_argument(
        "--background_rna",
        help="all genes expressed (as background for GO enrichment)")

    # pwm useful stuff
    parser.add_argument(
        "--pwms",
        help="pwm file to remove long pwms from analysis")
    parser.add_argument(
        "--pwm_metadata",
        help="pwm metadata to get gene IDs of interest")
    parser.add_argument(
        "--max_pwm_length", default=20, type=int,
        help="cutoff for removing long pwms")
    
    # out
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    
    # parse args
    args = parser.parse_args()

    return args

def get_max_delta_logit(gml_file):
    max_val = 0
    with open(gml_file, "r") as fp:
        for line in fp:
            if "deltalogits" in line:
                delta_logits = line.strip().split("deltalogits ")[1]
                delta_logits = delta_logits.replace("\"", "").split(",")
                delta_logits = [float(val) for val in delta_logits]
                max_delta_logit = np.min(delta_logits)
                #max_delta_logit = np.min(delta_logits[12:15])
                if max_delta_logit < max_val:
                    max_val = max_delta_logit
                    
    return max_val


def get_atac_signal(gml_file):
    max_signal = 0
    with open(gml_file, "r") as fp:
        for line in fp:
            if "ATACSIGNALSNORM" in line:
                signals = line.strip().split("ATACSIGNALSNORM ")[1]
                signals = signals.replace("\"", "").split(",")
                signals = [float(val) for val in signals]
                max_signal = np.max(signals)
                break

    return max_signal


def _run_great(bed_file, out_dir):
    """run great
    """
    run_cmd = "run_rgreat.R {} {}/{}".format(
        bed_file,
        out_dir,
        os.path.basename(bed_file).split(".bed")[0])
    print run_cmd
    os.system('GREPDB="{}"; /bin/bash -c "$GREPDB"'.format(run_cmd))
    
    return None


def _run_david(foreground_genes, background_genes, out_dir):
    """run david
    """
    # run david
    results_file = "{}/{}.rna.foreground.go_rdavid.GOTERM_BP_ALL.txt".format(
        out_dir, os.path.basename(foreground_genes).split(".rna.foreground")[0])
    if not os.path.isfile(results_file):
        run_david = "/users/dskim89/git/ggr-project/R/bioinformatics.go.rdavid.R {} {} {}".format(
            foreground_genes,
            background_genes,
            out_dir)
        print run_david
        os.system(run_david)
    
    # read in
    try:
        david_results = pd.read_table(results_file)
        if david_results.shape[0] > 10:
            david_results = david_results.iloc[0:10]
        functional_terms = david_results["Term"].values.tolist()
    except:
        functional_terms = []
            
    return functional_terms


def _run_gprofiler(foreground_genes, background_genes, out_dir):
    """run david
    """
    # run gprofiler
    results_file = "{}/{}.rna.foreground.go_gprofiler.txt".format(
        out_dir, os.path.basename(foreground_genes).split(".rna.foreground")[0])
    if not os.path.isfile(results_file):
        run_gprofiler = "/users/dskim89/git/ggr-project/R/bioinformatics.go.gProfileR.R {} {} {}".format(
            foreground_genes,
            background_genes,
            out_dir)
        print run_gprofiler
        os.system(run_gprofiler)

    # read in
    try:
        gprofiler_results = pd.read_table(results_file)
        gprofiler_results = gprofiler_results[gprofiler_results["domain"] == "BP"]
        functional_terms = gprofiler_results["term.name"].values.tolist()
    except:
        functional_terms = []
        
    return functional_terms


def get_functional_enrichment(foreground_genes, background_genes, out_dir, method="david"):
    """wrapper around different functional enrichment tools
    """
    if method == "david":
        results = _run_david(foreground_genes, background_genes, out_dir)
    elif method == "gprofiler":
        results = _run_gprofiler(foreground_genes, background_genes, out_dir)

    return results


def _expand_pwms_by_rna(pwms_df, split_cols, split_char=";"):
    """given column(s) with multiple values in a row, expand so that
    each row has a single value

    assumes that multiple split_cols are ordered same way
    """
    # make the duplicate values into lists
    for split_col in split_cols:
        pwms_df = pwms_df.assign(
            **{split_col: pwms_df[split_col].str.split(split_char)})

    # repeat rows based on list lengths
    pwms_df_expanded = pd.DataFrame({
        col: np.repeat(pwms_df[col].values, pwms_df[split_cols[0]].str.len())
        for col in pwms_df.columns})    

    # put a unique value into each row
    for split_col in split_cols:
        pwms_df_expanded = pwms_df_expanded.assign(
            **{split_col: np.concatenate(pwms_df[split_col].values)})[
                pwms_df.columns.tolist()]

    return pwms_df_expanded


def get_nearby_genes(
        bed_file,
        tss_file,
        k=2,
        max_dist=500000,
        tmp_file="tss_overlap.tmp.txt"):
    """get proximal genes
    """
    # bedtools closest
    closest = "bedtools closest -d -k {} -a {} -b {} > {}".format(
        k, bed_file, tss_file, tmp_file)
    os.system(closest)

    # load results and use distance cutoff
    data = pd.read_table(tmp_file, header=None)
    data = data[data[9] < max_dist]
    data = data[[6,9]]
    data.columns = ["gene_id", "distance"]
    data = data.set_index("gene_id")
    
    return data


def check_substrings(desired_substrings, main_strings):
    for main_string in main_strings:
        for substring in desired_substrings:
            if substring in main_string:
                return True
    return False


def get_num_lines(file_name):
    num_lines = 0
    with open(file_name, "r") as fp:
        for line in fp:
            num_lines += 1

    return num_lines


def annotate_grammars(args):
    """all the grammar annotation stuff
    """
    # generate blacklist pwms (length cutoff + manual)
    pwms = MotifSetManager.read_pwm_file(args.pwms)
    blacklist_pwms = []
    for pwm in pwms:
        if pwm.weights.shape[1] > args.max_pwm_length:
            blacklist_pwms.append(pwm.name)
    blacklist_pwms += MANUAL_BLACKLIST_PWMS
    logging.info("Blacklist pwms: {}".format(";".join(blacklist_pwms)))

    # read in dynamic genes
    dynamic_genes = pd.read_table(args.foreground_rna, index_col=0)
    
    # get list of interesting genes
    pwm_metadata = pd.read_table(args.pwm_metadata).dropna()
    pwm_to_rna_dict = dict(zip(
        pwm_metadata["hclust_model_name"].values.tolist(),
        pwm_metadata["expressed_hgnc"].values.tolist()))
    split_cols = ["expressed", "expressed_hgnc"]
    pwm_metadata = _expand_pwms_by_rna(pwm_metadata, split_cols)
    genes_w_pwms = dict(zip(
        pwm_metadata["expressed"].values.tolist(),
        pwm_metadata["expressed_hgnc"].values.tolist()))
    interesting_genes = genes_w_pwms
    # TODO filter these for just the dynamic ones
    for key in interesting_genes.keys():
        pass
    logging.info("Expressed genes with matching pwms: {}".format(len(genes_w_pwms)))
        
    # load grammars and adjust as needed
    args.grammars = sorted(args.grammars)
    grammars = [nx.read_gml(grammar) for grammar in args.grammars]
    for grammar in grammars:
        grammar.graph["examples"] = set(
            grammar.graph["examples"].split(","))
    logging.info("Starting with {} grammars".format(len(grammars)))

    # for each grammar, run analyses:
    nodes = [] # names of motifs in grammar
    rna_nodes = [] # names of TFs that match the motifs
    atac_signals = [] # max ATAC signal
    delta_logits = [] # max delta logit
    filt_grammar_files = [] # filtered list of grammar file names
    num_regions = [] # number of regions
    num_proximal_genes = [] # number of proximal genes
    rna_mean_vals = [] # max of mean vals of proximal genes
    region_to_rna_ratios = [] # region to proximal gene ratio
    mse_results = [] # weighted (by distance) MSE
    downstream_interesting = [] # nearby genes that are interesting
    functionally_enriched = [] # functional enrichment
    
    global_grammar_idx = 0
    for grammar_idx in xrange(len(grammars)):

        # get grammar file and grammar graph
        grammar_file = args.grammars[grammar_idx]
        grammar = grammars[grammar_idx]
        logging.debug("annotating {} with nodes {}".format(
            grammar_file, ";".join(grammar.nodes)))
        
        # ignore single node (not a grammar)
        if len(grammar.nodes) < 2:
            logging.info("skipping since 1 node")
            continue
        
        # immediately ignore blacklist
        node_names = list(grammar.nodes)
        if check_substrings(blacklist_pwms, node_names):
            logging.info("skipping since grammar contains blacklist motif")
            continue

        # after filtering, attach name
        clean_node_names = re.sub("HCLUST-\d+_", "", ",".join(grammar.nodes)).replace(".UNK.0.A", "")
        nodes.append(clean_node_names)
        # TODO ideally adjust for only the RNAs that had matched the pattern
        rna_node_names = ",".join([pwm_to_rna_dict[node_name] for node_name in grammar.nodes])
        rna_nodes.append(rna_node_names)
        
        # make a BED file
        grammar_bed = "{}/{}.bed".format(
            args.out_dir, os.path.basename(grammar_file).split(".gml")[0])
        get_bed_from_nx_graph(grammar, grammar_bed)
        num_regions.append(get_num_lines(grammar_bed))
        atac_signals.append(get_atac_signal(grammar_file))
        delta_logits.append(get_max_delta_logit(grammar_file))
        
        # run proximal RNA based analyses
        # get nearby genes
        max_dist = 500000
        nearby_genes = get_nearby_genes(grammar_bed, args.tss, k=2, max_dist=max_dist)
        logging.info("proximal genes within {} bp: {}".format(max_dist, nearby_genes.shape[0]))
        
        # intersect with foreground set (in our case, dynamic genes only)
        nearby_dynamic_genes = dynamic_genes.merge(nearby_genes, left_index=True, right_index=True)
        logging.info("proximal genes that are DYNAMIC: {}".format(nearby_dynamic_genes.shape[0]))

        # and filter vs the ATAC signal pattern
        atac_pattern = np.array(
            [float(val) for val in grammar.graph["ATACSIGNALSNORM"].split(",")])
        atac_pattern = atac_pattern[[0,2,3,4,5,6,9,10,12]] # remove media timepoints
        nearby_dynamic_genes["corr"] = nearby_dynamic_genes.iloc[:,:-1].apply(
            lambda x: pearsonr(x, atac_pattern)[0], axis=1)
        nearby_dynamic_genes["corr_pval"] = nearby_dynamic_genes.iloc[:,:-2].apply(
            lambda x: pearsonr(x, atac_pattern)[1], axis=1)
        nearby_dynamic_genes_atac_filt = nearby_dynamic_genes[nearby_dynamic_genes["corr"] > 0]
        nearby_dynamic_genes_atac_filt = nearby_dynamic_genes_atac_filt[
            nearby_dynamic_genes_atac_filt["corr_pval"] < 0.10]
        nearby_dynamic_genes_atac_filt = nearby_dynamic_genes_atac_filt.sort_values("corr", ascending=False)
        num_proximal_genes.append(nearby_dynamic_genes_atac_filt.shape[0])
        logging.info("proximal genes that are well correlated to ATAC signal: {}".format(
            nearby_dynamic_genes_atac_filt.shape[0]))

        # get the region to rna ratio
        region_to_rna_ratios.append(num_regions[-1] / float(num_proximal_genes[-1]))

        # check for relevant downstream genes
        interesting_downstream_genes = sorted(list(
            set(interesting_genes.keys()).intersection(
                set(nearby_dynamic_genes_atac_filt.index))))
        hgnc_interesting = [
            interesting_genes[ensembl_id]
            for ensembl_id in interesting_downstream_genes]
        downstream_interesting.append(",".join(hgnc_interesting))

        # set up zscores (for a weighted MSE and for plotting), attach max val and distance
        rna_z = zscore(nearby_dynamic_genes_atac_filt.values[:,:-3], axis=1)
        rna_z = pd.DataFrame(
            rna_z,
            index=nearby_dynamic_genes_atac_filt.index,
            columns=nearby_dynamic_genes_atac_filt.columns[:-3])
        rna_z["max"] = np.max(nearby_dynamic_genes_atac_filt.values[:,:-3], axis=1)
        rna_z["distance"] = nearby_dynamic_genes_atac_filt["distance"]
        # adjust distances with ENSG id to make sure unique
        decimals = "." + rna_z.reset_index()["index"].str.split("ENSG000", n=1, expand=True)[1]
        rna_z["decimal_id"] = decimals.values.astype(float)
        rna_z["distance"] = rna_z["distance"] + rna_z["decimal_id"]
        del rna_z["decimal_id"]
        # and now convert distances to a decay value
        coeff = -np.log(0.5) / (max_dist / 8.)
        rna_z["weights"] = 1 * np.exp(-coeff * rna_z["distance"].values)
        
        # get a weighted MSE (with a weighted mean average)
        rna_z_array = rna_z.values[:,:-3]
        distance_weights = rna_z.values[:,-1]
        weighted_patterns = np.multiply(
            rna_z_array,
            np.expand_dims(distance_weights, axis=1))
        weighted_pattern_sum = np.sum(weighted_patterns, axis=0)
        weighted_mean = weighted_pattern_sum / distance_weights.sum()
        grammar.graph["RNASIGNALS"] = weighted_mean.tolist() # saving out RNA vector
        weighted_mean = np.expand_dims(weighted_mean, axis=0)
        
        # get weighted MSE
        mean_sq_errors = np.square(
            np.subtract(rna_z_array, weighted_mean)).mean(axis=1)
        weighted_mean_sq_errors = np.multiply(
            mean_sq_errors,
            distance_weights)
        weighted_MSE = np.sum(weighted_mean_sq_errors) / distance_weights.sum()
        mse_results.append(weighted_MSE)
        logging.info("{}".format(weighted_MSE))

        # get a weighted mean but NOT normalized
        weighted_orig_rna = np.multiply(
            nearby_dynamic_genes_atac_filt.values[:,:-3],
            np.expand_dims(distance_weights, axis=1))
        weighted_orig_mean = np.sum(weighted_orig_rna, axis=0) / distance_weights.sum()
        rna_mean_vals.append(np.max(weighted_orig_mean))
        
        # adjust for plotting - just plot out the CLOSEST ones
        plotting = False
        if plotting:
            pass

        # run functional enrichment using these genes
        foreground_gene_file = "{}/{}.rna.foreground.txt.gz".format(
            args.out_dir, os.path.basename(grammar_bed).split(".bed")[0])
        nearby_dynamic_genes_atac_filt.reset_index().to_csv(
            foreground_gene_file, columns=["index"], header=False, index=False, compression="gzip")
        functional_terms = get_functional_enrichment(
            foreground_gene_file, args.background_rna, args.out_dir, method="gprofiler")
        if check_substrings(GOOD_GO_TERMS, functional_terms):
            logging.info("was functionally enriched: {}".format(",".join(functional_terms)))
            functionally_enriched.append(1)
        else:
            functionally_enriched.append(0)

        # and save out updated gml file
        new_grammar_file = "{}/{}.annot-{}.gml".format(args.out_dir, os.path.basename(grammar_file).split(".gml")[0], grammar_idx)
        nx.write_gml(stringize_nx_graph(grammar), new_grammar_file)
        filt_grammar_files.append(new_grammar_file)

    # save out summary
    summary_file = "{}/grammar_summary.txt".format(args.out_dir)
    data = {
        "filename": filt_grammar_files,
        "nodes": nodes,
        "nodes_rna": rna_nodes,
        "MSE": mse_results,
        "ATAC_signal": atac_signals,
        "delta_logit": delta_logits,
        "GO_terms": functionally_enriched,
        "region_to_rna": region_to_rna_ratios,
        "region_num": num_regions,
        "num_target_genes": num_proximal_genes,
        "max_rna_vals": rna_mean_vals,
        "downstream_interesting": downstream_interesting}
    summary_df = pd.DataFrame(data)
    summary_df = summary_df.sort_values("downstream_interesting")
    summary_df.to_csv(summary_file, sep="\t")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print summary_df

        #results_filt = summary_df.loc[summary_df["DAVID"] == 1]
        #results_filt = results_filt.loc[results_filt["blacklist"] != 1]
        #print results_filt.drop(columns=["filename"]).sort_values("delta_logit")
        #print results_filt.drop(columns=["filename"]).sort_values("region_num")
        #print results_filt.shape[0]

    logging.info("Filtered grammar total: {}".format(summary_df.shape[0]))

    # save out a filtered summary
    filt_summary_file = "{}/grammar_summary.filt.txt".format(args.out_dir)
    summary_df.loc[summary_df["GO_terms"] == 1].to_csv(filt_summary_file, sep="\t")

    return filt_summary_file


def plot_results(filt_summary_file, out_dir):
    """pull out matrices for plotting. need:
    1) motif presence in each grammar
    2) ATAC pattern for each grammar
    3) RNA pattern for each grammar
    """
    grammars_df = pd.read_table(filt_summary_file)

    # remove the lowest signals
    vals = grammars_df["ATAC_signal"].values
    print np.min(vals), np.max(vals)
    grammars_df = grammars_df[grammars_df["ATAC_signal"] > 2.85]

    # remove the poorest performers
    vals = grammars_df["delta_logit"].values
    print np.min(vals), np.max(vals)
    grammars_df = grammars_df[grammars_df["delta_logit"] < -0.09] # 0.8

    # don't mess with RNA
    vals = grammars_df["max_rna_vals"].values
    print np.min(vals), np.max(vals)

    # get number of grammars
    num_grammars = grammars_df.shape[0]

    # adjust ordering
    num_to_order_val = {
        0:1,
        1:10,
        2:11,
        3:12,
        4:13,
        5:14,
        7:2,
        8:3,
        9:4,
        10:5,
        11:6,
        12:7,
        13:8,
        14:9
    }
    
    for line_idx in range(num_grammars):
        grammar = nx.read_gml(grammars_df["filename"].iloc[line_idx])
        grammar_traj = int(
            grammars_df["filename"].iloc[line_idx].split("/")[1].split(".")[1].split("-")[1])
        
        # get motifs and merge into motif df
        motifs = grammars_df["nodes_rna"].iloc[line_idx].split(",")
        motif_presence = pd.DataFrame(
            num_to_order_val[grammar_traj]*np.ones(len(motifs)),
            index=motifs,
            columns=[grammars_df["nodes"].iloc[line_idx]])
        if line_idx == 0:
            motifs_all = motif_presence
        else:
            motifs_all = motifs_all.merge(
                motif_presence,
                left_index=True,
                right_index=True,
                how="outer")
            motifs_all = motifs_all.fillna(0)
            
        # extract RNA vector and append
        rna = [float(val) for val in grammar.graph["RNASIGNALS"].split(",")]
        if line_idx == 0:
            rna_all = np.zeros((num_grammars, len(rna)))
            rna_all[line_idx] = rna
        else:
            rna_all[line_idx] = rna

        # extract ATAC vector and append
        epigenome_signals = np.array([float(val) for val in grammar.graph["ATACSIGNALSNORM"].split(",")])
        atac = epigenome_signals[[0,2,3,4,5,6,9,10,12]]
        if line_idx == 0:
            atac_all = np.zeros((num_grammars, atac.shape[0]))
            atac_all[line_idx] = atac
        else:
            atac_all[line_idx] = atac

    # transpose motifs matrix
    motifs_all["rank"] = np.sum(motifs_all.values, axis=1) / np.sum(motifs_all.values != 0, axis=1)
    motifs_all = motifs_all.sort_values("rank")
    del motifs_all["rank"]
    motifs_all = (motifs_all > 0).astype(int)
    motifs_all = motifs_all.transpose()
    
    # zscore the ATAC data
    atac_all = zscore(atac_all, axis=1)

    # convert others to df
    atac_df = pd.DataFrame(atac_all)
    rna_df = pd.DataFrame(rna_all)
    
    # remove duplicates
    # TODO this should happen earlier - figure out how to correctly merge things
    if True:
        #dup_indices = motifs_all[motifs_all.duplicated()].index.values
        dup_indices = np.where(motifs_all.duplicated().values)[0]
        #print dup_indices
        #print motifs_all.index[dup_indices]
        
        motifs_all = motifs_all.drop(motifs_all.index[dup_indices], axis=0)
        atac_df = atac_df.drop(atac_df.index[dup_indices], axis=0)
        rna_df = rna_df.drop(rna_df.index[dup_indices], axis=0)
        print motifs_all.shape
        print atac_df.shape
        print rna_df.shape

        
    if False:
        # remove solos from motifs and adjust all matrices accordingly
        motif_indices = np.where(np.sum(motifs_all, axis=0) <= 1)[0]
        motifs_all = motifs_all.drop(motifs_all.columns[motif_indices], axis=1)
        orphan_grammar_indices = np.where(np.sum(motifs_all, axis=1) <= 1)[0]
        motifs_all = motifs_all.drop(motifs_all.index[orphan_grammar_indices], axis=0)
        atac_df = atac_df.drop(atac_df.index[orphan_grammar_indices], axis=0)
        rna_df = rna_df.drop(rna_df.index[orphan_grammar_indices], axis=0)
        print motifs_all.shape
        print atac_df.shape
        print rna_df.shape
        

    motifs_file = "{}/grammars.filt.motif_presence.mat.txt".format(out_dir)
    motifs_all.to_csv(motifs_file, sep="\t")

    atac_file = "{}/grammars.filt.atac.mat.txt".format(out_dir)
    atac_df.to_csv(atac_file, sep="\t")

    rna_file = "{}/grammars.filt.rna.mat.txt".format(out_dir)
    rna_df.to_csv(rna_file, sep="\t")
    
    # run R script
    
    
    return


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    setup_run_logs(args, os.path.basename(sys.argv[0]).split(".py")[0])

    # annotate grammars
    filt_summary_file = "{}/grammar_summary.filt.txt".format(args.out_dir)
    if not os.path.isfile(filt_summary_file):
        filt_summary_file = annotate_grammars(args)

    # generate matrices for plotting
    # NOTE removed solo motif grammars
    plot_results(filt_summary_file, args.out_dir)
    
    return None



if __name__ == "__main__":
    main()
