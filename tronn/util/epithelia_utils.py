# description: GGR specific util functions



import os
import re
import sys
import h5py
import glob
import logging
import argparse

import numpy as np
import pandas as pd
import networkx as nx

from scipy.stats import pearsonr
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, leaves_list

from tronn.interpretation.networks import get_bed_from_nx_graph
from tronn.interpretation.networks import stringize_nx_graph
from tronn.util.h5_utils import AttrKeys
from tronn.util.pwms import MotifSetManager
from tronn.util.scripts import setup_run_logs
from tronn.util.utils import DataKeys

# manually curated pwms to blacklist
MANUAL_BLACKLIST_PWMS = [
    "NANOG",
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
    "neuro",
    "axo",
    "dendrit",
    "pigment",
    "melanin",
    "melanocyte",
    "melanosome",
    "immune",
    "cytokine",
    "interleukin",
    "hematopoietic",
    "hemo",
    "B cell"
]

REMOVE_GO_TERMS = [
    "adaptive immune response based on ",
    "T cell",
    "of chemotaxis",
    "cell chemotaxis",
    "myeloid dendritic cell activation",
    "positive regulation of neuron projection development",
    "cell morphogenesis involved in neuron differentiation",
    "positive regulation of",
    "lymphocyte activation involved in"
    #"ameboidal",
    #"calcium-independent cell-cell adhesion via plasma membrane cell-adhesion molecules",
    #"spindle",
    #"via plasma membrane",
    #"adhesion molecules",
    #"cell spreading",
    #"endothelial",
    #"muscle"
]

REMOVE_EXACT_TERMS = [
    "immune system process",
    "adaptive immune response",
    "immune response regulation of immune system process",
    "regulation of immune response",
    "regulation of B cell proliferation",
    "hematopoietic or lymphoid organ development",
    "B cell activation involved in immune response",
    "regulation of neuron projection development",
    "central nervous system neuron differentiation",
    "central nervous system neuron development",
    "regulation of interleukin 4 production",
    "lymphocyte chemotaxis",
    "regulation of interleukin 2 production",
    "interleukin 2 production",
    "dendritic cell differentiation",
    "axon ensheathment",
    "humoral immune response"
    #"biological adhesion",
    #"positive regulation of cell migration",
    #"positive regulation of cell junction assembly",
    #"positive regulation of epithelial cell proliferation",
    #"regulation of epithelial cell proliferation",
    #"calcium-independent cell-matrix adhesion",
    #"positive regulation of cell proliferation",
    #"regulation of cell proliferation",
    #"positive regulation of cell adhesion",
    #"regulation of cell adhesion",
    #"regulation of cell migration",
    #"regulatino of cell substrate adhesion"
]

KEEP_GRAMMARS = [
]

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


def get_edge_type(gml_file):
    edge_type = "co_occurrence"
    with open(gml_file, "r") as fp:
        for line in fp:
            if "edgetype" in line:
                if "directed" in line:
                    edge_type = "directed"
                    break

    return edge_type


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
        run_gprofiler = "bioinformatics.go.gProfileR.R {} {} {}".format(
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


def expand_pwms_by_rna(pwms_df, split_cols, split_char=";"):
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
        k,
        max_dist,
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

    # cleanup
    os.system("rm {}".format(tmp_file))
    
    return data


def check_substrings(desired_substrings, main_strings):
    for main_string in main_strings:
        for substring in desired_substrings:
            if substring in main_string:
                return True
    return False


def get_substring_matches(desired_substrings, main_strings):
    matches = []
    for main_string in main_strings:
        for substring in desired_substrings:
            if substring in main_string:
                matches.append(main_string)

    return matches


def get_num_lines(file_name):
    num_lines = 0
    with open(file_name, "r") as fp:
        for line in fp:
            num_lines += 1

    return num_lines


def annotate_one_grammar(
        grammar_file,
        grammar,
        new_grammar_file,
        pwm_to_rna_dict,
        tss,
        dynamic_genes,
        interesting_genes,
        background_rna,
        out_dir,
        k_closest=2,
        max_dist=250000): # TODO adjust here - Nat gen GTEX paper, use 250kbp?
    """return a dict of results
    """
    results = {}
    
    # attach motif names
    clean_node_names = re.sub("HCLUST-\d+_", "", ",".join(grammar.nodes)).replace(".UNK.0.A", "")
    results["nodes"] = clean_node_names

    # attach TF names
    # TODO ideally adjust for only the RNAs that had matched the pattern (good correlation)
    rna_node_names = ",".join(
        [pwm_to_rna_dict[node_name] for node_name in grammar.nodes])
    results["nodes_rna"] = rna_node_names
    
    # make a BED file, get num regions
    grammar_bed = "{}/{}.bed".format(
        out_dir, os.path.basename(grammar_file).split(".gml")[0])
    get_bed_from_nx_graph(grammar, grammar_bed)
    results["region_num"] = get_num_lines(grammar_bed)

    # get signal and delta scores
    results["ATAC_signal"] = get_atac_signal(grammar_file)
    results["delta_logit"] = get_max_delta_logit(grammar_file)
    results["edge_type"] = get_edge_type(grammar_file)

    # run proximal RNA based analyses
    # get nearby genes
    nearby_genes = get_nearby_genes(grammar_bed, tss, k_closest, max_dist)
    logging.info("proximal genes within {} bp: {}".format(max_dist, nearby_genes.shape[0]))

    # intersect with foreground set (in our case, dynamic genes only)
    nearby_dynamic_genes = dynamic_genes.merge(nearby_genes, left_index=True, right_index=True)
    logging.info("proximal genes that are DYNAMIC: {}".format(nearby_dynamic_genes.shape[0]))
    
    # and filter vs the ATAC signal pattern
    atac_pattern = np.array(
        [float(val) for val in grammar.graph["ATACSIGNALSNORM"].split(",")])
    #atac_pattern = atac_pattern[[0,2,3,4,5,6,9,10,12]] # remove media timepoints
    logging.info("NOTE EPITHELIA HACK HERE")
    atac_pattern = atac_pattern[[1,2,3,4,5,6,7,8,9,10,11,12,13,14]] # remove media timepoints
    nearby_dynamic_genes["corr"] = nearby_dynamic_genes.iloc[:,:-1].apply(
        lambda x: pearsonr(x, atac_pattern)[0], axis=1)
    nearby_dynamic_genes["corr_pval"] = nearby_dynamic_genes.iloc[:,:-2].apply(
        lambda x: pearsonr(x, atac_pattern)[1], axis=1)
    nearby_dynamic_genes_atac_filt = nearby_dynamic_genes[nearby_dynamic_genes["corr"] > 0]
    nearby_dynamic_genes_atac_filt = nearby_dynamic_genes_atac_filt[
        nearby_dynamic_genes_atac_filt["corr_pval"] < 0.10]
    nearby_dynamic_genes_atac_filt = nearby_dynamic_genes_atac_filt.sort_values("corr", ascending=False)
    results["num_target_genes"] = nearby_dynamic_genes_atac_filt.shape[0]
    logging.info("proximal genes that are well correlated to ATAC signal: {}".format(
        nearby_dynamic_genes_atac_filt.shape[0]))

    # if no target genes, clear out
    if results["num_target_genes"] == 0:
        results["region_to_RNA"] = 0.0
        results["downstream_interesting"] = ""
        results["MSE"] = 0.0
        results["max_rna_vals"] = 0.0
        results["GO_terms"] = 0
        results["filename"] = "None"
        return results
        
    # get the region to rna ratio
    results["region_to_rna"] = results["region_num"] / float(results["num_target_genes"])
        
    # check for relevant downstream genes
    interesting_downstream_genes = sorted(list(
        set(interesting_genes.keys()).intersection(
            set(nearby_dynamic_genes_atac_filt.index))))
    hgnc_interesting = [
        interesting_genes[ensembl_id]
        for ensembl_id in interesting_downstream_genes]
    results["downstream_interesting"] = ",".join(hgnc_interesting)

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
    results["MSE"] = weighted_MSE
    logging.info("{}".format(weighted_MSE))

    # get a weighted mean but NOT normalized
    weighted_orig_rna = np.multiply(
        nearby_dynamic_genes_atac_filt.values[:,:-3],
        np.expand_dims(distance_weights, axis=1))
    weighted_orig_mean = np.sum(weighted_orig_rna, axis=0) / distance_weights.sum()
    results["max_rna_vals"] = np.max(weighted_orig_mean)

    # adjust for plotting - just plot out the CLOSEST ones
    plotting = False
    if plotting:
        pass

    # run functional enrichment using these genes
    foreground_gene_file = "{}/{}.rna.foreground.txt.gz".format(
        out_dir, os.path.basename(grammar_bed).split(".bed")[0])
    nearby_dynamic_genes_atac_filt.reset_index().to_csv(
        foreground_gene_file, columns=["index"], header=False, index=False, compression="gzip")
    functional_terms = get_functional_enrichment(
        foreground_gene_file, background_rna, out_dir, method="gprofiler")
    if check_substrings(GOOD_GO_TERMS, functional_terms):
        logging.info("was functionally enriched: {}".format(",".join(functional_terms)))
        results["GO_terms"] = 1
        results["GO_descriptions"] = ",".join(
            get_substring_matches(GOOD_GO_TERMS, functional_terms)).replace(" ", "_")
    else:
        results["GO_terms"] = 0
        results["GO_descriptions"] = "NA"

    # check keep grammars
    for keep_grammar in KEEP_GRAMMARS:
        if len(set(clean_node_names.split(",")).difference(set(keep_grammar))) == 0:
            results["GO_terms"] = 1
        
    # and save out updated gml file
    nx.write_gml(stringize_nx_graph(grammar), new_grammar_file)
    results["filename"] = new_grammar_file

    return results


def merge_graph_attrs(main_graph, merge_graph, key, merge_type="mean"):
    """merge graph attributes
    """
    if merge_type == "mean":
        if not isinstance(main_graph.graph[key], np.ndarray):
            # main graph
            main_graph_attr = np.array(
                [float(val) for val in main_graph.graph[key].split(",")])
        else:
            main_graph_attr = main_graph.graph[key]
        main_graph_examples = main_graph.graph["numexamples"]
        # merge graph
        merge_graph_attr = np.array(
            [float(val) for val in merge_graph.graph[key].split(",")])
        merge_graph_examples = merge_graph.graph["numexamples"]
        # merge
        merged_attr = np.divide(
            np.add(
                np.multiply(main_graph_examples, main_graph_attr),
                np.multiply(merge_graph_examples, merge_graph_attr)),
            np.add(main_graph_examples, merge_graph_examples))
        main_graph.graph[key] = merged_attr
    elif merge_type == "str_concat":
        main_graph.graph[key] = "{},{}".format(main_graph.graph[key], merge_graph.graph[key])
    elif merge_type == "sum":
        main_graph.graph[key] = main_graph.graph[key] + merge_graph.graph[key]

    return main_graph


def merge_multiple_node_attrs(main_graph, merge_graph, key, merge_type="mean"):
    """merge multiple
    """
    nodes = list(main_graph.nodes)
    for node in nodes:
        merge_node_attrs(main_graph, merge_graph, node, key, merge_type=merge_type)
    
    return main_graph


def merge_node_attrs(main_graph, merge_graph, node, key, merge_type="mean"):
    """
    """
    if merge_type == "mean":
        if not isinstance(main_graph.node[node][key], np.ndarray):
            # main graph
            main_graph_attr = np.array(
                [float(val) for val in main_graph.node[node][key].split(",")])
        else:
            main_graph_attr = main_graph.node[node][key]
        main_graph_examples = main_graph.node[node]["numexamples"]
        # merge graph
        merge_graph_attr = np.array(
            [float(val) for val in merge_graph.node[node][key].split(",")])
        merge_graph_examples = merge_graph.node[node]["numexamples"]
        # merge
        merged_attr = np.divide(
            np.add(
                np.multiply(main_graph_examples, main_graph_attr),
                np.multiply(merge_graph_examples, merge_graph_attr)),
            np.add(main_graph_examples, merge_graph_examples))
        main_graph.node[node][key] = merged_attr
    elif merge_type == "str_concat":
        main_graph.node[node][key] = "{},{}".format(
            main_graph.node[node][key], merge_graph.node[node][key])
    elif merge_type == "sum":
        main_graph.node[node][key] = main_graph.node[node][key] + merge_graph.node[node][key]

    return main_graph


def merge_multiple_edge_attrs(main_graph, merge_graph, key, merge_type="mean"):
    """merge multiple
    """
    edges = list(main_graph.edges)
    for edge in edges:
        try:
            merge_edge_attrs(main_graph, merge_graph, edge, edge, key, merge_type=merge_type)
        except KeyError as e:
            edge_flipped = (edge[1], edge[0], edge[2])
            merge_edge_attrs(main_graph, merge_graph, edge, edge_flipped, key, merge_type=merge_type)
    
    return main_graph


def merge_edge_attrs(
        main_graph, merge_graph, main_edge, merge_edge, key, merge_type="mean"):
    """
    """
    if merge_type == "mean":
        if not isinstance(main_graph.edges[main_edge][key], np.ndarray):
            # main graph
            main_graph_attr = np.array(
                [float(val) for val in main_graph.edges[main_edge][key].split(",")])
        else:
            main_graph_attr = main_garph.edges[main_edge][key]
        main_graph_examples = main_graph.edges[main_edge]["numexamples"]
        # merge graph
        merge_graph_attr = np.array(
            [float(val) for val in merge_graph.edges[merge_edge][key].split(",")])
        merge_graph_examples = merge_graph.edges[merge_edge]["numexamples"]
        # merge
        merged_attr = np.divide(
            np.add(
                np.multiply(main_graph_examples, main_graph_attr),
                np.multiply(merge_graph_examples, merge_graph_attr)),
            np.add(main_graph_examples, merge_graph_examples))
        main_graph.edges[main_edge][key] = merged_attr
    elif merge_type == "str_concat":
        main_graph.edges[main_edge][key] = "{},{}".format(
            main_graph.edges[main_edge][key], merge_graph.edges[merge_edge][key])
    elif merge_type == "sum":
        main_graph.edges[main_edge][key] = main_graph.edges[main_edge][key] + merge_graph.edges[merge_edge][key]

    return main_graph



def merge_grammars_from_files(grammar_files, out_dir):
    """take two gmls and produce a new one
    """
    # read in grammars
    grammars = [nx.read_gml(grammar) for grammar in grammar_files]

    # set up new name
    traj_prefix = "_x_".join(
        [os.path.basename(val).split(".")[1] for val in grammar_files])
    id_prefix = "_x_".join(
        [os.path.basename(val).split(".")[2].split("-")[1] for val in grammar_files])
    prefix = "ggr.{}.{}".format(traj_prefix, id_prefix)
    new_grammar_file = "{}/{}.gml".format(out_dir, prefix)
    print new_grammar_file
    print grammars[0].nodes

    # merge
    for grammar_idx in range(len(grammars)):
        old_grammar = grammars[grammar_idx]
        if grammar_idx == 0:
            new_grammar = old_grammar
        else:
            # merge graph: RNASIGNALS, ATACSIGNALSNORM, logitsnorm, examples, numexamples
            merge_graph_attrs(new_grammar, old_grammar, "RNASIGNALS", merge_type="mean")
            merge_graph_attrs(new_grammar, old_grammar, "ATACSIGNALSNORM", merge_type="mean")
            merge_graph_attrs(new_grammar, old_grammar, "logitsnorm", merge_type="mean")
            merge_graph_attrs(new_grammar, old_grammar, "examples", merge_type="str_concat")
            merge_graph_attrs(new_grammar, old_grammar, "numexamples", merge_type="sum")

            # merge nodes: deltalogits, examples, numexamples
            merge_multiple_node_attrs(
                new_grammar, old_grammar, "deltalogits", merge_type="mean")
            merge_multiple_node_attrs(
                new_grammar, old_grammar, "examples", merge_type="str_concat")
            merge_multiple_node_attrs(
                new_grammar, old_grammar, "numexamples", merge_type="sum")

            # merge edges: examples, numexamples
            merge_multiple_edge_attrs(
                new_grammar, old_grammar, "examples", merge_type="str_concat")
            merge_multiple_edge_attrs(
                new_grammar, old_grammar, "numexamples", merge_type="sum")

    # save out to new name
    nx.write_gml(stringize_nx_graph(new_grammar), new_grammar_file)
            
    return new_grammar_file
        

def merge_synergy_files(
        synergy_dirs,
        curr_grammars,
        merged_grammar_file,
        merged_synergy_dir):
    """merge synergy files into a new dir
    """
    # make dir for merged data
    new_dir = "{}/{}".format(
        merged_synergy_dir,
        os.path.basename(merged_grammar_file).split(".gml")[0])
    os.system("mkdir -p {}".format(new_dir))

    # confirm that all grammars are ordered the same way
    logging.info("confirming all synergy files are ordered the same way")
    for grammar_idx in range(len(curr_grammars)):
        grammar = curr_grammars[grammar_idx]
        print grammar
        grammar_prefix = os.path.basename(grammar).split(".gml")[0]
        
        # get the pwms order file
        order_files = []
        for synergy_dir in synergy_dirs:
            order_file = "{}/{}/ggr.synergy.pwms.order.txt".format(synergy_dir, grammar_prefix)
            if os.path.isfile(order_file):
                order_files.append(order_file)
        assert len(order_files) == 1
        pwm_file = order_files[0]
        pwms = pd.read_table(pwm_file).iloc[:,0].values.tolist()
            
        if grammar_idx == 0:
            master_pwms = pwms
        else:
            assert master_pwms == pwms

    # merge synergy files
    logging.info("merging synergy files")
    new_synergy_file = "{}/ggr.synergy.h5".format(new_dir)
    num_examples = 0
    with h5py.File(new_synergy_file, "w") as out:
        
        for grammar_idx in range(len(curr_grammars)):
            grammar = curr_grammars[grammar_idx]
            print grammar
            grammar_prefix = os.path.basename(grammar).split(".gml")[0]

            # get the synergy file
            synergy_files = []
            for synergy_dir in synergy_dirs:
                synergy_file = "{}/{}/ggr.synergy.h5".format(synergy_dir, grammar_prefix)
                if os.path.isfile(synergy_file):
                    synergy_files.append(synergy_file)
            assert len(synergy_files) == 1
            synergy_file = synergy_files[0]

            # get num examples and keys
            with h5py.File(synergy_file, "r") as hf:
                file_examples = hf[DataKeys.FEATURES].shape[0]
                keys = sorted(hf.keys())
                
                # make new file or concat into the file
                if grammar_idx == 0:
                    for key in keys:
                        resizable_shape = [None] + list(hf[key].shape[1:])
                        out.create_dataset(key, data=hf[key][:], maxshape=resizable_shape)
                else:
                    for key in keys:
                        out[key][num_examples:num_examples+file_examples] = hf[key][:]

            # mark new start
            num_examples += file_examples
    
        # and then reduce file to corect size
        for key in sorted(out.keys()):
            dataset_final_shape = [num_examples] + list(out[key].shape[1:])
            out[key].resize(dataset_final_shape)

        # and finally copy over pwm names
        with h5py.File(synergy_file, "r") as hf:
            out[DataKeys.FEATURES].attrs[AttrKeys.PWM_NAMES] = hf[
                DataKeys.FEATURES].attrs[AttrKeys.PWM_NAMES]
            out[DataKeys.MUT_MOTIF_LOGITS].attrs[AttrKeys.PWM_NAMES] = hf[
                DataKeys.MUT_MOTIF_LOGITS].attrs[AttrKeys.PWM_NAMES]
        
            
    # copy over aux data
    copy_aux = "cp {}/{}/*txt {}/".format(synergy_dirs[0], grammar_prefix, new_dir)
    print copy_aux
    os.system(copy_aux)
    
    return


def merge_duplicates(
        filt_summary_file,
        pwm_to_rna_dict,
        tss,
        dynamic_genes,
        interesting_genes,
        background_rna,
        out_dir,
        synergy_dirs=[],
        merged_synergy_dir=None):
    """for each line, check for duplicate lines and merge them all
    do not adjust df in place, make a new df
    """
    # read in table and sort - that way, can just go straight down the column
    grammars_df = pd.read_table(filt_summary_file, index_col=0)
    grammars_df = grammars_df.sort_values("nodes")

    # set up starting point
    curr_nodes = grammars_df["nodes"].iloc[0]
    curr_grammars = [grammars_df["filename"].iloc[0]]

    # set up results
    results = {
        "nodes": [],
        "nodes_rna": [],
        "ATAC_signal": [],
        "delta_logit": [],
        "filename": [],
        "region_num": [],
        "num_target_genes": [],
        "max_rna_vals": [],
        "region_to_rna": [],
        "MSE": [],
        "downstream_interesting": [],
        "edge_type": [],
        "GO_terms": [],
        "GO_descriptions": []}
    
    line_idx = 0
    while line_idx < grammars_df.shape[0]:
        if line_idx+1 == grammars_df.shape[0]:
            # done, do this to make sure last line gets read out
            next_nodes = "DONE"
        else:
            # check next line
            next_nodes = grammars_df["nodes"].iloc[line_idx+1]
            next_grammar = grammars_df["filename"].iloc[line_idx+1]

        # TODO check that all nodes and edges are same
        
        # if same, add (do all merging at end)
        if next_nodes == curr_nodes:
            curr_grammars.append(next_grammar)
        else:
            # merge and save out to new df
            if len(curr_grammars) > 1:
                curr_grammars = sorted(curr_grammars)
                print [nx.read_gml(grammar).nodes for grammar in curr_grammars]
                print [nx.read_gml(grammar).edges(data="edgetype") for grammar in curr_grammars]
                
                # merge
                merged_grammar_file = merge_grammars_from_files(curr_grammars, out_dir)
                merged_grammar = nx.read_gml(merged_grammar_file)
                merged_grammar.graph["examples"] = set(
                    merged_grammar.graph["examples"].split(","))
                new_grammar_file = "{}/{}".format(
                    out_dir, os.path.basename(merged_grammar_file))
                grammar_results = annotate_one_grammar(
                    merged_grammar_file,
                    merged_grammar,
                    new_grammar_file,
                    pwm_to_rna_dict,
                    tss,
                    dynamic_genes,
                    interesting_genes,
                    background_rna,
                    out_dir)

                # merge synergy files into new dir
                if len(synergy_dirs) > 0:
                    merge_synergy_files(
                        synergy_dirs,
                        curr_grammars,
                        merged_grammar_file,
                        merged_synergy_dir)
                    
            else:
                # just write out the same line again
                grammar_results = dict(zip(
                    grammars_df.columns,
                    grammars_df.iloc[line_idx]))

            # save out
            for key in results.keys():
                results[key].append(grammar_results[key])
                
            # update curr_nodes, curr_grammars
            curr_nodes = next_nodes
            curr_grammars = [next_grammar]
            
        line_idx += 1

    # finally save into new filt file
    summary_df = pd.DataFrame(results)
    summary_df.insert(0, "manual_filt", np.ones(summary_df.shape[0]))
    new_filt_summary_file = "{}/grammar_summary.filt.dedup.txt".format(out_dir)
    summary_df.to_csv(new_filt_summary_file, sep="\t")
    
    return new_filt_summary_file


def annotate_grammars(args, merge_grammars=False):
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
    logging.info("EPITHELIA HACK HERE")
    dynamic_genes = dynamic_genes.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
    
    # get list of interesting genes
    pwm_metadata = pd.read_table(args.pwm_metadata).dropna()
    pwm_to_rna_dict = dict(zip(
        pwm_metadata["hclust_model_name"].values.tolist(),
        pwm_metadata["expressed_hgnc"].values.tolist()))
    split_cols = ["expressed", "expressed_hgnc"]
    pwm_metadata = expand_pwms_by_rna(pwm_metadata, split_cols)
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

    # set up results 
    results = {
        "nodes": [],
        "nodes_rna": [],
        "ATAC_signal": [],
        "delta_logit": [],
        "filename": [],
        "region_num": [],
        "num_target_genes": [],
        "max_rna_vals": [],
        "region_to_rna": [],
        "MSE": [],
        "downstream_interesting": [],
        "edge_type": [],
        "GO_terms": [],
        "GO_descriptions": []}

    if True:
        # for each grammar, run analyses:
        for grammar_idx in range(len(grammars)):

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

            # annotate the grammar
            new_grammar_file = "{}/{}.annot-{}.gml".format(
                args.out_dir,
                os.path.basename(grammar_file).split(".gml")[0],
                grammar_idx)
            grammar_results = annotate_one_grammar(
                grammar_file,
                grammar,
                new_grammar_file,
                pwm_to_rna_dict,
                args.tss,
                dynamic_genes,
                interesting_genes,
                args.background_rna,
                args.out_dir)

            # do not save out if no results
            if grammar_results["filename"] == "None":
                logging.info("{} does not have proximal genes, not saving".format(grammar_file))
                continue
            
            # attach the results
            for key in grammar_results.keys():
                results[key].append(grammar_results[key])

        # save out summary
        summary_file = "{}/grammar_summary.txt".format(args.out_dir)
        summary_df = pd.DataFrame(results)
        summary_df = summary_df.sort_values("downstream_interesting")
        summary_df.insert(0, "manual_filt", np.ones(summary_df.shape[0]))
        
        # rearrange
        cols = list(summary_df.columns.values)
        cols.pop(cols.index("GO_descriptions"))
        summary_df = summary_df[cols+["GO_descriptions"]]

        # save out
        summary_df.to_csv(summary_file, sep="\t")
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print summary_df

        logging.info("Filtered grammar total: {}".format(summary_df.shape[0]))

        # save out a filtered summary (functional GO terms filtering)
        filt_summary_file = "{}/grammar_summary.filt.txt".format(args.out_dir)
        summary_df.loc[summary_df["GO_terms"] == 1].to_csv(filt_summary_file, sep="\t")

    # TODO merge needs to happen here
    if merge_grammars:
        filt_summary_file = "{}/grammar_summary.filt.txt".format(args.out_dir)
        filt_summary_file = merge_duplicates(
            filt_summary_file,
            pwm_to_rna_dict,
            args.tss,
            dynamic_genes,
            interesting_genes,
            args.background_rna,
            args.out_dir)
        
    # TODO make a function to adjust grammar colors?
    # color by function (not by ordering across time)?

    return filt_summary_file




def plot_results(filt_summary_file, out_dir):
    """pull out matrices for plotting. need:
    1) motif presence in each grammar
    2) ATAC pattern for each grammar
    3) RNA pattern for each grammar
    """
    grammars_df = pd.read_table(filt_summary_file)
    print grammars_df.shape

    # get number of grammars
    num_grammars = grammars_df.shape[0]

    # adjust ordering
    num_to_order_val = {
        8:1,
        10:3,
        1:2
        #3:12,
        #4:13,
        #5:14,
        #7:2,
        #8:3,
        #9:4,
        #10:5,
        #11:6,
        #12:7,
        #13:8,
        #14:9
    }
    
    for line_idx in range(num_grammars):
        grammar = nx.read_gml(grammars_df["filename"].iloc[line_idx])
        grammar_traj = int(
            os.path.basename(grammars_df["filename"].iloc[line_idx]).split(
                ".")[1].replace("DIFF_TOPK_LABELS-", "").split("_")[0].split("-")[0])
        
        # get motifs and merge into motif df
        motifs = grammars_df["nodes"].iloc[line_idx].split(",")
        motif_presence = pd.DataFrame(
            num_to_order_val[grammar_traj]*np.ones(len(motifs)),
            index=motifs,
            columns=[grammars_df["nodes_rna"].iloc[line_idx]])
        if line_idx == 0:
            motifs_all = motif_presence
        else:
            motifs_all = motifs_all.merge(
                motif_presence,
                left_index=True,
                right_index=True,
                how="outer")
            motifs_all = motifs_all.fillna(0)

        # also keep track of num regions per grammar (for adjacency matrix)
        num_examples = len(grammar.graph["examples"].split(","))
        motif_counts = pd.DataFrame(
            num_examples*np.ones(len(motifs)),
            index=motifs,
            columns=[grammars_df["nodes_rna"].iloc[line_idx]])
        if line_idx == 0:
            motifs_counts_all = motif_counts
        else:
            motifs_counts_all = motifs_counts_all.merge(
                motif_counts,
                left_index=True,
                right_index=True,
                how="outer")
            motifs_counts_all = motifs_counts_all.fillna(0)
            
        # extract RNA vector and append
        rna = [float(val) for val in grammar.graph["RNASIGNALS"].split(",")]
        if line_idx == 0:
            rna_all = np.zeros((num_grammars, len(rna)))
            rna_all[line_idx] = rna
        else:
            rna_all[line_idx] = rna

        # extract ATAC vector and append
        epigenome_signals = np.array([float(val) for val in grammar.graph["ATACSIGNALSNORM"].split(",")])
        atac = epigenome_signals[[1,2,3,4,5,6,7,8,9,10,11,12,13]]
        if line_idx == 0:
            atac_all = np.zeros((num_grammars, atac.shape[0]))
            atac_all[line_idx] = atac
        else:
            atac_all[line_idx] = atac

        # TODO extract GO terms and plot
        gprofiler_results_file = "{}.rna.foreground.go_gprofiler.txt".format(
            grammars_df["filename"].iloc[line_idx].split(".annot-")[0].split(".gml")[0])
        gprofiler_results = pd.read_table(gprofiler_results_file, sep="\t")
        gprofiler_results = gprofiler_results[gprofiler_results["domain"] == "BP"]
        gprofiler_results = gprofiler_results[["p.value", "term.id", "term.name"]]
        keep_indices = []
        for i in range(gprofiler_results.shape[0]):
            for go_term_str in GOOD_GO_TERMS:
                if go_term_str in gprofiler_results["term.name"].iloc[i]:
                    keep_indices.append(i)
        gprofiler_results = gprofiler_results.iloc[keep_indices]
        gprofiler_results = gprofiler_results.set_index("term.name")
        gprofiler_results[line_idx] = gprofiler_results["p.value"]
        gprofiler_results = gprofiler_results[[line_idx]]
        
        if line_idx == 0:
            go_all = gprofiler_results.copy()
        else:
            go_all = go_all.merge(
                gprofiler_results, how="outer", left_index=True, right_index=True)

    # clean up GO terms
    keep_indices = []
    for i in range(go_all.shape[0]):
        keep = True
        for bad_term_str in REMOVE_GO_TERMS:
            if bad_term_str in go_all.index[i]:
                print go_all.index[i]
                keep = False
        if keep:
            keep_indices.append(i)
    go_all = go_all.iloc[keep_indices]
    go_all = go_all.drop_duplicates()

    
    go_all = go_all[~go_all.index.isin(REMOVE_EXACT_TERMS)]
    
    # clean up and order
    go_all = -np.log10(go_all)
    go_all = go_all.fillna(0)
    go_clust = linkage(go_all.values, method="ward")
    reorder_indices = leaves_list(go_clust)
    go_all = go_all.iloc[reorder_indices,:]

    # transpose
    go_all = go_all.transpose()

    # tODO set up adjacency matrix
    motifs_counts_all = motifs_counts_all.transpose()
    num_motifs = motifs_counts_all.shape[1]
    motif_names = motifs_counts_all.columns
    adjacency_mat = np.zeros((num_motifs, num_motifs))
    for grammar_idx in range(motifs_counts_all.shape[0]):
        #print motif_counts_all.index[grammar_idx]
        indices = np.where(motifs_counts_all.iloc[grammar_idx,:].values != 0)[0]
        grammar_count = motifs_counts_all.iloc[grammar_idx,indices[0]]
        #print indices
        for idx in range(len(indices)-1):
            #print indices[idx], indices[idx+1]
            adjacency_mat[indices[idx], indices[idx+1]] += grammar_count
            adjacency_mat[indices[idx+1], indices[idx]] += grammar_count
    adj_df = pd.DataFrame(data=adjacency_mat, columns=motif_names, index=motif_names)

    # make a gml file
    graph = nx.from_pandas_adjacency(adj_df)
    nx.write_gml(graph, "test.gml")

    
    
    # transpose motifs matrix
    motifs_all["rank"] = np.sum(motifs_all.values, axis=1) / np.sum(motifs_all.values != 0, axis=1)
    motifs_all = motifs_all.sort_values("rank")
    del motifs_all["rank"]
    motifs_all = (motifs_all > 0).astype(int)
    motifs_all = motifs_all.transpose()

    # TODO clean up rownames for motif matrix
    tfs = motifs_all.index.values
    tfs_clean = []
    for tf_set in tfs:
        tf_set_clean = tf_set.replace("POU2F1;", "")
        tf_set_clean = tf_set_clean.replace("FOXA2;FOXO3;FOXP2;FOXC1;FOXM1;FOXK1;FOXO1;FOXP1", "FOXO1")
        tf_set_clean = tf_set_clean.replace(";SALL4", "")
        tf_set_clean = tf_set_clean.replace("ETV4;ETV5;ETS1;ETS2", "ETV5")
        tf_set_clean = tf_set_clean.replace("KLF12;SP1;SP3;SP2;SP4;KLF4;KLF5;KLF3;KLF6;KLF9", "KLF4")
        tf_set_clean = tf_set_clean.replace(";MAF", "")
        tf_set_clean = tf_set_clean.replace(";CEBPB;DBP;HLF", "")
        tf_set_clean = tf_set_clean.replace(";TFAP2C;ZNF750", "")
        tf_set_clean = tf_set_clean.replace("TAF1;", "")
        tf_set_clean = tf_set_clean.replace(";CEBPG", "")
        tf_set_clean = tf_set_clean.replace("TCF12;TCF3;TFAP4;TCF4", "TCF3")
        tf_set_clean = tf_set_clean.replace("ATF1;CREB1;ATF2;CREM", "CREB1")
        tf_set_clean = tf_set_clean.replace("CBFB;RUNX1;RUNX2", "RUNX1")
        tf_set_clean = tf_set_clean.replace("RELA;", "")
        tf_set_clean = tf_set_clean.replace(";NFKB2;REL", "")
        tf_set_clean = tf_set_clean.replace("TP53;TP63;TP73", "TP63")
        tf_set_clean = tf_set_clean.replace("NR2C1;RXRB;THRB;THRA", "NR2C1")
        tf_set_clean = tf_set_clean.replace("RORA;NR2F2;RARG;RXRA;RARA", "RORA")
        tf_set_clean = tf_set_clean.replace(";HSF2", "")
        tf_set_clean = tf_set_clean.replace(";SOX4", "")
        tf_set_clean = tf_set_clean.replace(";ISL1", "")
        tf_set_clean = tf_set_clean.replace(";TCF7L1;TCF7L2", "")
        tf_set_clean = tf_set_clean.replace(";NFATC2;NFATC3;NFATC4;ZNF384", "")
        tf_set_clean = tf_set_clean.replace("BACH1;BACH2;", "")
        tf_set_clean = tf_set_clean.replace(";ZNF554FGK", "")
        tf_set_clean = tf_set_clean.replace(";PKNOX1", "")
        tfs_clean.append(tf_set_clean)
    motifs_all.index = tfs_clean
        
    # zscore the ATAC data
    atac_all = zscore(atac_all, axis=1)

    # convert others to df
    atac_df = pd.DataFrame(atac_all)
    rna_df = pd.DataFrame(rna_all)
    
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

    adj_file = "{}/grammars.filt.motif_adjacency.mat.txt".format(out_dir)
    adj_df.to_csv(adj_file, sep="\t")
    
    motifs_file = "{}/grammars.filt.motif_presence.mat.txt".format(out_dir)
    motifs_all.to_csv(motifs_file, sep="\t")

    atac_file = "{}/grammars.filt.atac.mat.txt".format(out_dir)
    atac_df.to_csv(atac_file, sep="\t")

    rna_file = "{}/grammars.filt.rna.mat.txt".format(out_dir)
    rna_df.to_csv(rna_file, sep="\t")

    go_file = "{}/grammars.filt.go_terms.mat.txt".format(out_dir)
    go_all.to_csv(go_file, sep="\t")
    
    # run R script
    plot_file = "{}/grammars.filt.summary.pdf".format(out_dir)
    plot_summary = "ggr_plot_grammar_summary.R {} {} {} {} {}".format(
        motifs_file, atac_file, rna_file, go_file, plot_file)
    print plot_summary
    os.system(plot_summary)
    
    return



def compile_grammars(args):
    """compile grammars
    """

    for grammar_summary_idx in range(len(args.grammar_summaries)):
        grammar_summary = args.grammar_summaries[grammar_summary_idx]
        grammar_summary_path = os.path.dirname(grammar_summary)
        print grammar_summary

        # read in
        grammar_summary = pd.read_table(grammar_summary, index_col=0)

        # filter
        grammar_summary = grammar_summary[grammar_summary[args.filter] == 1]

        # copy relevant files to new folder
        for grammar_idx in range(grammar_summary.shape[0]):
            # need gprofiler and gml
            grammar_file = "{}/{}".format(
                grammar_summary_path,
                os.path.basename(grammar_summary.iloc[grammar_idx]["filename"]))
            copy_file = "cp {} {}/".format(grammar_file, args.out_dir)
            print copy_file
            os.system(copy_file)
            functional_file = re.sub(".annot-\d+.gml", "*gprofiler.txt", grammar_file)
            copy_file = "cp {} {}/".format(functional_file, args.out_dir)
            print copy_file
            os.system(copy_file)

            # and adjust file location
            grammar_index = grammar_summary.index[grammar_idx]
            grammar_summary.at[grammar_index, "filename"] = "{}/{}".format(
                args.out_dir,
                os.path.basename(grammar_file))
            
        # concat
        if grammar_summary_idx == 0:
            all_grammars = grammar_summary
        else:
            all_grammars = pd.concat([all_grammars,grammar_summary], axis=0)

    # save out to new dir
    all_grammars = all_grammars.sort_values("filename")
    if "manual_filt" not in all_grammars.columns:
        all_grammars.insert(0, "manual_filt", np.ones(all_grammars.shape[0]))
    new_grammar_summary_file = "{}/grammars_summary.txt".format(args.out_dir)
    all_grammars.to_csv(new_grammar_summary_file, sep="\t")
    
    return new_grammar_summary_file
