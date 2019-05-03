#!/usr/bin/env python

import os
import re
import sys
import h5py
import glob

import numpy as np
import pandas as pd

def aggregate_results(results_files, key):
    """aggregate and normalize results
    """
    num_files = 0
    results_df = None
    current_pwms = ""
    best_avg = 0
    best_results = None
    for filename_idx in range(len(results_files)):
        if filename_idx % 50 == 0:
            print filename_idx
        
        filename = results_files[filename_idx]
        
        # get comment line and make a metadata dict
        with open(filename, "r") as fp:
            comment_line = fp.readline().strip().strip("#").split(";")
            comment_line = comment_line[:-1] + comment_line[-1].split(",null_")
            metadata = dict([vals.split("=") for vals in comment_line])

        # adjust pwms names
        pwms = re.sub("HCLUST-\d+_", "", metadata["pwms"])
        pwms = re.sub(".UNK.0.A", "", pwms)
        pwms = pwms.replace("u'", "'")

        # check if pwms changes
        if (pwms != current_pwms) and (current_pwms != ""):
            # save out
            if results_df is None:
                results_df = best_results.copy()
            else:
                results_df = pd.concat([results_df, best_results], axis=0)
                
            # and update
            current_pwms = pwms
            best_avg = 0
            best_results = None
            num_files += 1

        # adjust
        if current_pwms == "":
            current_pwms = pwms
            
        # get null avg (to normalize)
        null_avgs = metadata["null_avgs"].strip("[").strip("]").split(", ")
        null_avgs = [float(val) for val in null_avgs]
        null_avg = np.max(null_avgs)

        # get grammar avg
        grammar_avg = float(metadata["grammar_avg"])
        grammar_avg_norm = grammar_avg - null_avg

        # build a df of all information, use pwms
        results_data = pd.read_csv(filename, sep="\t", comment="#")
        results_data = results_data[results_data["variable"] == "target"]
        results_data["signal"] = results_data["signal"] - null_avg
        results_data["variable"] = pwms
        
        # track best result
        if key == "variant":
            if abs(grammar_avg_norm) > best_avg:
                best_avg = abs(grammar_avg_norm)
                best_results = results_data.copy()
        else:
            if grammar_avg_norm > best_avg:
                best_avg = grammar_avg_norm
                best_results = results_data.copy()
    
    return results_df


def main():
    """run synergy calculations <- separate bc need R and such
    """
    # inputs
    out_dir = sys.argv[1]
    grammar_summary_file = sys.argv[2]
    os.system("mkdir -p {}".format(out_dir))
        
    # variant files
    variant_dir = "/datasets/ggr/1.0.0d/annotations/variants.validation"
    eqtl_bed_file = "{}/gtex.skin.atac_filt.bed.gz".format(variant_dir)
    eqtl_effects_file = "{}/Skin_Not_Sun_Exposed_Suprapubic.v7.signif_variant_gene_pairs.txt.gz".format(
        variant_dir)
    
    # read in grammar summary
    grammars_df = pd.read_table(grammar_summary_file)

    # for each grammar, analyze
    for grammar_idx in range(grammars_df.shape[0]):

        # get grammar file
        grammar_file = grammars_df.iloc[grammar_idx]["filename"]
        grammar_prefix = os.path.basename(grammar_file).split(".gml")[0]
        
        # get score files
        trajs = grammar_prefix.split(".")[1].split("_x_")
        score_files = ["{}/ggr.dmim.h5".format(traj) for traj in trajs]

        # get synergy file
        synergy_file = "synergy/{}/ggr.synergy.h5".format(grammar_prefix)

        # commands
        grammar_out_dir = "{}/{}".format(out_dir, grammar_prefix)
        if not os.path.isdir(grammar_out_dir):
            command = (
                "compare_grammar_to_null.py "
                "--grammar {} "
                "--score_files {} "
                "--synergy_file {} "
                "--eqtl_bed_file {} "
                "--eqtl_effects_file {} "
                "-o {} "
                "--prefix {}").format(
                    grammar_file,
                    " ".join(score_files),
                    synergy_file,
                    eqtl_bed_file,
                    eqtl_effects_file,
                    grammar_out_dir,
                    grammar_prefix)
            print command
            os.system(command)
        
        # also run synergy only
        grammar_out_dir_synergy = "{}/{}.synergy_only".format(out_dir, grammar_prefix)
        if not os.path.isdir(grammar_out_dir_synergy):
            command = (
                "compare_grammar_to_null.py "
                "--grammar {} "
                "--score_files {} "
                "--synergy_file {} "
                "--eqtl_bed_file {} "
                "--eqtl_effects_file {} "
                "-o {} "
                "--prefix {} "
                "--synergy_only").format(
                    grammar_file,
                    " ".join(score_files),
                    synergy_file,
                    eqtl_bed_file,
                    eqtl_effects_file,
                    grammar_out_dir_synergy,
                    grammar_prefix)
            print command
            os.system(command)
        
    # finally, aggregate
    # for ATAC/H3K27ac/variants, gather all files, sort, grab the commented lines
    # then go through and collect points, with null removed
    agg_dir = "{}/summary".format(out_dir)
    os.system("mkdir -p {}".format(agg_dir))
    
    keys = [
        "ATAC",
        "H3K27ac",
        "variant"]

    for key in keys:
        results_files = sorted(glob.glob("{}/*/results*{}*.txt".format(out_dir, key)))
        print len(results_files)
        
        # get synergy results
        out_file = "{}/agg.{}.synergy_only.txt".format(agg_dir, key)
        results_synergy_files = [filename for filename in results_files if "synergy" in filename]
        results_df = aggregate_results(results_synergy_files, key)
        results_df.to_csv(out_file, sep="\t")
        plot_cmd = "Rscript /datasets/inference.2019-03-12/dmim.shuffle/quick_summary_plot.R {} {}".format(
            out_file, "{}.pdf".format(out_file.split(".txt")[0]))
        os.system(plot_cmd)
        
        # get nonsynergy_results
        out_file = "{}/agg.{}.txt".format(agg_dir, key)
        results_nonsynergy_files = [filename for filename in results_files if "synergy" not in filename]
        results_df = aggregate_results(results_nonsynergy_files, key)
        results_df.to_csv(out_file, sep="\t")
        plot_cmd = "Rscript /datasets/inference.2019-03-12/dmim.shuffle/quick_summary_plot.R {} {}".format(
            out_file, "{}.pdf".format(out_file.split(".txt")[0]))
        os.system(plot_cmd)
    
    return

main()
