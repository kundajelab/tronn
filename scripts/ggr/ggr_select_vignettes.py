#!/usr/bin/env python

import os
import sys
import h5py

import numpy as np
import pandas as pd

from tronn.plot.visualization import scale_scores
from tronn.plot.visualization import plot_weights_group
from tronn.util.formats import array_to_bed


idx_to_letter = {
    0: "A",
    1: "C",
    2: "G",
    3: "T"
}


def get_vignettes(
        h5_file,
        bed_file,
        tmp_dir=".",
        importance_filter=True,
        rsid_to_genes=None,
        ensembl_to_hgnc=None):
    """get the example indices at the locations of interest
    """
    # prefix (scanmotifs)
    dirname = h5_file.split("/")[-2]
    print dirname
    
    # get a bed file from the h5 file and overlap
    with h5py.File(h5_file, "r") as hf:
        metadata = hf["example_metadata"][:,0]
        metadata_bed_file = "{}/{}.metadata.tmp.bed.gz".format(
            tmp_dir, dirname)
        array_to_bed(metadata, metadata_bed_file, name_key="all", merge=False)

    # overlap and read back in
    overlap_file = "{}.overlap.bed.gz".format(metadata_bed_file.split(".bed")[0])
    overlap_cmd = (
        "zcat {} | "
        "awk -F '\t' 'BEGIN{{OFS=\"\t\"}}{{ $2=$2+20; $3=$3-20; print }}' | " # offset
        "bedtools intersect -wo -a stdin -b {} | "
        "gzip -c > {}").format(metadata_bed_file, bed_file, overlap_file)
    print overlap_cmd
    os.system(overlap_cmd)
    overlap_data = pd.read_csv(overlap_file, sep="\t", header=None)
    print overlap_data.shape

    # for each, go back in and find the index, then check importance scores
    total = 0
    for overlap_i in range(overlap_data.shape[0]):
        metadata_i = overlap_data[3][overlap_i]
        h5_i = np.where(metadata == metadata_i)[0][0]
        rsid_i = overlap_data[9][overlap_i]
        variant_pos = overlap_data[7][overlap_i] - overlap_data[1][overlap_i]
        variant_pos -= 1 # offset by 1
        if variant_pos < 0:
            continue
        
        # filter here, if using rsid to genes
        if (rsid_to_genes is not None) and (len(rsid_to_genes.keys()) > 0):
            try:
                gene_id = rsid_to_genes[rsid_i]
            except:
                continue
        else:
            gene_id = "UNKNOWN"

        # and get hgnc
        hgnc_id = ensembl_to_hgnc.get(gene_id, "UNKNOWN")
            
        
        with h5py.File(h5_file, "r") as hf:
            #for key in sorted(hf.keys()): print key, hf[key].shape
            if False:
                variant_impt = hf["sequence-weighted.active"][h5_i,:,variant_pos,:]
                #variant_impt = hf["sequence-weighted"][h5_i,:,420:580][:,variant_pos]
            else:
                start_pos = max(variant_pos - 1, 0)
                stop_pos = min(variant_pos + 1, hf["sequence-weighted.active"].shape[2])
                variant_impt = hf["sequence-weighted.active"][h5_i,:,start_pos:stop_pos,:]
            variant_val = hf["sequence.active"][h5_i,:,variant_pos,:]
            variant_ref_bp = idx_to_letter[np.argmax(variant_val)]
            
        # variant scoring
        try:
            variant_impt_max = np.max(np.abs(variant_impt))
        except:
            import ipdb
            ipdb.set_trace()
        if variant_impt_max > 0:
            
            # plot out
            if True:
                with h5py.File(h5_file, "r") as hf:
                    # get the full sequences
                    orig_importances = hf["sequence-weighted"][h5_i][:,420:580]
                    match_importances = hf["sequence-weighted.active"][h5_i]

                
                importances = scale_scores(orig_importances, match_importances)
                    
            print gene_id
            metadata_string = metadata_i.split("features=")[-1].replace(":", "_")
            # TODO also add in chrom region
            plot_file = "{}/{}.{}-{}.{}.{}.{}.{}.{}.plot.pdf".format(
                tmp_dir, dirname, gene_id, hgnc_id, h5_i, rsid_i, variant_pos, variant_ref_bp, metadata_string)
            print plot_file
            plot_weights_group(importances, plot_file)
            total += 1
    print total
            
    return None



def main():
    """look for interesting vignettes with variants in them
    """
    # args
    gwas_file = sys.argv[1]
    out_file = sys.argv[2]
    #filter_genes_file = "/mnt/lab_data/kundaje/users/dskim89/ggr/integrative/v1.0.0a/data/ggr.rna.counts.pc.expressed.timeseries_adj.pooled.rlog.dynamic.traj.mat.txt.gz"
    filter_genes_file = "/mnt/lab_data/kundaje/users/dskim89/ggr/integrative/v1.0.0a/annotations/hg19.genodermatoses.ensembl_ids.txt.gz"
    h5_files = sys.argv[3:]

    # filter_genes
    genes_mat = pd.read_csv(filter_genes_file, sep="\t", index_col=0)
    filter_genes = genes_mat.index.values.tolist()

    # keep hgnc ids?
    if "genodermatoses" in filter_genes_file:
        ensembl_to_hgnc = dict(zip(genes_mat.index.values, genes_mat.iloc[:,0]))
    else:
        ensemble_to_hgnc = None
    
    # rsid to gene
    rsid_to_genes = {}
    if "gtex" in gwas_file:
        signif_variant_gene_pairs_files = [
            "/mnt/lab_data/kundaje/users/dskim89/ggr/variants/GTEx/GTEx_Analysis_v7_eQTL/Skin_Not_Sun_Exposed_Suprapubic.v7.signif_variant_gene_pairs.txt.gz",
            "/mnt/lab_data/kundaje/users/dskim89/ggr/variants/GTEx/GTEx_Analysis_v7_eQTL/Skin_Sun_Exposed_Lower_leg.v7.signif_variant_gene_pairs.txt.gz"]

        for filename in signif_variant_gene_pairs_files:
            print filename
            pairs = pd.read_csv(filename, sep="\t")
            pairs["gene_id"] = pairs["gene_id"].str.split(".").str[0]
            print pairs.shape
            
            if True:
                pairs = pairs[pairs["gene_id"].isin(filter_genes)]
            print pairs.shape
                
            tmp_rsid_to_genes = dict(zip(pairs["variant_id"], pairs["gene_id"]))
            rsid_to_genes.update(tmp_rsid_to_genes)
            
    tmp_dir = "."

    
    # go through each file
    for h5_file in h5_files:
        print h5_file
        get_vignettes(
            h5_file, gwas_file,
            importance_filter=True,
            rsid_to_genes=rsid_to_genes,
            ensembl_to_hgnc=ensembl_to_hgnc)
    
    return

main()
