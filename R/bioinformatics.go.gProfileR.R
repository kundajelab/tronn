#!/usr/bin/env Rscript

# run DAVID GO term analysis
# requires gene set and background gene set

library(gProfileR)

# folders files etc
args <- commandArgs(trailingOnly=TRUE)
gene_list_file <- args[1]
background_list_file <- args[2]
out_dir <- args[3]

# read in gene list and background gene list
gene_list <- read.table(gzfile(gene_list_file), stringsAsFactors=FALSE)$V1
background_list <- read.table(gzfile(background_list_file), stringsAsFactors=FALSE)$V1

prefix <- sub('\\.txt.gz$', "", basename(gene_list_file))
padj_cutoff <- 0.1

# run gProfileR
results <- gprofiler(
    gene_list,
    ordered_query=TRUE,
    organism="hsapiens",
    #max_p_value=padj_cutoff,
    #correction_method="fdr",
    custom_bg=background_list)
out_file <- paste(out_dir, "/", prefix, ".go_gprofiler.txt", sep="")
write.table(results, out_file, quote=FALSE, sep="\t")
