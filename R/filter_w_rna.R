#!/usr/bin/env Rscript

# filter out non expressed genes

library(stringr)

args <- commandArgs(trailingOnly=TRUE)
gene_list_file <- args[1]
conversion_file <- args[2]
rna_file <- args[3]
out_file <- args[4]

# params
rna_mean_cutoff <- 1

# =====================
# RNA data
# =====================

# bring in the RNA file, then get HGNC ids for these
rna_data <- read.table(gzfile(rna_file), header=TRUE, row.names=1, sep='\t')

# Average pairs of columns and convert to dataframe
rna_avg <- sapply(seq(1,ncol(rna_data),2), function(i) {
                      rowMeans(rna_data[,c(i, i+1)], na.rm=T)
                  })

# also remove zeros, normalize, and remove lowly expressed TFs
rna_nonzero <- rna_avg[rowSums(rna_avg[, -1])>0, ] # remove zeros
rna_asinh <- asinh(rna_nonzero) # asinh (to put on linear and more comparable scale)

# mean cutoff, remove lowly expressed (ie, off) TFs
rna_row_max <- apply(rna_asinh, 1, max)
rna_keep_indices <- which(rna_row_max > rna_mean_cutoff)
rna_meancutoff <- rna_asinh[rna_keep_indices,]

rna_final_df <- data.frame(rna_meancutoff)
rna_final_df$ensembl_gene_id <- rownames(rna_final_df)

# read in conversion table and add hgnc data
conversion_table <- read.table(gzfile(conversion_file), sep='\t', header=TRUE)
rna_data_w_hgnc <- merge(rna_final_df, conversion_table, by='ensembl_gene_id', all.x=TRUE, all.y=FALSE, sort=FALSE)

#print(head(rna_data_w_hgnc))

# =====================
# Read in filtered motifs
# =====================

motif_data <- read.table(gene_list_file, header=TRUE, row.names=1, sep='\t')
motif_data$ensembl_gene_id <- str_split_fixed(rownames(motif_data), "_", 3)[,2]
motif_data$motif_id <- rownames(motif_data)

#print(str_split_fixed(rownames(motif_data), "_", 3))
#print(motif_data)

# =====================
# Merge with rna
# =====================

motif_w_rna <- merge(motif_data, rna_data_w_hgnc, by='ensembl_gene_id', all.x=FALSE, all.y=FALSE, sort=FALSE)
motif_w_rna_ordercolumns <- motif_w_rna[,c(ncol(motif_w_rna),1:(ncol(motif_w_rna)-1))]

#print(motif_w_rna_ordercolumns)

write.table(motif_w_rna_ordercolumns, file=out_file, quote=FALSE, row.names=FALSE, sep='\t')




