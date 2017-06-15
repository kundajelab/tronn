#!/usr/bin/env Rscript

library('visNetwork')
library(reshape2)

# script to make network form of grammar
# need to have: node IDs, links (to and from), labels, node size (motif z score), links strength (correlation strength)

args <- commandArgs(trailingOnly=TRUE)
zscore_file <- args[1]
sim_file <- args[2]
out_file <- args[3]

node_size_factor <- 15
edge_width_factor <- 1

links <- read.table(sim_file, header=TRUE, row.names=NULL, sep='\t')
links$X <- NULL
links_dedup <- links[!(colnames(links)=="HOXD3_ENSG00000128652_D.1"), !(colnames(links)=="HOXD3_ENSG00000128652_D.1")]
rownames(links_dedup) <- colnames(links_dedup)

# remove duplicates
links_dedup_mat <- as.matrix(links_dedup)
links_dedup_mat[lower.tri(links_dedup_mat, diag=TRUE)] <- 0
links_dedup <- data.frame(links_dedup_mat)

links_dedup$id <- rownames(links_dedup)
links_melted <- melt(links_dedup)
colnames(links_melted) <- c('to', 'from', 'correlation')
links_melted <- links_melted[links_melted$correlation > 0.5,]
links_melted$width <- edge_width_factor*links_melted$correlation

nodes <- read.table(zscore_file, header=TRUE, sep='\t', stringsAsFactors=FALSE)
nodes_reduced <- nodes[,c('hgnc_symbol', 'zscore', 'motif_id')]
colnames(nodes_reduced) <- c('label', 'size', 'id')

nodes_reduced$size <- node_size_factor*nodes_reduced$size

# build the net
net <- visNetwork(nodes_reduced, links_melted)

#print(dirname(out_file))
#out_dir <- paste(getwd(), dirname(out_file), sep='/')
setwd(dirname(out_file))

#visSave(net, file=basename(out_file), selfcontained=TRUE) # requires pandoc
visSave(net, file=basename(out_file), selfcontained=FALSE)










