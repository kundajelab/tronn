#!/usr/bin/env Rscript

# quick script to do sequence clustering by motifs

library(ggplot2)
library(preprocessCore)
library(reshape)
library(fastcluster)
library(ReorderCluster)
library(stringr)
library(RColorBrewer)

set.seed(1337)

my_hclust <- function(d) {
    return(hclust(d, method='ward.D2'))
}

my_palette <- rev(colorRampPalette(brewer.pal(11, 'RdBu'))(49))


args <- commandArgs(trailingOnly=TRUE)
motif_mat_file <- args[1]
k_val <- args[2]
dendro_cutoff <- args[3]
prefix <- args[4]

# load data and clean
data <- read.table(gzfile(motif_mat_file), header=TRUE, row.names=NULL)[,-1]

indices <- data$indices
data$indices <- NULL

data_nonzero <- data[rowSums(data[, -1])>0, ]
indices <- indices[rowSums(data[, -1])>0]
data_nonzero[is.na(data_nonzero)] <- 0

print(dim(data_nonzero))

# normalize data
data_z <- t(scale(t(data_nonzero)))


# try a tsne and plot
library(Rtsne)
tsne_quant <- Rtsne(data_z, pca=TRUE, verbose=TRUE, check_duplicates=FALSE)
tsne_plot <- paste(prefix, '.tsne.png', sep='')
png(tsne_plot)
smoothScatter(tsne_quant$Y, pch=20)
dev.off()

print(dim(tsne_quant$Y))

# first do a kmeans clustering with high k
data_kmeans <- kmeans(data_z, centers=k_val, iter.max=50)

# plot the centers (hclust) to see if coherent
centers_matrix <- as.matrix(data_kmeans$centers)
colnames(centers_matrix) <- sub("_.*", "", colnames(centers_matrix))
print(dim(centers_matrix))
centers_dist <- dist(centers_matrix)
centers_hclust <- my_hclust(centers_dist)
centers_dendr <- as.dendrogram(centers_hclust)
center_groups <- as.factor(cutree(centers_hclust, k=dendro_cutoff))

rearrange_joseph <- RearrangeJoseph(centers_hclust, as.matrix(centers_dist), center_groups, TRUE)
centers_reorder <- rearrange_joseph$hcl
centers_reorder_dendr <- as.dendrogram(centers_reorder)

centers_plot = paste(prefix, '.centers.png', sep='')
png(centers_plot, height=10, width=60, units='in', res=200)
heatmap.2(centers_matrix,
          Rowv=centers_reorder_dendr,
          Colv=TRUE,
          hclustfun=my_hclust,
          dendrogram='row',
          trace='none',
          density.info='none',
          cexCol=0.5,
          col=my_palette)
dev.off()

# now save out groups using center groups, and also visualize
center_groups <- as.factor(cutree(centers_reorder, k=dendro_cutoff))

data_z <- data.frame(data_z)

data_z$reduced_clusters <- center_groups[match(data_kmeans$cluster, 1:k_val)]
data_z$indices <- indices

for (group_idx in 1:dendro_cutoff) {
    print(group_idx)

    data_z_subset <- data_z[data_z$reduced_clusters == group_idx, ]

    indices_subset <- data_z_subset$indices
    data_z_subset$indices <- NULL
    data_z_subset$reduced_clusters <- NULL

    # visualize out
    data_z_subset_kmeans <- kmeans(data_z_subset, centers=4, iter.max=50)
    data_z_subset_reordered <- data_z_subset[order(data_z_subset_kmeans$cluster), ]

    colnames(data_z_subset_reordered) <- sub("_.*", "", colnames(data_z_subset_reordered))
    
    plot_file = paste(prefix, '.group_', group_idx, '.motif_mat.png', sep='')
    png(plot_file, height=10, width=60, units='in', res=200)
    heatmap.2(as.matrix(data_z_subset_reordered),
              Rowv=FALSE,
              Colv=TRUE,
              hclustfun=my_hclust,
              dendrogram='none',
              trace='none',
              density.info='none',
              cexCol=0.5,
              col=my_palette)
    dev.off()
    
    # TODO save out indices
    indices_file = paste(prefix, '.group_', group_idx, '.indices.txt.gz', sep='')
    write.table(indices_subset, gzfile(indices_file), quote=FALSE, sep='\t', row.names=FALSE, col.names=FALSE)

}



