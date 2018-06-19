#!/usr/bin/env Rscript

# description: given a key, aggregate per cluster per task
# NOTE: rhdf5 transposes axes!

library(rhdf5)
library(gplots)
library(reshape2)
library(RColorBrewer)

# helper functions
make_heatmap <- function(
    data,
    out_pdf_file,
    cluster_columns) {
    
    # color palette
    my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)

    # grid
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(2,6,2)
    mylhei = c(0.5,12,1.5)

    # plot
    pdf(out_pdf_file, height=18, width=10)
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=cluster_columns,
        dendrogram="none",
        trace="none",
        density.info="none",
        cexCol=1.25,
        cexRow=1.25,
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(9.1,0,2.1,0),
            mgp=c(3,2,0),
            cex.axis=2.0,
            font.axis=2),
        margins=c(3,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        col=my_palette,
        useRaster=FALSE)
    dev.off()

}

# =========================================
# START CODE
# =========================================

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
cluster_key <- args[2]
cluster_col <- as.numeric(args[3]) + 1
remove_final_cluster <- as.numeric(args[4])
normalize <- as.numeric(args[5])
cluster_rows <- as.numeric(args[6])
key <- args[7]

# read in clusters
clusters <- h5read(h5_file, cluster_key)
clusters <- t(clusters)
clusters <- clusters[,cluster_col]
cluster_ids <- unique(clusters)
cluster_ids <- cluster_ids[order(cluster_ids)]
if (remove_final_cluster == 1) {
    cluster_ids <- cluster_ids[1:(length(cluster_ids)-1)]
}

if (length(args) > 7) {
    indices <- as.numeric(args[8:length(args)]) + 1
} else {
    indices <- NA
}


data <- h5read(h5_file, key, read.attributes=TRUE)
data <- t(data)
if (!is.na(indices)) {
    data <- data[,indices]
}
print(dim(data))

# for each cluster, extract keys and aggregate
for (cluster_idx in 1:length(cluster_ids)) {

    # extract information from dataset
    cluster_data <- data[clusters==cluster_ids[cluster_idx],]
    
    # aggregate
    cluster_data <- colMeans(cluster_data) # consider medians
    
    # and append
    if (cluster_idx == 1) {
        all_data <- cluster_data
    } else {
        all_data <- rbind(all_data, cluster_data)
    }
    
}

# add row and colnames
print(head(all_data))
rownames(all_data) <- cluster_ids
colnames(all_data) <- paste("idx", indices, sep="-")

print(head(all_data))

# make heatmap
heatmap_file <- paste(
    sub(".h5", "", h5_file),
    key,
    "by_cluster",
    "pdf", sep=".")
print(heatmap_file)
make_heatmap(all_data, heatmap_file, FALSE)
