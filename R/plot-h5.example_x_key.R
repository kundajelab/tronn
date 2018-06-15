#!/usr/bin/env Rscript

# description: plot example x pwm from hdf5 file
# NOTE: rhdf5 transposes axes!
#h5ls(h5_file)

library(rhdf5)
library(gplots)
library(RColorBrewer)
library(reshape2)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
cluster_key <- args[2]
cluster_col <- as.numeric(args[3]) + 1 # MAKE SURE YOU USE 0-START (from python)
remove_final_cluster <- as.numeric(args[4]) # 1 for yes, 0 for no
normalize <- args[5]
cluster_columns <- as.numeric(args[6])
dataset_key <- args[7]

# whether to cluster the columns
if (cluster_columns == 1) {
    print("clustering the columns")
    cluster_columns <- TRUE
} else {
    print("not clustering the columns")
    cluster_columns <- FALSE
}

# TODO add in color choice option

# read in data
data <- h5read(h5_file, dataset_key, read.attributes=TRUE)
#rownames(data) <- attr(data, "pwm_names") # TODO fix this
clusters <- h5read(h5_file, cluster_key)

# transpose (fastest changing axis is opposite order in R vs python)
data <- t(data)
if (length(args) > 7) {
    indices <- as.numeric(args[8:length(args)]) # which indices to grab, 0-START
    indices <- indices + 1
    print(indices)
    data <- data[,indices]
}
print(dim(data))
clusters <- t(clusters)

# get correct column, and sort by column order
clusters <- clusters[,cluster_col]
data <- data[order(clusters),]
clusters <- clusters[order(clusters)]

# remove final cluster if desired
if (remove_final_cluster == 1) {
    final_cluster <- max(unique(clusters))
    data <- data[clusters != final_cluster,]
    print("Removed final cluster (ie the non-clustered)")
}

# normalize
if (normalize == 1) {
    rowmax <- apply(data, 1, function(x) max(x))
    data_norm <- data / rowmax
    data_norm[is.na(data_norm)] <- 0
} else {
    data_norm <- data
}

# colors
color_granularity <- 50
data_melted <- melt(data_norm)
my_breaks <- seq(
    quantile(data_melted$value, 0.01),
    quantile(data_melted$value, 0.90),
    length.out=color_granularity)


# draw ordered sample
plottable_nrow <- 1500 #2000 # important - useRaster in heatmap!
if (nrow(data_norm) > plottable_nrow) {
    skip_int <- as.integer(nrow(data_norm) / plottable_nrow)
    ordered_sample_indices <- seq(1, nrow(data_norm), skip_int)
    data_ordered <- data_norm[ordered_sample_indices,]
} else {
    data_ordered <- data_norm
}

print(dim(data_ordered))

# color palette
my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)

# grid
mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(2,6,2)
mylhei = c(0.5,12,1.5)

# plot
heatmap_file <- paste(
    sub(".h5", "", h5_file),
    dataset_key,
    paste(cluster_key, cluster_col-1, sep="-"), # to match 0-START from python
    "pdf", sep=".")
print(heatmap_file)
pdf(heatmap_file, height=18, width=10)
heatmap.2(
    as.matrix(data_ordered),
    Rowv=FALSE,
    Colv=cluster_columns,
    dendrogram="none",
    trace="none",
    density.info="none",
    labRow="",
    labCol="",
    cexCol=0.5,
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
    #breaks=my_breaks,
    useRaster=TRUE)
dev.off()

