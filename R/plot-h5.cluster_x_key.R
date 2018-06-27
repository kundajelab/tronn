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
    #my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)

    # grid
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(0.25,1.5,0.5)
    mylhei = c(0.25,4,0.75) # 0.5

    # plot
    pdf(out_pdf_file, height=7, width=3, family="ArialMT")
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=cluster_columns,
        dendrogram="none",
        trace="none",
        density.info="none",
        colsep=1:ncol(data),
        rowsep=1:nrow(data),
        sepcolor="black",
        sepwidth=c(0.01,0.01),
        cexCol=1.25,
        cexRow=1.25,
        srtCol=45,
        
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(3.1,1,3.1,1),
            mgp=c(3,1,0),
            cex.axis=1.0,
            font.axis=2),
        key.xtickfun=function() {
            breaks <- pretty(parent.frame()$breaks)
            #breaks <- breaks[c(1,length(breaks))]
            list(at = parent.frame()$scale01(breaks),
                 labels = breaks)},
        
        margins=c(1,0),
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
cluster_col <- as.numeric(args[3]) + 1 # put in -1 if using onehot clusters
remove_final_cluster <- as.numeric(args[4])
normalize <- as.numeric(args[5])
cluster_rows <- as.numeric(args[6])
key <- args[7]

# read in clusters
clusters <- h5read(h5_file, cluster_key)
if (length(dim(clusters)) > 1 & cluster_col > 0) {
    clusters <- t(clusters)
    clusters <- clusters[,cluster_col]
    cluster_ids <- unique(clusters)
} else if ( length(dim(clusters)) > 1 ) {
    clusters <- t(clusters)
    cluster_ids <- 1:ncol(clusters)
} else {
    cluster_ids <- unique(clusters)
}

# remove final cluster
if (remove_final_cluster == 1) {
    cluster_ids <- cluster_ids[1:(length(cluster_ids)-1)]
}

# get indices
if (length(args) > 7) {
    indices <- as.numeric(unlist(strsplit(args[8], ",", fixed=TRUE))) + 1
} else {
    indices <- c()
}

# read data
data <- h5read(h5_file, key, read.attributes=TRUE)
data <- t(data)
if (length(indices) > 0) {
    data <- data[,indices]
}
print(dim(data))

# for each cluster, extract keys and aggregate
for (cluster_idx in 1:length(cluster_ids)) {

    # extract information from dataset
    if (cluster_col > 0) {
        cluster_data <- data[clusters==cluster_ids[cluster_idx],]
    } else {
        cluster_data <- data[clusters[,cluster_idx] >= 1,]
    }
    
    # aggregate
    cluster_data <- colMeans(cluster_data) # consider medians
    cluster_data <- apply(x, 2, median)

    # and append
    if (cluster_idx == 1) {
        all_data <- cluster_data
    } else {
        all_data <- rbind(all_data, cluster_data)
    }
    
}

# add row and colnames
rownames(all_data) <- cluster_ids
if (length(indices) > 0) {
    colnames(all_data) <- paste("idx", indices-1, sep="-")
} else {
    colnames(all_data) <- paste("idx", 1:ncol(all_data)-1, sep="-")
}

# make heatmap
if (length(indices) > 0) {
    indices_string <- paste(
        "indices_",
        indices[1]-1,
        "-",
        indices[length(indices)]-1, sep="")
    key_string <- paste(
        key,
        indices_string,
        sep=".")
} else {
    key_string <- key
}

heatmap_file <- paste(
    sub(".h5", "", h5_file),
    key_string,
    "by_cluster",
    "pdf", sep=".")
print(heatmap_file)
make_heatmap(all_data, heatmap_file, FALSE)
