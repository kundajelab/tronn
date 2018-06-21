#!/usr/bin/env Rscript

# description: plot example x pwm from hdf5 file
# NOTE: rhdf5 transposes axes!
#h5ls(h5_file)

library(rhdf5)
library(gplots)
library(reshape2)
library(RColorBrewer)

# helper functions
make_heatmap <- function(
    data,
    out_pdf_file,
    cluster_columns,
    is_signal,
    use_raster) {
    
    # color palette
    my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)

    # if signal, adjust the color scale
    if (is_signal) {
        color_granularity <- 50
        data_melted <- melt(data)
        my_breaks <- seq(
            quantile(data_melted$value, 0.01),
            quantile(data_melted$value, 0.90),
            length.out=color_granularity)
    } else {
        my_breaks <- NULL
    }

    # adjust labels based on use_raster
    if (use_raster) {
        labCol <- ""
    } else {
        labCol <- colnames(data)
    }
    
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
        
        labRow="",
        labCol=labCol,
        cexCol=1.0,
        
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(9.1,1,2.1,1),
            mgp=c(3,2,0),
            cex.axis=2.0,
            font.axis=2),
        key.xtickfun=function() {
            breaks <- pretty(parent.frame()$breaks)
            #breaks <- breaks[c(1,length(breaks))]
            list(at = parent.frame()$scale01(breaks),
                 labels = breaks)},
        
        margins=c(3,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        
        col=my_palette,
        breaks=my_breaks,
        useRaster=use_raster)
    dev.off()

}

# =========================================
# START CODE
# =========================================

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
cluster_key <- args[2]
cluster_col <- as.numeric(args[3]) + 1 # MAKE SURE YOU USE 0-START (from python)
remove_final_cluster <- as.numeric(args[4]) # 1 for yes, 0 for no
row_normalize <- as.numeric(args[5])
signal_normalize <- as.numeric(args[6])
cluster_columns <- as.numeric(args[7])
use_raster <- as.numeric(args[8])
dataset_key <- args[9]

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
print(dim(data))
if (length(args) > 9) {
    indices <- as.numeric(unlist(strsplit(args[10], ",", fixed=TRUE))) + 1
    data <- data[,indices]
} else {
    indices <- c()
}


if (length(dim(clusters)) > 1) {
    clusters <- t(clusters)
    clusters <- clusters[,cluster_col]
}

# get correct column, and sort by column order
data <- data[order(clusters),]
clusters <- clusters[order(clusters)]
print(table(clusters))

# remove final cluster if desired
if (remove_final_cluster == 1) {
    final_cluster <- max(unique(clusters))
    data <- data[clusters != final_cluster,]
    print("Removed final cluster (ie the non-clustered)")
}

# row normalize
if (row_normalize == 1) {
    rowmax <- apply(data, 1, function(x) max(x))
    data_norm <- data / rowmax
    data_norm[is.na(data_norm)] <- 0
} else {
    data_norm <- data
}

# signal normalize
if (signal_normalize == 1) {
    is_signal <- TRUE
} else {
    is_signal <- FALSE
}

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

# add column names
if (length(indices) > 0) {
    colnames(data_ordered) <- paste("idx", indices-1, sep="-")
} else {
    colnames(data_ordered) <- paste("idx", 1:ncol(data_ordered)-1, sep="-")
}

# make heatmap
if (length(indices) > 0) {
    indices_string <- paste(
        "indices_",
        indices[1]-1,
        "-",
        indices[length(indices)]-1, sep="")
    key_string <- paste(
        dataset_key,
        indices_string,
        sep=".")
} else {
    key_string <- dataset_key
}

# raster
if (use_raster == 1) {
    use_raster <- TRUE
} else {
    use_raster <- FALSE
}

heatmap_file <- paste(
    sub(".h5", "", h5_file),
    key_string,
    paste(cluster_key, cluster_col-1, sep="-"), # to match 0-START from python
    "pdf", sep=".")
print(heatmap_file)
make_heatmap(data_ordered, heatmap_file, cluster_columns, is_signal, use_raster)
