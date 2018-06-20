#!/usr/bin/env Rscript

# description: given keys, aggregate info and plot
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
    cluster_columns) {
    
    # color palette
    my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)

    # grid
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(0.25,1.5,0.5)
    mylhei = c(0.25,4,0.75)

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
        
        labCol="",
        cexRow=1.0,
        
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
            breaks <- breaks[c(1,length(breaks))]
            list(at = parent.frame()$scale01(breaks),
                 labels = breaks)},
        
        margins=c(1,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        col=my_palette)
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
data_strings <- args[5:length(args)]

# read in clusters
clusters <- h5read(h5_file, cluster_key)
clusters <- t(clusters)
clusters <- clusters[,cluster_col]
cluster_ids <- unique(clusters)
cluster_ids <- cluster_ids[order(cluster_ids)]
if (remove_final_cluster == 1) {
    cluster_ids <- cluster_ids[1:(length(cluster_ids)-1)]
}

# for each cluster, extract keys and aggregate
for (cluster_idx in 1:length(cluster_ids)) {
    
    for (key_idx in 1:length(data_strings)) {

        # split out the data string
        key_and_indices <- unlist(
            strsplit(data_strings[key_idx], "=", fixed=TRUE))
        key <- key_and_indices[1]
        
        if (length(key_and_indices) > 1) {
            indices <- as.numeric(unlist(
                strsplit(key_and_indices[2], ",", fixed=TRUE))) + 1
        } else {
            indices <- NA
        }
       
        # extract from h5 file
        data <- h5read(h5_file, key, read.attributes=TRUE)
        data <- t(data)
        if (!is.na(indices)) {
            data <- data[,indices]
        }
        data <- data[clusters==cluster_ids[cluster_idx],]

        # TODO may need to normalize the scales
        rowmax <- apply(data, 1, function(x) max(x))
        data_norm <- data / rowmax
        data_norm[is.na(data_norm)] <- 0
        data <- data_norm
        
        # aggregate
        data <- colMeans(data)
        
        # append
        if (key_idx == 1) {
            all_data <- data
            key_string <- key
            data_names <- c(key)
        } else {
            all_data <- rbind(all_data, data)
            key_string <- paste(key_string, key, sep="-")
            data_names <- c(data_names, key)
        }

    }

    # add row names
    rownames(all_data) <- data_names
    
    # make heatmap
    heatmap_file <- paste(
        sub(".h5", "", h5_file),
        key_string,
        paste(cluster_key, cluster_col-1, sep="-"),
        paste("cluster", cluster_ids[cluster_idx], sep="-"),
        "pdf", sep=".")
    print(heatmap_file)
    make_heatmap(all_data, heatmap_file, FALSE)
    
}
