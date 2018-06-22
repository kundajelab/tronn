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
    out_pdf_file) {
    
    # color palette
    my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)

    # grid
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(0.25,1.5,0.50)
    mylhei = c(0.25,4,0.75)

    # plot
    pdf(out_pdf_file, height=7, width=3, family="ArialMT")
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=FALSE,
        dendrogram="none",
        trace="none",
        density.info="none",
        colsep=1:ncol(data),
        rowsep=1:nrow(data),
        sepcolor="black",
        sepwidth=c(0.01,0.01),
        cexCol=1.25,
        srtCol=45,
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(3.1,0,3.1,0),
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
        col=my_palette,
        symkey=FALSE)
    dev.off()

}

# =========================================
# START CODE
# =========================================

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
dataset_key <- args[2]
row_attr_key <- args[3]
filter_vector_key <- args[4]

# read in data
data <- h5read(h5_file, dataset_key, read.attributes=TRUE)
filter_vector <- h5read(h5_file, filter_vector_key)
print(dim(data))

# make plot for each cluster
num_clusters <- dim(data)[3]
for (cluster_idx in 1:num_clusters) {
    cluster_data <- data[,,cluster_idx]
    rownames(cluster_data) <- attr(data, row_attr_key)
    cluster_data <- cluster_data[filter_vector > 0,]

    heatmap_file <- paste(
        sub("h5", "", h5_file),
        dataset_key,
        ".cluster-",
        cluster_idx-1,
        ".pdf", sep="")
    print(heatmap_file)
    make_heatmap(cluster_data, heatmap_file)
    
    
}
