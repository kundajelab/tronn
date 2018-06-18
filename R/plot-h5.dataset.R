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
    mylhei = c(0.25,4,0.5)

    # plot
    pdf(out_pdf_file, height=18, width=10)
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
            mar=c(2.1,0,2.1,0),
            mgp=c(3,1,0),
            cex.axis=1.0,
            font.axis=2),
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

# read in data
data <- h5read(h5_file, dataset_key, read.attributes=TRUE)
rownames(data) <- attr(data, row_attr_key)
print(dim(data))

# make heatmap
heatmap_file <- paste(
    sub(".h5", "", h5_file),
    dataset_key,
    "pdf", sep=".")
print(heatmap_file)
make_heatmap(data, heatmap_file)
