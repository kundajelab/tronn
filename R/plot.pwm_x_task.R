#!/usr/bin/env Rscript

# description: plot pwm x task

#library(fastcluster)
library(gplots)
library(RColorBrewer)

args <- commandArgs(trailingOnly=TRUE)
filename <- args[1]
plot_file <- args[2]

# read data
data <- read.table(filename, header=TRUE, row.names=NULL)
data$community <- NULL

# row normalize
rowmax <- apply(data, 1, max)
data_norm <- data / rowmax
data_norm[is.na(data_norm)] <- 0

# transpose to make prettier
data_norm <- t(data_norm)

# set up colors
my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
breaks <- seq(0, 1, length.out=50)

# grid
mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(2,6,4)
mylhei = c(0.5,12,1.5)

# heatmap
pdf(plot_file, height=18, width=5)
heatmap.2(
    as.matrix(data_norm),
    Rowv=TRUE,
    Colv=FALSE,
    dendrogram="none",
    trace="none",
    density.info="none",
    cexRow=1.5,
    cexCol=3,
    keysize=0.1,
    key.title=NA,
    key.xlab=NA,
    key.par=list(pin=c(4,0.1),
        mar=c(6.1,0,5.1,0),
        mgp=c(3,2,0),
        cex.axis=2.0,
        font.axis=2),
    margins=c(3,0),
    lmat=mylmat,
    lwid=mylwid,
    lhei=mylhei,
    col=my_palette,
    breaks=breaks)
dev.off()
