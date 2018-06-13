#!/usr/bin/env Rscript

#library(fastcluster)
library(gplots)
library(RColorBrewer)

args <- commandArgs(trailingOnly=TRUE)
filename <- args[1]
plot_file <- args[2]

# read in data
data <- read.table(filename, header=TRUE)
data_cor <- data

# hclust out here to be able to save out ordering
if (TRUE) {
    hc <- hclust(as.dist(1-data_cor))
    order <- data.frame(order=hc$order, names=rownames(data_cor)[hc$order])
    dend <- as.dendrogram(hc)
    
    order_file <- paste("global", "ordering.tmp", sep=".")
    write.table(order, file=order_file, sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)
    #print(order)
} else {
    order <- read.table("global.ordering.tmp", header=FALSE, row.names=NULL)
    order$order <- order$V1
    #print(order)
}

data_cor <- data_cor[order$order, order$order]

# set up colors
my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
my_breaks <- c(seq(0,0.69,length=25),seq(0.7,1,length=25))

#heatmap_file <- paste(prefix, ".pwm_x_pwm.corr.pdf", sep="")
pdf(plot_file)
heatmap.2(
    as.matrix(data_cor),
    Rowv=FALSE,
    Colv=FALSE,
    dendrogram="none",
    trace="none",
    density.info="none",
    cexRow=0.2,
    cexCol=0.2,
    col=my_palette)
dev.off()
