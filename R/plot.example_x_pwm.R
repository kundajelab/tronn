#!/usr/bin/env Rscript

# description: plot region x pwm (pre-clustered matrix) plot

library(gplots)
library(RColorBrewer)

args <- commandArgs(trailingOnly=TRUE)
filename <- args[1]
plot_file <- args[2]

# read in data
data <- read.table(filename, header=TRUE, row.names=1)
data$community <- NULL

# normalize
rowmax <- apply(data, 1, function(x) max(x))
data_norm <- data / rowmax
data_norm[is.na(data_norm)] <- 0

# draw ordered sample
plottable_nrow <- 1500
if (nrow(data_norm) > plottable_nrow) {
    skip_int <- as.integer(nrow(data_norm) / plottable_nrow)
    ordered_sample_indices <- seq(1, nrow(data_norm), skip_int)
    data_ordered <- data_norm[ordered_sample_indices,]
} else {
    data_ordered <- data_norm
}

print(dim(data_ordered))

# TODO - adjust scale?
my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)

# grid
mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(2,6,2)
mylhei = c(0.5,12,1.5)

# heatmap
#heatmap_file <- paste(prefix, ".example_x_pwm.hclust.pdf", sep="")
heatmap_file <- sub("txt", "example_x_pwm.pdf")
pdf(heatmap_file, height=18, width=20)
heatmap.2(
    as.matrix(data_ordered),
    Rowv=FALSE,
    Colv=TRUE,
    dendrogram="column",
    trace="none",
    density.info="none",
    labRow="",
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
    col=my_palette)
dev.off()
