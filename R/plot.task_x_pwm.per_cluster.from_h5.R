#!/usr/bin/env Rscript

# description: plot summary of grammar and mutagenesis
# NOTE: rhdf5 transposes axes!

library(rhdf5)
library(gplots)
library(ggplot2)
library(reshape)
library(RColorBrewer)
#library(scales)

# helper functions
se <- function(x) sd(x) / sqrt(length(x))

#h5ls(h5_file)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
key <- args[2]
prefix <- args[3]

# read in data
task_x_pwm <- h5read(h5_file, key, read.attributes=TRUE)
print(dim(task_x_pwm))
num_clusters <- dim(task_x_pwm)[3]

logits <- h5read(h5_file, "dmim-scores.probs", read.attributes=TRUE)
labels <- h5read(h5_file, "dmim-scores.labels", read.attributes=TRUE)

# get pwm names
pwm_names <- gsub(".UNK.*", "", attr(task_x_pwm, "pwm_names"))
pwm_names <- gsub("HCLUST-.*_", "", pwm_names)
rownames(task_x_pwm) <- pwm_names
days <- c(
    "d0.0",
    "d0.5",
    "d1.0",
    "d1.5",
    "d2.0",
    "d2.5",
    "d3.0",
    "d4.5",
    "d5.0",
    "d6.0")
colnames(task_x_pwm) <- days

#my_palette <- rev(colorRampPalette(brewer.pal(9, "RdBu"))(49))
my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)


# adjust settings
mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(0.25,1.5,0.50)
mylhei = c(0.25,4,0.5)


# set up a motif ordering
flattened <- task_x_pwm
dim(flattened) <- c(dim(task_x_pwm)[1], dim(task_x_pwm)[2]*dim(task_x_pwm)[3])

hc <- hclust(dist(flattened), method="ward.D2")
ordering <- hc$order
print(ordering)

for (i in 1:num_clusters) {

    # extract the data
    data <- task_x_pwm[,,i]
    data <- data[ordering,]
    
    # plot file
    my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)

    plot_file <- paste(prefix, ".cluster-", i, ".pdf", sep="")        
    print(plot_file)
    pdf(plot_file, height=7, width=3, family="ArialMT")
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=FALSE,
        dendrogram="none",
        trace="none",
        density.info="none",
        colsep=1:ncol(task_x_pwm),
        rowsep=1:nrow(task_x_pwm),
        sepcolor="black",
        sepwidth=c(0.01,0.01),
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(2.1,0,2.1,0),
            mgp=c(3,1,0),
            cex.axis=1.0,
            font.axis=2),
        srtCol=45,
        cexCol=1.25,
        margins=c(1,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        col=my_palette,
        symkey=FALSE)
    dev.off()

    # also plot out logits and labels
    my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    data <- rbind(labels[,i], logits[,i])
    colnames(data) <- days
    rownames(data) <- c("actual", "predictions")
    plot_file <- paste(prefix, ".cluster-", i, ".activations.pdf", sep="")        
    print(plot_file)
    pdf(plot_file, height=7, width=3, family="ArialMT")
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=FALSE,
        dendrogram="none",
        trace="none",
        density.info="none",
        colsep=1:ncol(task_x_pwm),
        rowsep=1:nrow(task_x_pwm),
        sepcolor="black",
        sepwidth=c(0.01,0.01),
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(2.1,0,2.1,0),
            mgp=c(3,1,0),
            cex.axis=1.0,
            font.axis=2),
        srtCol=45,
        cexCol=1.25,
        margins=c(1,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        col=my_palette,
        symkey=FALSE)
    dev.off()

    

    
}
