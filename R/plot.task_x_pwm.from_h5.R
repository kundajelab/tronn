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

# data coming from....?

# read in data
task_x_pwm <- h5read(h5_file, key, read.attributes=TRUE)

# get pwm names
pwm_names <- gsub(".UNK.*", "", attr(task_x_pwm, "pwm_names"))
pwm_names <- gsub("HCLUST-.*_", "", pwm_names)
rownames(task_x_pwm) <- pwm_names
colnames(task_x_pwm) <- c(
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

#my_palette <- rev(colorRampPalette(brewer.pal(9, "RdBu"))(49))
my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)


# adjust settings
mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(0.25,1.5,0.50)
mylhei = c(0.25,4,0.5)


plot_file <- paste(prefix, ".pdf", sep="")        
print(plot_file)
pdf(plot_file, height=7, width=3, family="ArialMT")
heatmap.2(
    as.matrix(task_x_pwm),
    #main=paste("task-", task_j, sep=""),
    Rowv=TRUE,
    Colv=FALSE,
    dendrogram="none",
    trace="none",
    density.info="none",
    colsep=1:ncol(task_x_pwm),
    rowsep=1:nrow(task_x_pwm),
    sepcolor="black",
    sepwidth=c(0.01,0.01),
    #labCol=c(
    #    rep("", floor(ncol(task_x_pwm)/2)),
    #    label[i],
    #    rep("", ceiling(ncol(task_x_pwm)/2)-1)),
    #labRow=rep("", nrow(task_x_pwm)),
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

