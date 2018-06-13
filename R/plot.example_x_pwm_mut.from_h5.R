#!/usr/bin/env Rscript

# description: plot region x pwm from hdf5 file
# NOTE: rhdf5 transposes axes!

library(rhdf5)
library(gplots)
library(RColorBrewer)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
key <- args[2]
prefix <- args[3]

#h5ls(h5_file)

# read in data
data <- h5read(h5_file, key, read.attributes=TRUE)

num_mutations <- dim(data)[2]

#pwm_names <- gsub(".UNK.*", "", attr(dmim_results, "pwm_names"))
#pwm_names <- gsub("HCLUST-", "", pwm_names)

# colors
my_palette <- rev(colorRampPalette(brewer.pal(9, "RdBu"))(49))

for (mut_i in 1:num_mutations) {

    mut_data <- t(data[,mut_i,]) # {example, motif}
    data_norm <- mut_data
    
    # draw ordered sample
    plottable_nrow <- 1000
    if (nrow(data_norm) > plottable_nrow) {
        skip_int <- as.integer(nrow(data_norm) / plottable_nrow)
        ordered_sample_indices <- seq(1, nrow(data_norm), skip_int)
        data_ordered <- data_norm[ordered_sample_indices,]
    } else {
        data_ordered <- data_norm
    }

    print(dim(data_ordered))
    
    plot_file <- paste(prefix, ".mut-", mut_i, ".taskidx-0", ".pdf", sep="")
    print(plot_file)
    pdf(plot_file)
    heatmap.2(
        as.matrix(data_ordered),
        Rowv=TRUE,
        Colv=TRUE,
        dendrogram="none",
        trace="none",
        density.info="none",
        col=my_palette,
        symkey=TRUE)
    dev.off()
    
}










quit()



pwm_data <- h5read(h5_file, "pwm-scores.taskidx-0", read.attributes=TRUE)
rownames(data) <- attr(data, "pwm_names") # TODO figure out how to make this a default setting

# transpose (fastest changing axis is opposite order in R vs python)
data <- t(data)
pwm_data <- t(pwm_data)
print(dim(data))
print(dim(pwm_data))

# normalize
rowmax <- apply(pwm_data, 1, function(x) max(x))
pwm_norm <- pwm_data / rowmax
pwm_norm[is.na(pwm_norm)] <- 0
pwm_data <- pwm_norm

data_norm <- data

# draw ordered sample
plottable_nrow <- 1500
if (nrow(data_norm) > plottable_nrow) {
    skip_int <- as.integer(nrow(data_norm) / plottable_nrow)
    ordered_sample_indices <- seq(1, nrow(data_norm), skip_int)
    data_ordered <- data_norm[ordered_sample_indices,]
    pwm_data <- pwm_data[ordered_sample_indices,]
} else {
    data_ordered <- data_norm
}

# order out here
hc <- hclust(dist(data_ordered))
data_ordered <- data_ordered[hc$order,]
pwm_data <- pwm_data[hc$order,]

# color palette
my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
my_palette <- rev(colorRampPalette(brewer.pal(9, "RdBu"))(49))

# grid
mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(2,6,2)
mylhei = c(0.5,12,1.5)

# heatmap
heatmap_file <- paste(
    paste(
        sub(".h5", "", h5_file),
        dataset_key,
        "1", sep="-"), # to match 0-START from python
    "pdf", sep=".")
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


# heatmap
my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
heatmap_file <- paste(
    paste(
        sub(".h5", "", h5_file),
        dataset_key,
        "0", sep="-"), # to match 0-START from python
    "example_x_pwm",
    "pdf", sep=".")
pdf(heatmap_file, height=18, width=20)
heatmap.2(
    as.matrix(pwm_data),
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
