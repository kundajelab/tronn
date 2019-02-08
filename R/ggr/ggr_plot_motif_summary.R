#!/usr/bin/env Rscript

# description - code to plot out summary after scanmotifs

library(gplots)
library(RColorBrewer)
library(reshape2)

set.seed(1337)

# hclust function with ward distance as default
my_hclust <- function(data) {
    return(hclust(data, method="ward.D2"))
}


# make heatmap fn
make_heatmap <- function(data, my_palette, my_breaks) {
    
    # grid
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(2,5,3)
    mylhei = c(0.5,12,1.5)

    # plot
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=FALSE,
        hclustfun=my_hclust,
        dendrogram="none",
        trace="none",
        density.info="none",
        colsep=1:ncol(data),
        rowsep=1:nrow(data),
        sepcolor="black",
        sepwidth=c(0.005, 0.005),
        cexCol=1,
        cexRow=1,
        srtCol=45,

        #key=key,
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
                                        #breaks <- breaks[c(1,length(breaks))]
            list(at = parent.frame()$scale01(breaks),
                 labels = breaks)},

        margins=c(1,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        col=my_palette,
        breaks=my_breaks,
        useRaster=FALSE

        )

}

# args
args <- commandArgs(trailingOnly=TRUE)
pwm_traj_presence_file <- args[1]
pwm_patterns_file <- args[2]
tf_traj_presence_file <- args[3]
tf_patterns_file <- args[4]

# read motif files and adjust as necessary
pwm_traj_presence <- read.table(pwm_traj_presence_file, header=TRUE, row.names=1, sep="\t")
rownames(pwm_traj_presence) <- gsub("HCLUST-\\d+_", "", rownames(pwm_traj_presence))
rownames(pwm_traj_presence) <- gsub(".UNK.0.A", "", rownames(pwm_traj_presence))

pwm_patterns <- read.table(pwm_patterns_file, header=TRUE, row.names=1, sep="\t")
rownames(pwm_patterns) <- gsub("HCLUST-\\d+_", "", rownames(pwm_patterns))
rownames(pwm_patterns) <- gsub(".UNK.0.A", "", rownames(pwm_patterns))
pwm_patterns <- pwm_patterns / apply(pwm_patterns, 1, max)
#pwm_patterns <- apply(pwm_patterns, 1, function(x) x/sum(x))
#pwm_patterns_norm <- t(scale(t(pwm_patterns))) # TODO check this
#rownames(pwm_patterns_norm) <- rownames(pwm_patterns)
#pwm_patterns <- pwm_patterns_norm

# motif ordering
pwm_dists <- dist(cbind(pwm_traj_presence, t(scale(t(pwm_patterns)))))
hc <- my_hclust(pwm_dists)

# manual reorder top and bottom first cut
hc_dend <- as.dendrogram(hc)
hc_dend[[1]] <- rev(hc_dend[[1]])
hc_dend[[2]] <- rev(hc_dend[[2]])

# and get ordering and apply
pwm_ordering <- order.dendrogram(hc_dend)
pwm_traj_presence <- pwm_traj_presence[pwm_ordering,]
pwm_patterns <- pwm_patterns[pwm_ordering,]

# plot
pwm_traj_presence_plot_file <- gsub(".txt", ".pdf", pwm_traj_presence_file)
my_palette <- colorRampPalette(brewer.pal(9, "Purples"))(49)
pdf(pwm_traj_presence_plot_file, height=13, width=3)
make_heatmap(pwm_traj_presence, my_palette, NULL)
dev.off()

my_breaks <- quantile(melt(pwm_patterns)$value, probs=seq(0.10, 1, length.out=30))
my_breaks <- my_breaks[!duplicated(my_breaks)]

pwm_patterns_plot_file <- gsub(".txt", ".pdf", pwm_patterns_file)
my_palette <- colorRampPalette(brewer.pal(9, "Oranges")[1:8])(length(my_breaks)-1)
pdf(pwm_patterns_plot_file, height=13, width=3)
make_heatmap(pwm_patterns, my_palette, my_breaks)
dev.off()


# read RNA files and adjust as necessary
tf_traj_presence <- read.table(tf_traj_presence_file, header=TRUE, row.names=1, sep="\t")
rownames(tf_traj_presence) <- gsub("HCLUST-\\d+_", "", rownames(tf_traj_presence))
rownames(tf_traj_presence) <- gsub(".UNK.0.A", "", rownames(tf_traj_presence))

tf_patterns <- read.table(tf_patterns_file, header=TRUE, row.names=1, sep="\t")
rownames(tf_patterns) <- gsub("HCLUST-\\d+_", "", rownames(tf_patterns))
rownames(tf_patterns) <- gsub(".UNK.0.A", "", rownames(tf_patterns))
tf_patterns <- tf_patterns / apply(tf_patterns, 1, max)

#tf_patterns_norm <- t(scale(t(tf_patterns))) # TODO check this
#rownames(tf_patterns_norm) <- rownames(tf_patterns)
#tf_patterns <- tf_patterns_norm

# tf ordering
tf_dists <- dist(cbind(tf_traj_presence, t(scale(t(tf_patterns)))))
hc <- my_hclust(tf_dists)
tf_ordering <- hc$order
tf_traj_presence <- tf_traj_presence[tf_ordering,]
tf_patterns <- tf_patterns[tf_ordering,]

# plot
tf_traj_presence_plot_file <- gsub(".txt", ".pdf", tf_traj_presence_file)
my_palette <- colorRampPalette(brewer.pal(9, "Purples"))(49)
pdf(tf_traj_presence_plot_file, height=13, width=3)
make_heatmap(tf_traj_presence, my_palette, NULL)
dev.off()

my_breaks <- quantile(melt(tf_patterns)$value, probs=seq(0.10, 1, length.out=20))
my_breaks <- my_breaks[!duplicated(my_breaks)]

tf_patterns_plot_file <- gsub(".txt", ".pdf", tf_patterns_file)
my_palette <- colorRampPalette(brewer.pal(9, "Blues"))(length(my_breaks)-1)
pdf(tf_patterns_plot_file, height=13, width=3)
make_heatmap(tf_patterns, my_palette, my_breaks)
dev.off()
