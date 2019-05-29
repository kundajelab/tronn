#!/usr/bin/env Rscript

# description: get the traj heatmap from a model, just test set
library(rhdf5)
library(gplots)
library(RColorBrewer)
library(reshape2)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
labels_plot_file <- args[2]
logits_plot_file <- args[3]

# traj reorder vector
traj_reorder_cols <- c(1,8,9,10,11,12,13,14,15,2,3,4,5,6,7)

# load in datasets
labels <- h5read(h5_file, "ATAC_SIGNALS.NORM")
logits <- h5read(h5_file, "logits")
traj_labels <- h5read(h5_file, "TRAJ_LABELS")

# aperm
labels <- data.frame(aperm(labels))
logits <- data.frame(aperm(logits))
traj_labels <- data.frame(aperm(traj_labels))

# pull specific columns
cols <- c(1,2,3,4,5,6,7,10,11,13)
labels <- labels[,cols]
logits <- logits[,cols]
traj_labels <- traj_labels[,traj_reorder_cols]

# remove nondynamic
labels <- labels[rowSums(traj_labels)!=0,]
logits <- logits[rowSums(traj_labels)!=0,]
traj_labels <- traj_labels[rowSums(traj_labels)!=0,]

print(head(rowSums(traj_labels)))

#print(dim(labels))
#print(dim(logits))
#print(dim(traj_labels))

# order the rows
#print(head(traj_labels))
#rownames(traj_labels) <- 1:nrow(traj_labels)
ordering <- order(
	 -traj_labels[[1]],
	 -traj_labels[[2]],
	 -traj_labels[[3]],
	 -traj_labels[[4]],
	 -traj_labels[[5]],
	 -traj_labels[[6]],
	 -traj_labels[[7]],
	 -traj_labels[[8]],
	 -traj_labels[[9]],
	 -traj_labels[[10]],
	 -traj_labels[[11]],
	 -traj_labels[[12]],
	 -traj_labels[[13]],
	 -traj_labels[[14]],
	 -traj_labels[[15]])

labels <- labels[ordering,]
logits <- logits[ordering,]

# normalize
labels <- labels - labels[,1]
logits <- logits - logits[,1]
diff <- labels - logits

# clip
thresholds <- quantile(melt(labels)$value, c(0.05, 0.95))
labels[labels < thresholds[1]] <- thresholds[1]
labels[labels > thresholds[2]] <- thresholds[2]

thresholds <- quantile(melt(logits)$value, c(0.05, 0.95))
logits[logits < thresholds[1]] <- thresholds[1]
logits[logits > thresholds[2]] <- thresholds[2]

# subsample but ordered
sample_indices <- seq(1, nrow(labels), length.out=1000)
labels_subsample <- labels[sample_indices,]
logits_subsample <- logits[sample_indices,]

# plot
plot_heatmap <- function(data_z) {
my_palette <- rev(colorRampPalette(brewer.pal(11, "RdBu"))(49))

# color bar
#cluster_palette <- colorRampPalette(brewer.pal(11, "Spectral"))(nrow(cluster_means))
#cluster_palette <- get_trajectory_palette(nrow(cluster_means))
#cluster_colors <- cluster_palette[cluster_ids_per_example]
#cluster_colors <- cluster_palette[color_bar_clusters]

# heatmap2 grid
#mylmat = rbind(c(0,0,3,0),c(4,1,2,0),c(0,0,5,0))
#mylwid = c(0.05,0.1,1,0.05)
#mylhei = c(0.25,4,0.5)

mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(0.1,1,0.05)
mylhei = c(0.25,4,0.5)


heatmap.2(
    as.matrix(data_z),
    Rowv=FALSE,
    Colv=FALSE,
    dendrogram="none",
    trace='none',
    density.info="none",
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
    labRow="",
    margins=c(1,0),
    col=my_palette,
    lmat=mylmat,
    lwid=mylwid,
    lhei=mylhei)
    #rowsep=rowsep,
    #sepcolor="black")
    #RowSideColors=cluster_colors)

}

pdf(labels_plot_file, height=7, width=2, family="ArialMT")
plot_heatmap(labels_subsample)
dev.off()

pdf(logits_plot_file, height=7, width=2, family="ArialMT")
plot_heatmap(logits_subsample)
dev.off()


quit()