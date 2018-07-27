#!/usr/bin/env Rscript

library(rhdf5)

# description: plot summary of grammar and mutagenesis
# NOTE: rhdf5 transposes axes!

# ----------------------------------------
# load relevant functions
# ----------------------------------------
load_heatmap_fns <- system("which heatmap_fns.R", intern=TRUE)
source(load_heatmap_fns)

# ----------------------------------------
# args
# ----------------------------------------
args <- commandArgs(trailingOnly=TRUE)

# key args
h5_file <- args[1]
agg_data_key <- args[2]
pwm_names_attr_key <- args[3]
# TODO save cluster ids to dataset

# read in data
data <- h5read(h5_file, agg_data_key, read.attributes=TRUE) # {motif, task, cluster}
print(dim(data))

# get pwm names
pwm_names <- gsub(".UNK.*", "", attr(data, pwm_names_attr_key))
pwm_names <- gsub("HCLUST-.*_", "", pwm_names)


# TODO eventually set up a colnames key
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


# set up a motif ordering
flattened <- data
dim(flattened) <- c(dim(data)[1], dim(data)[2]*dim(data)[3])
hc <- hclust(dist(flattened), method="ward.D2")
ordering <- hc$order
print(ordering)

# first plot global
data_global <- apply(data, c(1,2), mean) # {motif, cluster}
print(dim(data))
print(dim(data_global))
rownames(data_global) <- pwm_names
colnames(data_global) <- days
data_global <- data_global[ordering,]

plot_file <- paste(
    sub("h5", "", h5_file),
    agg_data_key,
    "global",
    ".pdf", sep="")        
print(plot_file)
make_agg_heatmap(data_global, plot_file, FALSE)

# then plot clusters
num_clusters <- dim(data)[3]
for (i in 1:num_clusters) {
    
    # extract the data
    cluster_data <- data[,,i]
    rownames(cluster_data) <- pwm_names
    colnames(cluster_data) <- days
    cluster_data <- cluster_data[ordering,]

    plot_file <- paste(
        sub("h5", "", h5_file),
        agg_data_key,
        ".clusteridx-", i-1, ".pdf", sep="")        
    print(plot_file)
    make_agg_heatmap(cluster_data, plot_file, FALSE)
    
}

