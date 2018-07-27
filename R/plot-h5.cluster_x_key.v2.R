#!/usr/bin/env Rscript

library(rhdf5)

# description: given a key, aggregate per cluster per task
# NOTE: rhdf5 transposes axes! adjust accordingly
# USEFUL: h5ls(h5_file) to list keys

# ----------------------------------------
# load relevant functions
# ----------------------------------------
load_h5_data_fn <- system("which load_h5_data_fn.R", intern=TRUE)
source(load_h5_data_fn)

load_cluster_fns <- system("which clustering_fns.R", intern=TRUE)
source(load_cluster_fns)

load_normalization_fns <- system("which normalization_fns.R", intern=TRUE)
source(load_normalization_fns)

load_heatmap_fns <- system("which heatmap_fns.R", intern=TRUE)
source(load_heatmap_fns)

# ----------------------------------------
# read data
# ----------------------------------------
data <- h5read(h5_file, data_key, read.attributes=TRUE)
data_colnames <- attr(data, colnames_attr_key) # TODO adjust based on available indices
clusters <- h5read(h5_file, cluster_key, read.attributes=TRUE)
cluster_ids <- attr(clusters, cluster_id_attr_key)

# transpose
data <- aperm(data) # {N, ...}
clusters <- aperm(clusters)

# order
data <- order_by_clusters(data, clusters)

# for each cluster, extract keys and aggregate
for (cluster_idx in 1:length(cluster_ids)) {

    # get data
    cluster_id <- cluster_ids[cluster_idx]
    cluster_data <- get_cluster_data(data, clusters, cluster_id)
    
    # aggregate
    cluster_data <- apply(cluster_data, 2, median)
    
    # and append
    if (cluster_idx == 1) {
        all_data <- cluster_data
    } else {
        all_data <- rbind(all_data, cluster_data)
    }
    
}

# add row and colnames
rownames(all_data) <- cluster_ids

# make heatmap
heatmap_file <- paste(
    sub(".h5", "", h5_file),
    key_string,
    "by_cluster",
    "pdf", sep=".")
print(heatmap_file)
make_agg_heatmap(all_data, heatmap_file, FALSE)
