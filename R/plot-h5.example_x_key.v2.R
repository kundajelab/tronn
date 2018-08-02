#!/usr/bin/env Rscript

library(rhdf5)

# description: plot example x key from hdf5 file
# NOTE: rhdf5 transposes axes! adjust accordingly.
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

# ----------------------------------------
# plot data
# ----------------------------------------
data_sample <- get_ordered_sample(data)

# visualize
if (three_dims) {
    
    # split up at axis and view
    num_slices <- dim(data_sample)[2]
    for (i in 1:num_slices) {
        slice_data <- data_sample[,i,]
        colnames(slice_data) <- data_colnames

        if (length(data_indices) > 0) {
            slice_data <- slice_data[,data_indices]
        }
        print(dim(slice_data))

        # normalize
        if (row_normalize) {
            slice_data <- normalize_rows(slice_data)
        }
        
        heatmap_file <- paste(
            sub(".h5", "", h5_file),
            key_string,
            cluster_key,
            paste("slice", i-1, sep="-"), # to match 0-START from python
            "pdf", sep=".")
        print(heatmap_file)
        make_heatmap(slice_data, heatmap_file, cluster_columns, signal_normalize, large_view, use_raster)

    }
    
} else {
    colnames(data_sample) <- data_colnames

    if (length(data_indices) > 0) {
        data_sample <- data_sample[,data_indices]
    }
    print(dim(data_sample))
    
    heatmap_file <- paste(
        sub(".h5", "", h5_file),
        key_string,
        cluster_key,
        "pdf", sep=".")
    print(heatmap_file)
    make_heatmap(data_sample, heatmap_file, cluster_columns, signal_normalize, large_view, use_raster)

}
