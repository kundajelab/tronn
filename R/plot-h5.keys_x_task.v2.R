#!/usr/bin/env Rscript

library(rhdf5)

# description: given keys, aggregate info and plot
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
data_strings <- unlist(strsplit(data_key, ",", fixed=TRUE))
clusters <- h5read(h5_file, cluster_key, read.attributes=TRUE)
cluster_ids <- attr(clusters, cluster_id_attr_key)

# ----------------------------------------
# plot by cluster
# ----------------------------------------

for (cluster_idx in 1:length(cluster_ids)) {
    cluster_id <- cluster_ids[cluster_idx]
    
    for (key_idx in 1:length(data_strings)) {
        
        # split out the data string
        key_and_indices <- unlist(
            strsplit(data_strings[key_idx], "=", fixed=TRUE))
        key <- key_and_indices[1]
        
        if (length(key_and_indices) > 1) {
            indices <- as.numeric(unlist(
                strsplit(key_and_indices[2], ",", fixed=TRUE))) + 1
        } else {
            indices <- c()
        }
       
        # extract from h5 file
        data <- h5read(h5_file, key, read.attributes=TRUE)
        data <- aperm(data)

        # get cluster
        cluster_data <- get_cluster_data(data, clusters, cluster_id)
        
        # set up indices
        if (length(indices) > 0) {
            cluster_data <- cluster_data[,indices]
            indices_string <- paste(
                "_indices_",
                indices[1]-1,
                "-",
                indices[length(indices)]-1, sep="")
        } else {
            indices_string <- ""
        }

        # normalize if needed
        cluster_data <- normalize_rows(cluster_data)

        # aggregate
        data <- colMeans(data)
        
        # append
        if (key_idx == 1) {
            all_data <- data
            key_string <- paste(key, indices_string, sep="")
            data_names <- c(key)
        } else {
            all_data <- rbind(all_data, data)
            key_tmp <- paste(key, indices_string, sep="")
            key_string <- paste(key_string, key_tmp, sep="-")
            data_names <- c(data_names, key)
        }
    }
    
    # add row names
    rownames(all_data) <- data_names
    
    # make heatmap
    heatmap_file <- paste(
        sub(".h5", "", h5_file),
        key_string,
        cluster_key,
        "pdf", sep=".")
    print(heatmap_file)
    make_agg_heatmap(all_data, heatmap_file, FALSE)
}
