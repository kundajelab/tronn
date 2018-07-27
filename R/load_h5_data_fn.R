#!/usr/bin/env Rscript

library(rhdf5)

# description: load data from h5
# and transform as is necessary

# ----------------------------------------
# args
# ----------------------------------------
args <- commandArgs(trailingOnly=TRUE)

# key args
h5_file <- args[1]
data_key <- args[2]
cluster_key <- args[3]

# attr args
cluster_id_attr_key <- args[4]
colnames_attr_key <- args[5]

# params args
three_dims <- as.logical(as.numeric(args[6]))
cluster_columns <- as.logical(as.numeric(args[7]))
row_normalize <- as.logical(as.numeric(args[8]))
signal_normalize <- as.logical(as.numeric(args[9]))
large_view <- as.logical(as.numeric(args[10]))
use_raster <- as.logical(as.numeric(args[11]))

# indices on data
if (length(args) > 11) {
    data_indices <- as.numeric(
        unlist(strsplit(args[12], ",", fixed=TRUE)))
    indices_string <- paste(
        "indices_",
        indices[1]-1,
        "-",
        indices[length(indices)]-1, sep="")
    key_string <- paste(
        data_key,
        indices_string,
        sep=".")
} else {
    data_indices <- c()
    key_string <- data_key
}









