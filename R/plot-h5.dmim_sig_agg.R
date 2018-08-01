#!/usr/bin/env Rscript

library(rhdf5)
library(reshape2)
library(ggplot2)

# description: plot out master map of dmim results

# ----------------------------------------
# args
# ----------------------------------------
args <- commandArgs(trailingOnly=TRUE)

# key args
h5_file <- args[1]
data_key <- args[2]
pwm_names_attr_key <- args[3]

# read data
data <- h5read(h5_file, data_key, read.attributes=TRUE)
pwm_names <- attr(data, pwm_names_attr_key)

pwm_names <- gsub("HCLUST-.*_", "", pwm_names)
pwm_names <- gsub(".UNK.*", "", pwm_names)

# set the dimnames in prep for melt
data_dimnames <- list(
    "mutated_motif"=pwm_names,
    "task"=1:dim(data)[2],
    "response_motif"=pwm_names)
dimnames(data) <- data_dimnames
data_melted <- melt(data)

# normalize



# plot with ggplot
p <- ggplot(data_melted, aes(x=task, y=response_motif)) +
    facet_grid(response_motif ~ mutated_motif, scales="free", space="free_x") +
        geom_tile(aes(fill=value))

ggsave("test.pdf")
   




