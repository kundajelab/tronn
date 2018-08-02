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
max_cutoff <- quantile(abs(data_melted$value), 0.99)
data_melted$value <- data_melted$value / max_cutoff
data_melted$value[data_melted$value < -1.0] <- -1.0

# plot with ggplot
p <- ggplot(data_melted, aes(x=task, y=response_motif)) +
    facet_grid(response_motif ~ mutated_motif, scales="free", space="free_x", switch="y") +
    geom_tile(aes(fill=value)) +
    scale_fill_gradient(high="white", low="steelblue") +
    theme_bw() +
    theme(
        axis.text.y=element_blank(),
        strip.text.y=element_text(angle=180),
        strip.background=element_blank(),
        panel.spacing.x=unit(1, "lines"),
        panel.spacing.y=unit(0.1, "lines"))

        
ggsave("test.pdf", width=15, height=4)
   




