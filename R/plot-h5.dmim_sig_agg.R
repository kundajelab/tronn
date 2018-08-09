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
    "response_motif"=pwm_names,
    "task"=1:dim(data)[2],
    "mutated_motif"=pwm_names)
dimnames(data) <- data_dimnames
data_melted <- melt(data)

# normalize
if (TRUE) {
    max_cutoff <- quantile(abs(data_melted$value), 0.99)
    data <- data / max_cutoff
    data[data < -1.0] <- -1.0
    data[data > 1.0] <- 1.0
    data_melted <- melt(data)
}

# ordering by mutational similarity
if (TRUE) {
    flattened <- data
    dim(flattened) <- c(dim(data)[1], dim(data)[2]*dim(data)[3])
    hc <- hclust(dist(flattened), method="ward.D2")
    ordering <- hc$order
    data_melted$mutated_motif <- factor(
        data_melted$mutated_motif,
        levels=pwm_names[ordering])
    data_melted$response_motif <- factor(
        data_melted$response_motif,
        levels=pwm_names[ordering])
}
    
# plot with ggplot
p <- ggplot(data_melted, aes(x=task, y=response_motif)) +
    facet_grid(response_motif ~ mutated_motif, scales="free", space="free_x", switch="y") +
    geom_tile(aes(fill=value)) +
    #scale_fill_gradient(high="white", low="steelblue") +
                                        #scale_fill_gradient(high="steelblue", low="white") +
    scale_fill_gradient2(low="steelblue", mid="white", high="red", midpoint=0) +
    theme_bw() +
    theme(
        axis.text.y=element_blank(),
        strip.text.y=element_text(angle=180),
        strip.background=element_blank(),
        panel.spacing.x=unit(1, "lines"),
        panel.spacing.y=unit(0.1, "lines"))

        
ggsave("test.pdf", width=15, height=4)
   




