#!/usr/bin/env Rscript

# description: plot out variants against each other
library(rhdf5)
library(ggplot2)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
out_prefix <- args[2]
key <- "logits"
sig_key <- "variants.sig"

# read in data
data <- h5read(h5_file, key)
sig <- h5read(h5_file, sig_key)
num_tasks <- dim(data)[1]
print(num_tasks)

for (task_idx in 1:num_tasks) {

    # set up task data
    task_data <- data.frame(aperm(data)[,,task_idx])
    colnames(task_data) <- c("ref", "alt")
    task_sig <- data.frame(aperm(sig)[,task_idx])
    task_data$diff <- task_sig[,1]
    
    # plot
    plot_file <- paste(out_prefix, ".variants.", task_idx, ".pdf", sep="")
    ggplot(task_data, aes(x=ref, y=alt)) +
        geom_point(data=subset(task_data, diff==2), colour="black") +
        geom_point(data=subset(task_data, diff==1), colour="grey")
    ggsave(plot_file, height=7, width=7)

}

