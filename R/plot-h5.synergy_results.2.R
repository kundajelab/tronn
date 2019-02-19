#!/usr/bin/env Rscript

# description - code to plot out synergy scores
library(rhdf5)
library(reshape2)
library(ggplot2)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
synergy_scores_key <- args[2] # {N, syn, task}
diff_key <- args[3]
diff_sig_key <- args[4]
dists_key <- args[5] # {N}
max_dist_key <- args[6]
out_prefix <- args[7]
sample_key <- args[8]
aux_keys <- args[9:length(args)]

# read in data
data <- h5read(h5_file, synergy_scores_key, read.attributes=TRUE)
diffs <- h5read(h5_file, diff_key)
diff_sig <- h5read(h5_file, diff_sig_key)
dists <- h5read(h5_file, dists_key)
max_dist <- as.numeric(h5read(h5_file, max_dist_key))
labels <- attr(data, "labels")
num_tasks <- dim(data)[1]
if (nchar(sample_key) > 0) {
    sample_bool <- h5read(h5_file, sample_key)
}

# run for every tasks
for (task_idx in 1:length(num_tasks)) {

    # set up task data
    task_prefix <- paste(out_prefix, ".taskidx-", task_idx, sep="")
    task_data <- data.frame(aperm(data)[,,task_idx]) # {N, syn}
    colnames(task_data) <- c("a", "b")
    task_data$diffs <- aperm(diffs)[,task_idx]
    task_data$diff_sig <- aperm(diff_sig)[,task_idx]
    task_data$dists <- abs(dists)

    if (nchar(sample_key) > 0) {
        task_data$sample <- sample_bool
    }

    # remove dists < 12 (overlap in pwms)
    task_data <- task_data[task_data$dists > 12,]
    
    # plot comparison between fold changes
    plot_file <- paste(out_prefix, ".fc_compare.pdf", sep="")
    p <- ggplot(task_data, aes(x=b, y=a)) +
        geom_point(data=subset(task_data, diff_sig==2), colour="black") +
        geom_point(data=subset(task_data, diff_sig==1), colour="gray") +
        labs(y=labels[1], x=labels[2]) +
        theme_bw() +
        theme(
            axis.title=element_text(size=32),
            axis.text.y=element_text(size=16),
            axis.text.x=element_text(size=12, angle=30, hjust=1),
            legend.title=element_text(size=32),
            legend.text=element_text(size=12),
            legend.position="none")
    if (nchar(sample_key) > 0) {
        p + geom_point(data=subset(task_data, sample==1), colour="red")
    }
    ggsave(plot_file, height=7, width=7)
    
    # plot with distance
    plot_file <- paste(out_prefix, ".dist_x_diff.pdf", sep="")
    p <- ggplot(task_data, aes(x=dists, y=diffs)) +
        geom_point(data=subset(task_data, diff_sig==2), colour="black") +
        geom_point(data=subset(task_data, diff_sig==1), colour="gray") +
        geom_vline(xintercept=max_dist, colour="gray", linetype="dashed") +
        labs(y="synergy", x="PWM distance (bp)") +
        theme_bw() +
        theme(
            axis.title=element_text(size=32),
            axis.text.y=element_text(size=16),
            axis.text.x=element_text(size=12, angle=30, hjust=1),
            legend.title=element_text(size=32),
            legend.text=element_text(size=12),
            legend.position="none")
    if (nchar(sample_key) > 0) {
        p + geom_point(data=subset(task_data, sample==1), colour="red")
    }
    ggsave(plot_file, height=7, width=7)

}
