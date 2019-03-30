#!/usr/bin/env Rscript

# description - code to plot out synergy scores
library(rhdf5)
library(reshape2)
library(ggplot2)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
dist_key <- args[2]
logits_key <- args[3]
group_key <- args[4]
out_prefix <- args[5]

# read in data
distances <- h5read(h5_file, dist_key)
logits <- h5read(h5_file, logits_key)
groups <- h5read(h5_file, group_key)
num_tasks <- dim(logits)[1]

print(levels(factor(groups)))

# run for every tasks
for (task_idx in 1:num_tasks) {

    # set up task data
    task_prefix <- paste(out_prefix, ".taskidx-", task_idx-1, sep="")
    task_data <- data.frame(
        dists=distances,
        logits=aperm(logits)[,task_idx],
        groups=aperm(groups)[,1])

    print(task_data[1:20,])
    
    # plot with distance
    plot_file <- paste(task_prefix, ".dist_x_diff.pdf", sep="")
    p <- ggplot(task_data, aes(x=dists, y=logits)) +
        geom_point(position="jitter", aes(colour=factor(groups))) +
        stat_summary(aes(y=logits, group=factor(groups), colour=factor(groups)), fun.y=mean, geom="line") + 
        labs(y="logit", x="PWM distance (bp)") +
        theme_bw() +
        theme(
            axis.title=element_text(size=32),
            axis.text.y=element_text(size=16),
            axis.text.x=element_text(size=12, angle=30, hjust=1),
            legend.title=element_text(size=32),
            legend.text=element_text(size=12))
            #legend.position="none")
    ggsave(plot_file, height=7, width=14)

}
