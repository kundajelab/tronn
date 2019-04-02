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
    print(task_prefix)
    task_data <- data.frame(
        dists=distances,
        logits=aperm(logits)[,task_idx],
        groups=aperm(groups)[,1])

    #print(dim(task_data))
    
    # seprate odds (embedded) from evens (non-embedded) and subtract
    task_data_embed <- task_data[seq(1,nrow(task_data)-1,2),]
    task_data_orig <- task_data[seq(2,nrow(task_data),2),]
    task_data_embed$diff <- task_data_embed$logits - task_data_orig$logits
    task_data <- task_data_embed
    #task_data <- task_data_orig

    #print(task_data_embed[1:10,])
    #print(task_data_orig[1:100,])
    
    #print(task_data[1:100,])
    #quit()
    #print(dim(task_data_embed))
    #print(head(task_data_embed))
    #task_data <- task_data[task_data$groups == "HCLUST-110_GRHL2.UNK.0.A+;HCLUST-105_ATF4.UNK.0.A+",]
    
    # plot with distance
    plot_file <- paste(task_prefix, ".dist_x_logFC.pdf", sep="")
    p <- ggplot(task_data, aes(x=dists, y=diff)) +
        geom_point(aes(colour=factor(groups)), alpha=0.5) +
        #geom_point(position="jitter", aes(colour=factor(groups)), alpha=0.5) +
        geom_smooth(se=FALSE, aes(group=factor(groups), colour=factor(groups))) +
        #stat_summary(aes(y=logits, group=factor(groups), colour=factor(groups)), fun.y=mean, geom="line") + 
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
