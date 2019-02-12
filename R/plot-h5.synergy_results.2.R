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
out_prefix <- args[6]
aux_keys <- args[7:length(args)]

# read in data
data <- h5read(h5_file, synergy_scores_key, read.attributes=TRUE)
diff_sig <- h5read(h5_file, diff_sig_key)
labels <- attr(data, "labels")
num_tasks <- dim(data)[1]
print(dim(diff_sig))

# run for every tasks
for (task_idx in 1:length(num_tasks)) {

    # set up task data
    task_prefix <- paste(out_prefix, ".taskidx-", task_idx, sep="")
    task_data <- data.frame(aperm(data)[,,task_idx]) # {N, syn}
    colnames(task_data) <- c("a", "b")
    task_data$diff <- aperm(diff_sig)[,task_idx]
    print(head(task_data))

    # plot comparison between fold changes
    plot_file <- paste(out_prefix, ".fc_compare.pdf", sep="")
    ggplot(task_data, aes(x=b, y=a)) +
        geom_point(aes(colour = factor(diff))) + 
        labs(y=labels[1], x=labels[2]) +
        theme_bw() +
        theme(
            axis.title=element_text(size=32),
            axis.text.y=element_text(size=16),
            axis.text.x=element_text(size=12, angle=30, hjust=1),
            legend.title=element_text(size=32),
            legend.text=element_text(size=12),
            legend.position="none")
    ggsave(plot_file, height=7, width=7)
    q()

    # plot with distance
    

}



q()

if (TRUE) {
    other_data <- h5read(h5_file, key2, read.attributes=TRUE)
    #other_data <- data.frame(aperm(other_data)[,task_idx])[,1]
    other_data <- data.frame(aperm(other_data)[,3])[,1]
    print(dim(other_data))
    other_label <- gsub(".", "_", key2, fixed=TRUE)
    print(other_label)
}

index_data <- h5read(h5_file, key3)
index_data <- data.frame(aperm(index_data)[,4,]) # {N, 4, 2}
index_diff <- abs(index_data[,1] - index_data[,2])
#index_diff <- (index_data[,1] - index_data[,2])
#print(head(index_diff))

# plot data
plot_data <- data.frame(
    synergy=diff_data,
    y=other_data,
    diff=index_diff,
    a=index_data[,1],
    b=index_data[,2])
plot_data <- plot_data[plot_data$diff > 12,]
plot_data <- plot_data[plot_data$synergy > 0.0,]
print(head(plot_data))

plot_data <- data.frame(
    synergy=data[,1],
    diff=data[,2])
plot_data$differential <- abs(diff_data) > 2*stdev
print(dim(plot_data))
print(sum(plot_data$differential))

# set up for ggplot
#data_melted <- melt(plot_data)
#print(head(data_melted))
data_melted <- plot_data

ggplot(data_melted, aes(x=diff, y=synergy)) +
    geom_point(aes(colour = factor(differential))) + 
    labs(y=labels[1], x=labels[2]) +
    theme_bw() +
    theme(
        axis.title=element_text(size=32),
        axis.text.y=element_text(size=16),
        axis.text.x=element_text(size=12, angle=30, hjust=1),
        legend.title=element_text(size=32),
        legend.text=element_text(size=12),
        legend.position="none") +
     scale_fill_brewer(palette="Dark2") +
     scale_color_brewer(palette="Dark2")

ggsave(out_file, height=7, width=7)

# and then also plot distance
plot_data <- data.frame(
    synergy=diff_data,
    y=other_data,
    differential=plot_data$differential,
    diff=index_diff,
    a=index_data[,1],
    b=index_data[,2])
plot_data <- plot_data[plot_data$diff > 12,]
#plot_data <- plot_data[plot_data$synergy > 0.0,]

ggplot(plot_data, aes(x=diff, y=synergy)) +
    geom_point(aes(colour = factor(differential))) + 
    labs(y=labels[1], x=labels[2]) +
    theme_bw() +
    theme(
        axis.title=element_text(size=32),
        axis.text.y=element_text(size=16),
        axis.text.x=element_text(size=12, angle=30, hjust=1),
        legend.title=element_text(size=32),
        legend.text=element_text(size=12),
        legend.position="none") +
     scale_fill_brewer(palette="Dark2") +
     scale_color_brewer(palette="Dark2")

ggsave("test.2.pdf", height=7, width=7)





